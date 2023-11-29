// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::compute::kernels::take::take;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::IpcWriteOptions;
use arrow::ipc::CompressionType;
use arrow::pyarrow::ToPyArrow;
use arrow_array::builder::PrimitiveBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, UInt16Type, UInt32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt16Array, UInt32Array, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema as ArrowSchema};
use either::Either;
use futures::StreamExt;
use lance::dataset::Dataset as LanceDataset;
use lance_core::ROW_ID;
use num_traits::NumCast;
use pyo3::exceptions::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyBytes, PyDict, PySlice, PySliceIndices, PyString};
use pyo3::{intern, prelude::*};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::{Dataset, RT};

/// How a dataset should be sampled.
///
/// This struct has a sibling dataclass in python/lance/sampler.py.
///
/// Intuitively, sampling follows these steps:
/// 1. Apply the filter to the dataset.
/// 2. Sample the rows that pass the filter.
/// 3. Batch the sampled rows.
/// 4. Shuffle the batches.
#[derive(Clone, Serialize, Deserialize, FromPyObject, PartialEq)]
pub struct SampleParams {
    /// A filter to apply before sampling.
    predicate: Option<String>,
    /// The maximum number of rows to group in a batch before shuffling. This
    /// keeps rows that are close together in the same batch, for more efficient
    /// IO.
    batch_size: usize,
    /// Whether to randomize the order of rows in the dataset.
    shuffle: bool,
    /// What proportion of rows to sample. This is applied after the filter.
    sample_rate: Option<f32>,
    /// The seed to use for shuffling and sampling.
    seed: Option<u64>,
}

impl IntoPy<PyObject> for SampleParams {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let constructor = py
            .import(intern!(py, "lance.sampler"))
            .expect("module lance.sampler not found.")
            .getattr(intern!(py, "SampleParams"))
            .expect("lance.sampler.SampleParams not found.");
        constructor
            .call1((
                self.predicate.clone(),
                self.batch_size,
                self.shuffle,
                self.sample_rate,
                self.seed,
            ))
            .unwrap()
            .into()
    }
}

/// Metrics about a sample.
///
/// This struct has a sibling NamedTuple in python/lance/sampler.py
#[derive(Debug, Clone, FromPyObject, Serialize, Deserialize, PartialEq)]
pub struct SampleMetrics {
    pub dataset_size: usize,
    pub matched_rows: usize,
    pub sampled_rows: usize,
}

impl SampleMetrics {
    pub fn new(dataset_size: usize) -> Self {
        Self {
            dataset_size,
            matched_rows: 0,
            sampled_rows: 0,
        }
    }
}

impl IntoPy<PyObject> for SampleMetrics {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let constructor = py
            .import(intern!(py, "lance.sampler"))
            .expect("module lance.sampler not found.")
            .getattr(intern!(py, "SampleMetrics"))
            .expect("lance.sampler.SampleMetrics not found.");
        constructor
            .call1((self.dataset_size, self.matched_rows, self.sampled_rows))
            .unwrap()
            .into()
    }
}

/// A sample of batches of a dataset.
///
/// These batches might be sampled and their order randomized.
#[pyclass(module = "lance.sampler")]
#[derive(Clone)]
pub struct DatasetSample {
    /// The original parameters used to construct the sample.
    #[pyo3(get)]
    params: SampleParams,
    /// A mask of row positions that pass the filter
    // Note: This could be much more efficient once they implement:
    // https://github.com/RoaringBitmap/roaring-rs/issues/12
    row_id_mask: Option<Arc<RoaringTreemap>>,
    /// Batches of contiguous row positions, represented as [start, end) pairs. These
    /// batches are potentially shuffled. The ranges also may contain rows that
    /// don't pass the filter, so the mask should be used to filter them out.
    ///
    /// If the row count in the dataset < u32::MAX, then this is a UInt32Array.
    /// Otherwise, it is a UInt64Array.
    batch_starts: Arc<dyn Array>,
    batch_lengths: Arc<UInt16Array>,
    #[pyo3(get)]
    metrics: SampleMetrics,
}


impl DatasetSample {
    fn get_start(&self, i: usize) -> PyResult<usize> {
        if i >= self.batch_starts.len() {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of bounds for DatasetSample of length {}",
                i,
                self.batch_starts.len()
            )));
        }

        match self.batch_starts.data_type() {
            DataType::UInt32 => {
                let start = self.batch_starts.as_primitive::<UInt32Type>().value(i);
                Ok(start as usize)
            }
            DataType::UInt64 => {
                let start = self.batch_starts.as_primitive::<UInt64Type>().value(i);
                Ok(start as usize)
            }
            _ => Err(PyTypeError::new_err(format!(
                "batch_starts must be UInt32Array or UInt64Array, but was {}",
                self.batch_starts.data_type()
            ))),
        }
    }

    fn num_rows(&self) -> u64 {
        if let Some(mask) = self.row_id_mask.as_ref() {
            mask.len()
        } else {
            self.batch_lengths.values().iter().map(|v| *v as u64).sum()
        }
    }
}

#[derive(FromPyObject)]
enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

#[pymethods]
impl DatasetSample {
    /// The number of rows in the sample.
    #[getter(num_rows)]
    fn num_rows_py(&self) -> u64 {
        self.num_rows()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let mut repr = String::new();
        repr.push_str("DatasetSample(params=");

        let params_repr = self
            .params
            .clone()
            .into_py(py)
            .call_method0(py, intern!(py, "__repr__"))
            .unwrap()
            .extract::<String>(py)
            .unwrap();
        repr.push_str(&params_repr);

        repr.push_str(", row_id_mask=");

        if let Some(mask) = &self.row_id_mask {
            repr.push_str(&format!("Mask<n={}>", mask.len(),));
        } else {
            repr.push_str("None");
        }

        repr.push_str(", batch_starts=");
        repr.push_str(&format!("{:?}", self.batch_starts));

        repr.push_str(", batch_lengths=");
        repr.push_str(&format!("{:?}", self.batch_lengths));

        repr.push_str(", metrics=");
        let metrics_repr = self
            .metrics
            .clone()
            .into_py(py)
            .call_method0(py, intern!(py, "__repr__"))
            .unwrap()
            .extract::<String>(py)
            .unwrap();
        repr.push_str(&metrics_repr);

        repr.push(')');

        repr
    }

    /// Get the number of batches in the sample.
    fn __len__(&self) -> usize {
        self.batch_starts.len()
    }

    fn __iter__(&self) -> DatasetSampleIter {
        DatasetSampleIter::new(self.clone())
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.params == other.params
                && self.row_id_mask == other.row_id_mask
                && self.batch_starts.as_ref() == other.batch_starts.as_ref()
                && self.batch_lengths == other.batch_lengths
                && self.metrics == other.metrics),
            _ => Err(PyNotImplementedError::new_err("Only == is supported.")),
        }
    }

    fn __getitem__(&self, py: Python<'_>, idx: SliceOrInt) -> PyResult<PyObject> {
        match idx {
            SliceOrInt::Slice(slice) => {
                let PySliceIndices {
                    start,
                    stop,
                    step,
                    slicelength,
                } = slice.indices(self.__len__() as i64)?;
                let start = start as usize;
                let stop = stop as usize;
                let length = slicelength as usize;

                let (batch_starts, batch_lengths) = if step == 1 {
                    (
                        self.batch_starts.slice(start, length),
                        Arc::new(self.batch_lengths.slice(start, length)),
                    )
                } else {
                    let indices = (start as u64..stop as u64)
                        .step_by(step as usize)
                        .collect::<UInt64Array>();
                    let batch_starts = take(&self.batch_starts, &indices, None).map_err(|err| {
                        PyValueError::new_err(format!(
                            "Failed to compute DatasetSample slice: {err}"
                        ))
                    })?;

                    let batch_lengths = self.batch_lengths.as_ref() as &dyn Array;
                    let batch_lengths = take(batch_lengths, &indices, None).map_err(|err| {
                        PyValueError::new_err(format!(
                            "Failed to compute DatasetSample slice: {err}"
                        ))
                    })?;
                    let batch_lengths = batch_lengths
                        .as_any()
                        .downcast_ref::<UInt16Array>()
                        .ok_or_else(|| {
                            PyTypeError::new_err(format!(
                                "batch_lengths must be UInt16Array, but was {}",
                                batch_lengths.data_type()
                            ))
                        })?;

                    (batch_starts, Arc::new(batch_lengths.clone()))
                };

                // Also need to shrink the mask.
                let row_id_mask = if let Some(mask) = self.row_id_mask.as_ref() {
                    let mut range_mask = RoaringTreemap::new();

                    let starts = match batch_starts.data_type() {
                        DataType::UInt32 => Either::Left(
                            batch_starts
                                .as_primitive::<UInt32Type>()
                                .values()
                                .iter()
                                .map(|v| *v as u64),
                        ),
                        DataType::UInt64 => Either::Right(
                            batch_starts
                                .as_primitive::<UInt64Type>()
                                .values()
                                .iter()
                                .copied(),
                        ),
                        _ => {
                            return Err(PyTypeError::new_err(format!(
                                "batch_starts must be UInt32Array or UInt64Array, but was {}",
                                batch_starts.data_type()
                            )))
                        }
                    };
                    for (start, length) in
                        starts.zip(batch_lengths.values().iter().map(|v| *v as u64))
                    {
                        let end = start + length;
                        range_mask.insert_range(start..end);
                    }

                    Some(mask.as_ref() & range_mask)
                } else {
                    None
                };

                let out = Self {
                    params: self.params.clone(),
                    row_id_mask: row_id_mask.map(Arc::new),
                    batch_starts,
                    batch_lengths,
                    // TODO: fix the metrics, or make some warning to the user about them.
                    metrics: self.metrics.clone(),
                };
                Ok(out.into_py(py))
            }
            SliceOrInt::Int(i) => {
                let start = self.get_start(i as usize)? as u64;
                let len = self.batch_lengths.value(i as usize) as u64;
                let end = start + len;
                let indices = (start..end)
                    .filter(|i| {
                        self.row_id_mask
                            .as_ref()
                            .map(|mask| mask.contains(*i))
                            .unwrap_or(true)
                    })
                    .collect::<UInt64Array>();
                Ok(indices.to_data().to_pyarrow(py)?.into_py(py))
            }
        }
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let data = py
            .import(intern!(py, "io"))?
            .getattr(intern!(py, "BytesIO"))?
            .call1((state,))?;
        let new_self = Self::deserialize_from(py, data)?;

        *self = new_self;
        Ok(())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let output = py
            .import(intern!(py, "io"))?
            .getattr(intern!(py, "BytesIO"))?
            .call0()?;
        self.serialize_into(py, output)?;

        Ok(output.call_method0(intern!(py, "getvalue"))?.into_py(py))
    }

    /// Serialize the sample to a file object.
    ///
    /// The data written can be treated as opaque bytes and read back with
    /// :meth:`DatasetSample.deserialize_from`. However, it is written as a
    /// gzip-compressed tarfile with the following structure:
    ///
    /// * ``params.json``: the parameters used to construct the sample.
    /// * ``metrics.json``: the metrics of the sample.
    /// * ``row_id_mask.bin``: the row_id_mask as a binary file.
    /// * ``batches.arrow``: the batch_starts and batch_lengths as an Arrow file.
    ///
    /// You can save this file with a ``.tar.gz`` extension, and it will be
    /// readable by other tools. This can be useful for debugging purposes.
    ///
    /// Parameters
    /// ----------
    /// fileobj : string path or file-like object
    ///     This is the file to write the sample to. If it is a string, it is
    ///     treated as a path to a file. Otherwise, it is treated as a file-like
    ///     object. If it is a file, it must be opened in binary mode.
    fn serialize_into(&self, py: Python<'_>, path_or_file: &PyAny) -> PyResult<()> {
        // open file object as a tarfile in gz mode.
        let archive = tarfile::open(path_or_file, "w:gz")?;

        // params -> params.json
        let params_data = serde_json::to_vec_pretty(&self.params)
            .map_err(|err| PyIOError::new_err(format!("Failed to serialize params: {err}")))?;
        let output = PyBytes::new(py, &params_data);
        tarfile::add_file(&archive, "params.json", &output)?;

        // metrics -> metrics.json
        let metrics_data = serde_json::to_vec_pretty(&self.metrics)
            .map_err(|err| PyIOError::new_err(format!("Failed to serialize params: {err}")))?;
        let output = PyBytes::new(py, &metrics_data);
        tarfile::add_file(&archive, "metrics.json", &output)?;

        // row_id_mask -> row_id_mask.bin
        if let Some(mask) = self.row_id_mask.as_ref() {
            let row_id_data =
                PyBytes::new_with(py, mask.serialized_size(), |bytes: &mut [u8]| {
                    mask.serialize_into(bytes).map_err(|err| {
                        PyIOError::new_err(format!("Failed to serialize row_id_mask: {err}"))
                    })
                })?;
            tarfile::add_file(&archive, "row_id_mask.bin", &row_id_data)?;
        }

        // batch_start, batch_lengths -> batches.arrow
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("batch_start", self.batch_starts.data_type().clone(), false),
                Field::new(
                    "batch_lengths",
                    self.batch_lengths.data_type().clone(),
                    false,
                ),
            ])),
            vec![self.batch_starts.clone(), self.batch_lengths.clone()],
        )
        .map_err(|err| PyIOError::new_err(format!("Failed to create RecordBatch: {err}")))?;
        let batch_data = serialize_record_batch(&batch)
            .map_err(|err| PyIOError::new_err(format!("Failed to serialize RecordBatch: {err}")))?;
        let batch_data = PyBytes::new(py, &batch_data);

        tarfile::add_file(&archive, "batches.arrow", &batch_data)?;

        // Close the tarfile
        archive.call_method0(intern!(py, "close"))?;

        Ok(())
    }

    #[staticmethod]
    fn deserialize_from(py: Python<'_>, path_or_file: &PyAny) -> PyResult<Self> {
        // open file object as a tarfile in gz mode.
        let archive = tarfile::open(path_or_file, "r:gz")?;

        // Read each attribute from the tarfile:
        // params -> params.json
        let params_data = tarfile::read_file(archive, intern!(py, "params.json"))?
            .ok_or_else(|| PyIOError::new_err("Failed to deserialize: params.json is missing"))?;
        let params: SampleParams = serde_json::from_slice(params_data.as_bytes())
            .map_err(|err| PyIOError::new_err(format!("Failed to deserialize params: {err}")))?;

        // metrics -> metrics.json
        let metrics_data = tarfile::read_file(archive, intern!(py, "metrics.json"))?
            .ok_or_else(|| PyIOError::new_err("Failed to deserialize: metrics.json is missing"))?;
        let metrics: SampleMetrics = serde_json::from_slice(metrics_data.as_bytes())
            .map_err(|err| PyIOError::new_err(format!("Failed to deserialize params: {err}")))?;

        // row_id_mask -> row_id_mask.bin
        let row_id_mask = if let Some(data) =
            tarfile::read_file(archive, intern!(py, "row_id_mask.bin"))?
        {
            let row_id_mask = RoaringTreemap::deserialize_from(data.as_bytes()).map_err(|err| {
                PyIOError::new_err(format!("Failed to deserialize row_id_mask: {err}"))
            })?;
            Some(Arc::new(row_id_mask))
        } else {
            None
        };

        // batch_start, batch_lengths -> batches.arrow
        let batch_data = tarfile::read_file(archive, intern!(py, "batches.arrow"))?
            .ok_or_else(|| PyIOError::new_err("Failed to deserialize: batches.arrow is missing"))?;
        let mut batch_reader = FileReader::try_new(Cursor::new(batch_data.as_bytes()), None)
            .map_err(|err| {
                PyIOError::new_err(format!("Failed to create Arrow file reader: {err}"))
            })?;
        let batch = batch_reader
            .next()
            .ok_or_else(|| PyIOError::new_err("Failed to read RecordBatch: no batches found."))?
            .map_err(|err| PyIOError::new_err(format!("Failed to read RecordBatch: {err}")))?;

        let batch_starts = batch
            .column_by_name("batch_start")
            .ok_or_else(|| {
                PyIOError::new_err("Failed to read RecordBatch: no batch_start column.")
            })?
            .clone();
        let batch_lengths = batch
            .column_by_name("batch_lengths")
            .ok_or_else(|| {
                PyIOError::new_err("Failed to read RecordBatch: no batch_lengths column.")
            })?
            .as_primitive::<UInt16Type>()
            .clone();
        let batch_lengths = Arc::new(batch_lengths);

        // Close the tarfile
        Ok(Self {
            params,
            row_id_mask,
            batch_starts,
            batch_lengths,
            metrics,
        })
    }
}


/// A struct that maps row_ids to row positions.
struct DatasetRowIdMapper {
    /// The last bin that was accessed. This is used to speed up consecutive lookups.
    last_i: usize,
    /// A pair of (row id range, position offset) for each batch.
    ranges: Vec<(Range<u64>, u64)>,
}

impl DatasetRowIdMapper {
    fn new(dataset: &LanceDataset) -> PyResult<Self> {
        let mut ranges: Vec<(Range<u64>, u64)> = dataset
            .get_fragments()
            .into_iter()
            .map(|frag| {
                let start = (frag.id() as u64) << 32;
                let physical_rows = frag.metadata().physical_rows.ok_or_else(|| {
                    PyValueError::new_err("Internal failure: expected physical_rows to be set.")
                })?;
                let end = start + physical_rows as u64;
                let row_id_range = start..end;
                Ok((row_id_range, physical_rows as u64))
            })
            .collect::<PyResult<_>>()?;

        let mut next_offset = 0;
        for (_range, offset) in &mut ranges {
            let tmp = *offset;
            *offset = next_offset;
            next_offset += tmp;
        }

        ranges.sort_by_key(|(range, _)| range.start);

        // By default, start in the middle.
        let last_i = ranges.len() / 2;
        Ok(Self { ranges, last_i })
    }

    fn map(&mut self, row_id: u64) -> u64 {
        // Starting with last_i, do binary search to find the range that contains
        // the row_id.
        let mut i = self.last_i;

        // TODO: we don't handle the case where row_id is out of bounds.
        loop {
            let (range, offset) = &self.ranges[i];
            self.last_i = i;
            if row_id < range.start {
                i /= 2;
            } else if row_id >= range.end {
                i = (i + self.ranges.len()) / 2;
            } else {
                return row_id - range.start + offset;
            }
        }
    }
}

async fn compute_mask(
    dataset: &LanceDataset,
    predicate: &str,
    sample_rate: f32,
    metrics: &mut SampleMetrics,
    rng: &mut impl Rng,
) -> PyResult<Arc<RoaringTreemap>> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.filter(predicate).map_err(|err| {
        PyValueError::new_err(format!("Invalid filter predicate in SampleParams: {err}"))
    })?;
    let mut stream = scanner
        .try_into_stream()
        .await
        .map_err(|err| PyIOError::new_err(err.to_string()))?;

    let mut row_id_mapper = DatasetRowIdMapper::new(dataset)?;

    let mut row_id_mask = RoaringTreemap::new();
    while let Some(res) = stream.next().await {
        let batch = res.map_err(|err| PyIOError::new_err(err.to_string()))?;
        metrics.matched_rows += batch.num_rows();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| PyIOError::new_err("Internal failure: expected _rowid column."))?
            .as_primitive::<UInt64Type>();
        if sample_rate >= 1.0 {
            for row_id in row_ids.values() {
                row_id_mask.insert(row_id_mapper.map(*row_id));
            }
            metrics.sampled_rows += row_ids.len();
        } else {
            for i in 0..batch.num_rows() {
                let row_id = row_ids.value(i);
                if rng.gen::<f32>() < sample_rate {
                    row_id_mask.insert(row_id);
                    metrics.sampled_rows += 1;
                }
            }
        }
    }

    Ok(Arc::new(row_id_mask))
}

#[pyfunction(name = "_build_shuffle_sample")]
pub fn build_shuffle_sample(dataset: &Dataset, params: SampleParams) -> PyResult<DatasetSample> {
    let dataset = dataset.ds.clone();
    RT.spawn(None, async move {
        build_shuffle_sample_impl(dataset, params).await
    })?
}

async fn build_shuffle_sample_impl(
    dataset: Arc<LanceDataset>,
    mut params: SampleParams,
) -> PyResult<DatasetSample> {
    let dataset_size = dataset.count_rows().await.map_err(|err| {
        PyIOError::new_err(format!(
            "Failed to get number of dataset rows while building DatasetSample: {err}"
        ))
    })?;

    // To allow users to reproduce the sample, we save the seed to the parameters.
    if params.seed.is_none() {
        params.seed = Some(rand::thread_rng().gen());
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(params.seed.unwrap());

    // To save space, we use u32 indices if possible.
    let large_indices = dataset_size > u32::MAX as usize;
    let mut metrics = SampleMetrics::new(dataset_size);

    let (row_id_mask, mut batch_starts, mut batch_lengths, mut metrics) =
        match (&params.predicate, params.sample_rate) {
            (None, None) => {
                let mut metrics = SampleMetrics::new(dataset_size);
                metrics.matched_rows = dataset_size;
                // No filter or sample rate, so just batch the whole dataset.
                // TODO: optimize this
                let (batch_starts, batch_lengths) =
                    make_batches(0..dataset_size as u64, params.batch_size, large_indices)?;
                (None, batch_starts, batch_lengths, metrics)
            }
            (None, Some(sample_rate)) => {
                // No filter, but we do have a sample rate.

                metrics.matched_rows = dataset_size;
                let row_iter = (0..dataset_size as u64).filter(|_| rng.gen::<f32>() < sample_rate);
                let (batch_starts, batch_lengths) =
                    make_batches(row_iter, params.batch_size, large_indices)?;
                (None, batch_starts, batch_lengths, metrics)
            }
            (Some(predicate), _) => {
                // There is some filter, so we need to compute a mask.
                let sample_rate = params.sample_rate.unwrap_or(1.0);

                let mask =
                    compute_mask(&dataset, predicate, sample_rate, &mut metrics, &mut rng).await?;

                let (batch_starts, batch_lengths) =
                    make_batches(mask.iter(), params.batch_size, large_indices)?;
                (Some(mask), batch_starts, batch_lengths, metrics)
            }
        };

    // If we haven't already, count the number of sampled rows.
    if metrics.sampled_rows == 0 {
        metrics.sampled_rows = batch_lengths.values().iter().map(|v| *v as usize).sum();
    }

    if params.shuffle {
        let random_indices = random_indices(batch_starts.len(), &mut rng)?;
        batch_starts = take(&batch_starts, &random_indices, None).map_err(|err| {
            PyValueError::new_err(format!("Failed to compute DatasetSample slice: {err}"))
        })?;
        batch_lengths = Arc::new(
            take(batch_lengths.as_ref(), &random_indices, None)
                .map_err(|err| {
                    PyValueError::new_err(format!("Failed to compute DatasetSample slice: {err}"))
                })?
                .as_primitive::<UInt16Type>()
                .clone(),
        );
    }

    Ok(DatasetSample {
        params,
        row_id_mask,
        batch_starts,
        batch_lengths,
        metrics,
    })
}

fn make_batches(
    row_iter: impl Iterator<Item = u64>,
    batch_size: usize,
    large_indices: bool,
) -> PyResult<(Arc<dyn Array>, Arc<UInt16Array>)> {
    if large_indices {
        make_batches_impl::<UInt64Type>(row_iter, batch_size)
    } else {
        make_batches_impl::<UInt32Type>(row_iter, batch_size)
    }
}

fn make_batches_impl<T: ArrowPrimitiveType>(
    mut row_iter: impl Iterator<Item = u64>,
    batch_size: usize,
) -> PyResult<(Arc<dyn Array>, Arc<UInt16Array>)>
where
    T::Native: NumCast,
{
    // TODO: sampled but no filter case doesn't have a size hint.
    let min_length = row_iter.size_hint().0 / batch_size;
    let mut batch_starts = PrimitiveBuilder::<T>::with_capacity(min_length);
    let mut batch_lengths: Vec<u16> = Vec::with_capacity(min_length);

    let mut curr_start: u64 = row_iter
        .next()
        .ok_or_else(|| PyValueError::new_err("Zero rows have been sampled."))?;
    let mut curr_end: u64 = curr_start + 1;
    for value in row_iter {
        if value - curr_start >= batch_size as u64 {
            // We've reached the end of a batch.
            batch_starts.append_value(NumCast::from(curr_start).expect("invalid cast"));
            batch_lengths.push((curr_end - curr_start) as u16);
            curr_start = value;
        }
        curr_end = value + 1;
    }

    // Add the last batch.
    batch_starts.append_value(NumCast::from(curr_start).expect("invalid cast"));
    batch_lengths.push((curr_end - curr_start) as u16);

    let batch_lengths = UInt16Array::from(batch_lengths);
    let batch_starts: ArrayRef = Arc::new(batch_starts.finish());
    Ok((batch_starts, Arc::new(batch_lengths)))
}

fn random_indices(num_indices: usize, rng: &mut impl Rng) -> PyResult<ArrayRef> {
    if num_indices > u32::MAX as usize {
        let mut indices = (0..num_indices as u64).collect::<Vec<_>>();
        indices.shuffle(rng);
        Ok(Arc::new(UInt64Array::from(indices)))
    } else {
        let mut indices = (0..num_indices as u32).collect::<Vec<_>>();
        indices.shuffle(rng);
        Ok(Arc::new(UInt32Array::from(indices)))
    }
}

#[pyclass]
struct DatasetSampleIter {
    sample: DatasetSample,
    i: usize,
}

impl DatasetSampleIter {
    pub fn new(sample: DatasetSample) -> Self {
        Self { sample, i: 0 }
    }
}

#[pymethods]
impl DatasetSampleIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyObject>> {
        if slf.i >= slf.sample.__len__() {
            return Ok(None);
        }
        let out = slf
            .sample
            .__getitem__(slf.py(), SliceOrInt::Int(slf.i as isize));
        slf.i += 1;
        out.map(Some)
    }
}

fn serialize_record_batch(batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
    let mut batch_data = Vec::new();
    let mut batch_writer = arrow::ipc::writer::FileWriter::try_new_with_options(
        &mut batch_data,
        batch.schema().as_ref(),
        IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))?,
    )?;
    batch_writer.write(&batch)?;
    batch_writer.finish()?;
    drop(batch_writer);
    Ok(batch_data)
}

mod tarfile {
    use super::*;

    /// Open a tarfile with the given mode.
    pub fn open<'py>(path_or_file: &'py PyAny, mode: &str) -> PyResult<&'py PyAny> {
        let py = path_or_file.py();
        if path_or_file.is_instance_of::<PyString>() {
            py.import(intern!(py, "tarfile"))?
                .getattr(intern!(py, "open"))?
                .call1((path_or_file, mode))
        } else {
            let args = PyDict::new(py);
            args.set_item("fileobj", path_or_file)?;
            // Should we even set mode?
            args.set_item("mode", mode)?;
            py.import(intern!(py, "tarfile"))?
                .getattr(intern!(py, "open"))?
                .call((), Some(args))
        }
    }

    /// Add a file to a tarfile based on the name and provided bytes
    pub fn add_file(
        tarfile: &PyAny,
        name: impl IntoPy<Py<PyString>>,
        data: &PyBytes,
    ) -> PyResult<()> {
        let py = tarfile.py();
        let name = name.into_py(py);
        let info = py
            .import(intern!(py, "tarfile"))?
            .getattr(intern!(py, "TarInfo"))?
            .call1((name,))?;
        info.setattr(intern!(py, "size"), data.len()?)?;

        let bytes_io = py
            .import(intern!(py, "io"))?
            .getattr(intern!(py, "BytesIO"))?;

        tarfile
            .getattr(intern!(tarfile.py(), "addfile"))?
            .call1((info, bytes_io.call1((data,))?))?;
        Ok(())
    }

    /// Read a file from a tarfile.
    pub fn read_file<'py>(
        tarfile: &'py PyAny,
        name: impl IntoPy<Py<PyString>>,
    ) -> PyResult<Option<&'py PyBytes>> {
        let name = name.into_py(tarfile.py());
        let file = tarfile
            .call_method1(intern!(tarfile.py(), "extractfile"), (&name,))
            .map_err(|err| {
                PyIOError::new_err(format!(
                    "Failed to extract file {} from tarfile: {err}",
                    name
                ))
            })?;
        if file.is_none() {
            return Ok(None);
        }
        file.call_method0(intern!(tarfile.py(), "read"))?
            .downcast::<PyBytes>()
            .map_err(|err| {
                PyIOError::new_err(format!("Failed to read file {} from tarfile: {err}", name))
            })
            .map(Some)
    }
}
