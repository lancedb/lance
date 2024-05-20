// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow::{
    array::AsArray,
    datatypes::{Float32Type, UInt8Type},
};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array};
use async_trait::async_trait;
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::{l2_distance_uint_scalar, MetricType};
use lance_table::format::SelfDescribingFileReader;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{
    vector::{
        graph::{storage::DistCalculator, VectorStorage},
        quantizer::{QuantizerMetadata, QuantizerStorage},
        SQ_CODE_COLUMN,
    },
    IndexMetadata, INDEX_METADATA_SCHEMA_KEY,
};

use super::scale_to_u8;

pub const SQ_METADATA_KEY: &str = "lance:sq";

#[derive(Clone, Serialize, Deserialize)]
pub struct ScalarQuantizationMetadata {
    pub dim: usize,
    pub num_bits: u16,
    pub bounds: Range<f64>,
}

#[async_trait]
impl QuantizerMetadata for ScalarQuantizationMetadata {
    async fn load(reader: &FileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(SQ_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading SQ metadata: metadata key {} not found",
                    SQ_METADATA_KEY
                ),
                location: location!(),
            })?;
        serde_json::from_str(metadata_str).map_err(|_| Error::Index {
            message: format!("Failed to parse index metadata: {}", metadata_str),
            location: location!(),
        })
    }
}

#[derive(Clone)]
pub struct ScalarQuantizationStorage {
    metric_type: MetricType,

    // Metadata
    num_bits: u16,
    bounds: Range<f64>,

    // Row IDs and SQ codes
    batch: RecordBatch,

    // Helper fields, references to the batch
    row_ids: Arc<UInt64Array>,
    sq_codes: Arc<FixedSizeListArray>,
}

impl ScalarQuantizationStorage {
    pub fn new(
        num_bits: u16,
        metric_type: MetricType,
        bounds: Range<f64>,
        batch: RecordBatch,
    ) -> Result<Self> {
        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .ok_or(Error::Index {
                    message: "Row ID column not found in the batch".to_owned(),
                    location: location!(),
                })?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .clone(),
        );
        let sq_codes = Arc::new(
            batch
                .column_by_name(SQ_CODE_COLUMN)
                .ok_or(Error::Index {
                    message: "SQ code column not found in the batch".to_owned(),
                    location: location!(),
                })?
                .as_fixed_size_list()
                .clone(),
        );

        Ok(Self {
            num_bits,
            metric_type,
            bounds,
            batch,
            row_ids,
            sq_codes,
        })
    }

    pub fn num_bits(&self) -> u16 {
        self.num_bits
    }

    pub fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    pub fn bounds(&self) -> Range<f64> {
        self.bounds.clone()
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn row_ids(&self) -> &[u64] {
        self.row_ids.values()
    }

    pub fn sq_codes(&self) -> &Arc<FixedSizeListArray> {
        &self.sq_codes
    }

    pub async fn load(object_store: &ObjectStore, path: &Path) -> Result<Self> {
        let reader = FileReader::try_new_self_described(object_store, path, None).await?;
        let schema = reader.schema();

        let metadata_str = schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading SQ storage: index key {} not found",
                    INDEX_METADATA_SCHEMA_KEY
                ),
                location: location!(),
            })?;
        let index_metadata: IndexMetadata =
            serde_json::from_str(metadata_str).map_err(|_| Error::Index {
                message: format!("Failed to parse index metadata: {}", metadata_str),
                location: location!(),
            })?;
        let metric_type: MetricType = MetricType::try_from(index_metadata.distance_type.as_str())?;
        let metadata = ScalarQuantizationMetadata::load(&reader).await?;

        Self::load_partition(&reader, 0..reader.len(), metric_type, &metadata).await
    }
}

#[async_trait]
impl QuantizerStorage for ScalarQuantizationStorage {
    type Metadata = ScalarQuantizationMetadata;
    /// Load a partition of SQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// - *reader: file reader
    /// - *range: row range of the partition
    /// - *metric_type: metric type of the vectors
    /// - *metadata: scalar quantization metadata
    async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        metric_type: MetricType,
        metadata: &Self::Metadata,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema, None).await?;

        Self::new(
            metadata.num_bits,
            metric_type,
            metadata.bounds.clone(),
            batch,
        )
    }
}

impl VectorStorage for ScalarQuantizationStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn row_ids(&self) -> &[u64] {
        self.row_ids.values()
    }

    /// Return the metric type of the vectors.
    fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator<'a>(&'a self, query: ArrayRef) -> Box<dyn DistCalculator<'a> + 'a> {
        Box::new(SQDistCalculator::new(
            query,
            self.sq_codes.as_ref(),
            self.bounds.clone(),
        ))
    }

    fn dist_calculator_from_id<'a>(&'a self, id: u32) -> Box<dyn DistCalculator<'a> + 'a> {
        let dim = self.sq_codes.value_length() as usize;
        Box::new(SQDistCalculator {
            query_sq_code: get_sq_code(&self.sq_codes, id).to_vec(),
            sq_codes: self.sq_codes.values().as_primitive::<UInt8Type>().values(),
            dim,
        })
    }

    fn distance_between(&self, a: u32, b: u32) -> f32 {
        l2_distance_uint_scalar(
            get_sq_code(&self.sq_codes, a),
            get_sq_code(&self.sq_codes, b),
        )
    }
}

struct SQDistCalculator<'a> {
    query_sq_code: Vec<u8>,
    sq_codes: &'a [u8],
    dim: usize,
}

impl<'a> SQDistCalculator<'a> {
    fn new(query: ArrayRef, sq_codes: &'a FixedSizeListArray, bounds: Range<f64>) -> Self {
        let query_sq_code =
            scale_to_u8::<Float32Type>(query.as_primitive::<Float32Type>().values(), bounds);
        let sq_codes_values = sq_codes.values().as_primitive::<UInt8Type>();
        let dim = sq_codes.value_length() as usize;
        Self {
            query_sq_code,
            sq_codes: sq_codes_values.values(),
            dim,
        }
    }
}

impl<'a> DistCalculator<'a> for SQDistCalculator<'a> {
    fn distance(&self, id: u32) -> f32 {
        let sq_code = &self.sq_codes[id as usize * self.dim..(id as usize + 1) * self.dim];
        l2_distance_uint_scalar(sq_code, &self.query_sq_code)
    }
    #[allow(unused_variables)]
    fn prefetch(&self, _id: u32) {
        const CACHE_LINE_SIZE: usize = 64;
        unsafe {
            use std::process::id;
            let base_ptr = self.sq_codes.as_ptr().add(id as usize * self.dim);

            // Loop over the sq_code to prefetch each cache line
            for offset in (0..self.dim).step_by(CACHE_LINE_SIZE) {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    use core::arch::x86_64::_mm_prefetch;
                    const _MM_HINT_T0: i32 = 1;
                    _mm_prefetch(base_ptr.add(offset) as *const i8, _MM_HINT_T0);
                }
            }
        }
    }
}

fn get_sq_code(sq_codes: &FixedSizeListArray, id: u32) -> &[u8] {
    let dim = sq_codes.value_length() as usize;
    let values: &[u8] = sq_codes
        .values()
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap()
        .values();
    &values[id as usize * dim..(id as usize + 1) * dim]
}
