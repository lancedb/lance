// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use std::collections::HashMap;
use std::{any::Any, sync::Arc};

use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow::datatypes::UInt64Type;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::SchemaRef;
use deepsize::DeepSizeOf;
use futures::prelude::stream::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::v2::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_linalg::distance::DistanceType;
use prost::Message;
use snafu::location;

use crate::vector::pq::transform::TransposeTransformer;
use crate::{
    pb,
    vector::{
        ivf::storage::{IvfModel, IVF_METADATA_KEY},
        quantizer::Quantization,
    },
};

use super::quantizer::{QuantizationType, Quantizer};
use super::transform::Transformer;
use super::DISTANCE_TYPE_KEY;

/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;
    fn distance_all(&self) -> Vec<f32>;
    fn prefetch(&self, _id: u32) {}
}

pub const STORAGE_METADATA_KEY: &str = "storage_metadata";

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory or on-disk, raw vector or quantized vectors.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
///
/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait VectorStore: Send + Sync + Sized + Clone {
    type DistanceCalculator<'a>: DistCalculator
    where
        Self: 'a;

    /// Create a [VectorStore] from a [RecordBatch].
    /// The batch should consist of row IDs and quantized vector.
    fn try_from_batch(batch: RecordBatch, distance_type: DistanceType) -> Result<Self>;

    fn as_any(&self) -> &dyn Any;

    fn schema(&self) -> &SchemaRef;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send>;

    fn remap(&self, mapping: &HashMap<u64, Option<u64>>) -> Result<Self> {
        let batches = self
            .to_batches()?
            .map(|b| {
                let mut indices = Vec::with_capacity(b.num_rows());
                let mut new_row_ids = Vec::with_capacity(b.num_rows());

                let row_ids = b.column(0).as_primitive::<UInt64Type>().values();
                for (i, row_id) in row_ids.iter().enumerate() {
                    match mapping.get(row_id) {
                        Some(Some(new_id)) => {
                            indices.push(i as u32);
                            new_row_ids.push(*new_id);
                        }
                        Some(None) => {}
                        None => {
                            indices.push(i as u32);
                            new_row_ids.push(*row_id);
                        }
                    }
                }

                let indices = UInt32Array::from(indices);
                let new_row_ids = Arc::new(UInt64Array::from(new_row_ids));
                let new_vectors = arrow::compute::take(b.column(1), &indices, None)?;

                Ok(RecordBatch::try_new(
                    self.schema().clone(),
                    vec![new_row_ids, new_vectors],
                )?)
            })
            .collect::<Result<Vec<_>>>()?;

        let batch = concat_batches(self.schema(), batches.iter())?;
        Self::try_from_batch(batch, self.distance_type())
    }

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return [DistanceType].
    fn distance_type(&self) -> DistanceType;

    /// Get the lance ROW ID from one vector.
    fn row_id(&self, id: u32) -> u64;

    fn row_ids(&self) -> impl Iterator<Item = &u64>;

    /// Append Raw [RecordBatch] into the Storage.
    /// The storage implement will perform quantization if necessary.
    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self>;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calculator can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_>;

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_>;
}

pub struct StorageBuilder<Q: Quantization> {
    distance_type: DistanceType,
    _quantizer: Q,
    transformers: Vec<Arc<dyn Transformer>>,
}

impl<Q: Quantization> StorageBuilder<Q> {
    pub fn new(distance_type: DistanceType, quantizer: Q) -> Result<Self> {
        let transformers = if matches!(Q::quantization_type(), QuantizationType::Product) {
            let metadata = quantizer.metadata(None)?;
            vec![Arc::new(TransposeTransformer::new(metadata.to_string())?) as _]
        } else {
            Vec::new()
        };
        Ok(Self {
            distance_type,
            _quantizer: quantizer,
            transformers,
        })
    }

    pub fn build(&self, batch: Vec<RecordBatch>) -> Result<Q::Storage> {
        let batches = self
            .transformers
            .iter()
            .try_fold(batch, |batches, transformer| {
                batches
                    .into_iter()
                    .map(|b| transformer.transform(&b))
                    .collect::<Result<Vec<_>>>()
            })?;

        let batch = concat_batches(batches[0].schema_ref(), batches.iter())?;
        Q::Storage::try_from_batch(batch, self.distance_type)
    }
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage {
    reader: FileReader,

    distance_type: DistanceType,
    metadata: Vec<String>,

    ivf: IvfModel,
}

impl DeepSizeOf for IvfQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.ivf.deep_size_of_children(context)
    }
}

#[allow(dead_code)]
impl IvfQuantizationStorage {
    /// Open a Loader.
    ///
    ///
    pub async fn try_new(reader: FileReader) -> Result<Self> {
        let schema = reader.schema();

        let distance_type = DistanceType::try_from(
            schema
                .metadata
                .get(DISTANCE_TYPE_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", DISTANCE_TYPE_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;

        let ivf_pos = schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!("{} not found", IVF_METADATA_KEY),
                location: location!(),
            })?
            .parse()
            .map_err(|e| Error::Index {
                message: format!("Failed to decode IVF metadata: {}", e),
                location: location!(),
            })?;
        let ivf_bytes = reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_bytes)?)?;

        let metadata: Vec<String> = serde_json::from_str(
            schema
                .metadata
                .get(STORAGE_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", STORAGE_METADATA_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;
        Ok(Self {
            reader,
            distance_type,
            metadata,
            ivf,
        })
    }

    pub fn quantizer<Q: Quantization>(&self) -> Result<Quantizer> {
        let metadata = self.metadata::<Q>()?;
        Q::from_metadata(&metadata, self.distance_type)
    }

    pub fn metadata<Q: Quantization>(&self) -> Result<Q::Metadata> {
        Ok(serde_json::from_str(&self.metadata[0])?)
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.reader.schema().as_ref().into())
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    pub async fn load_partition<Q: Quantization>(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let batch = if range.is_empty() {
            let schema = self.reader.schema();
            let arrow_schema = arrow_schema::Schema::from(schema.as_ref());
            RecordBatch::new_empty(Arc::new(arrow_schema))
        } else {
            let batches = self
                .reader
                .read_stream(
                    ReadBatchParams::Range(range),
                    u32::MAX,
                    16,
                    FilterExpression::no_filter(),
                )?
                .try_collect::<Vec<_>>()
                .await?;
            let schema = Arc::new(self.reader.schema().as_ref().into());
            concat_batches(&schema, batches.iter())?
        };
        let batch = batch.add_metadata(
            STORAGE_METADATA_KEY.to_owned(),
            // TODO: this is a hack, cause the metadata is just the quantizer metadata
            // it's all the same for all partitions, so now we store only one copy of it
            self.metadata[0].clone(),
        )?;
        Q::Storage::try_from_batch(batch, self.distance_type)
    }
}
