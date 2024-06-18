// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use std::{any::Any, sync::Arc};

use arrow::compute::concat_batches;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, SchemaRef};
use deepsize::DeepSizeOf;
use futures::prelude::stream::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::v2::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_linalg::distance::DistanceType;
use prost::Message;
use snafu::{location, Location};

use crate::{
    pb,
    vector::{
        ivf::storage::{IvfModel, IVF_METADATA_KEY},
        quantizer::Quantization,
    },
    INDEX_METADATA_SCHEMA_KEY,
};

use super::DISTANCE_TYPE_KEY;

/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;
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

    fn try_from_batch(batch: RecordBatch, distance_type: DistanceType) -> Result<Self>;

    fn as_any(&self) -> &dyn Any;

    fn schema(&self) -> &SchemaRef;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>>;

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
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_>;

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_>;

    fn distance_between(&self, a: u32, b: u32) -> f32;

    fn dist_calculator_from_native(&self, _query: ArrayRef) -> Self::DistanceCalculator<'_> {
        todo!("Implement this")
    }
}

pub struct StorageBuilder<Q: Quantization> {
    column: String,
    distance_type: DistanceType,
    quantizer: Q,
}

impl<Q: Quantization> StorageBuilder<Q> {
    pub fn new(column: String, distance_type: DistanceType, quantizer: Q) -> Self {
        Self {
            column,
            distance_type,
            quantizer,
        }
    }

    pub fn build(&self, batch: &RecordBatch) -> Result<Q::Storage> {
        let vectors = batch.column_by_name(&self.column).ok_or(Error::Schema {
            message: format!("column {} not found", self.column),
            location: location!(),
        })?;
        let code_array = self.quantizer.quantize(vectors.as_ref())?;
        let batch = batch.drop_column(&self.column)?.try_with_column(
            Field::new(
                self.quantizer.column(),
                code_array.data_type().clone(),
                true,
            ),
            code_array,
        )?;
        let batch = batch.add_metadata(
            STORAGE_METADATA_KEY.to_owned(),
            self.quantizer.metadata(None)?.to_string(),
        )?;
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
                    message: format!("{} not found", INDEX_METADATA_SCHEMA_KEY),
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

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    pub async fn load_partition<Q: Quantization>(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let batches = self
            .reader
            .read_stream(
                ReadBatchParams::Range(range),
                4096,
                16,
                FilterExpression::no_filter(),
            )?
            .try_collect::<Vec<_>>()
            .await?;
        let schema = Arc::new(self.reader.schema().as_ref().into());
        let batch = concat_batches(&schema, batches.iter())?;
        let batch = batch.add_metadata(
            STORAGE_METADATA_KEY.to_owned(),
            self.metadata[part_id].clone(),
        )?;
        Q::Storage::try_from_batch(batch, self.distance_type)
    }
}
