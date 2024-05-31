// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, sync::Arc};

use arrow::compute::concat_batches;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use deepsize::DeepSizeOf;
use futures::prelude::stream::{StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_file::v2::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_linalg::distance::{DistanceType, MetricType};
use prost::Message;
use snafu::{location, Location};

use crate::{
    pb,
    vector::{
        ivf::storage::{IvfData, IVF_METADATA_KEY},
        quantizer::{Quantization, Quantizer},
    },
    INDEX_METADATA_SCHEMA_KEY,
};

use super::DISTANCE_TYPE_KEY;

/// WARNING: Internal API,  API stability is not guaranteed
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;
    fn prefetch(&self, _id: u32) {}
}

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory raw vectors or on disk PQ code.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
///
/// WARNING: Internal API,  API stability is not guaranteed
pub trait VectorStore: Send + Sync {
    type DistanceCalculator<'a>: DistCalculator
    where
        Self: 'a;

    fn try_from_batch(batch: RecordBatch, distance_type: DistanceType) -> Result<Self>
    where
        Self: Sized;

    fn to_batch(&self) -> Result<RecordBatch>;

    fn as_any(&self) -> &dyn Any;

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn row_ids(&self) -> &[u64];

    /// Return the metric type of the vectors.
    fn metric_type(&self) -> MetricType;

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
        Q::Storage::try_from_batch(batch, self.distance_type)
    }
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    quantizer: Quantizer,
    metadata: Q::Metadata,

    ivf: IvfData,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // self.reader.deep_size_of_children(context)
        self.quantizer.deep_size_of_children(context)
            + self.metadata.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
    }
}

// impl<Q: Quantization> Clone for IvfQuantizationStorage<Q> {
//     fn clone(&self) -> Self {
//         Self {
//             reader: self.reader.clone(),
//             metric_type: self.metric_type,
//             quantizer: self.quantizer.clone(),
//             metadata: self.metadata.clone(),
//             ivf: self.ivf.clone(),
//         }
//     }
// }

#[allow(dead_code)]
impl<Q: Quantization> IvfQuantizationStorage<Q> {
    /// Open a Loader.
    ///
    ///
    pub async fn open(reader: FileReader) -> Result<Self> {
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

        let ivf_pb_bytes =
            hex::decode(schema.metadata.get(IVF_METADATA_KEY).ok_or(Error::Index {
                message: format!("{} not found", IVF_METADATA_KEY),
                location: location!(),
            })?)
            .map_err(|e| Error::Index {
                message: format!("Failed to decode IVF metadata: {}", e),
                location: location!(),
            })?;
        let ivf = IvfData::try_from(pb::Ivf::decode(ivf_pb_bytes.as_ref())?)?;

        let quantizer_metadata: Q::Metadata = serde_json::from_str(
            schema
                .metadata
                .get(Q::metadata_key())
                .ok_or(Error::Index {
                    message: format!("{} not found", Q::metadata_key()),
                    location: location!(),
                })?
                .as_str(),
        )?;
        let quantizer = Q::from_metadata(&quantizer_metadata, distance_type)?;
        Ok(Self {
            reader,
            distance_type,
            quantizer,
            metadata: quantizer_metadata,
            ivf,
        })
    }

    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    pub fn metadata(&self) -> &Q::Metadata {
        &self.metadata
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    pub async fn load_partition(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let batches = self
            .reader
            .read_stream(ReadBatchParams::Range(range), 4096, 16)?
            .peekable()
            .try_collect::<Vec<_>>()
            .await?;
        let schema = Arc::new(self.reader.schema().as_ref().into());
        let batch = concat_batches(&schema, batches.iter())?;
        Q::Storage::try_from_batch(batch, self.distance_type)
    }
}
