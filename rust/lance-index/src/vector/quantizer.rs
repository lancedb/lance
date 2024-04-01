// Copyright 2024 Lance Developers.
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

use std::sync::Arc;

use async_trait::async_trait;
use lance_arrow::ArrowFloatType;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::traits::Reader;
use lance_linalg::distance::{Dot, MetricType, L2};
use lance_table::format::SelfDescribingFileReader;
use snafu::{location, Location};

use crate::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

use super::{
    graph::VectorStorage,
    ivf::storage::IvfData,
    pq::{
        storage::{ProductQuantizationMetadata, ProductQuantizationStorage},
        ProductQuantizer, ProductQuantizerImpl,
    },
    sq::{
        storage::{ScalarQuantizationMetadata, ScalarQuantizationStorage},
        ScalarQuantizer,
    },
};

pub trait Quantizer {
    type Metadata: QuantizerMetadata + Send + Sync;
    type Storage: QuantizerStorage<Metadata = Self::Metadata> + VectorStorage;
}

#[async_trait]
pub trait QuantizerMetadata: Clone + Sized {
    async fn load(reader: &FileReader) -> Result<Self>;
}

#[async_trait::async_trait]
pub trait QuantizerStorage: Clone + Sized {
    type Metadata: QuantizerMetadata;

    async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        metric_type: MetricType,
        metadata: &Self::Metadata,
    ) -> Result<Self>;
}

impl Quantizer for ScalarQuantizer {
    type Metadata = ScalarQuantizationMetadata;
    type Storage = ScalarQuantizationStorage;
}

impl Quantizer for dyn ProductQuantizer {
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;
}

impl<T: ArrowFloatType + Dot + L2> Quantizer for ProductQuantizerImpl<T> {
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;
}

/// Loader to load partitioned PQ storage from disk.
pub struct IvfQuantizationStorage<Q: Quantizer> {
    reader: FileReader,

    metric_type: MetricType,
    metadata: Q::Metadata,

    ivf: IvfData,
}

impl<Q: Quantizer> Clone for IvfQuantizationStorage<Q> {
    fn clone(&self) -> Self {
        Self {
            reader: self.reader.clone(),
            metric_type: self.metric_type,
            metadata: self.metadata.clone(),
            ivf: self.ivf.clone(),
        }
    }
}

#[allow(dead_code)]
impl<Q: Quantizer> IvfQuantizationStorage<Q> {
    /// Open a Loader.
    ///
    ///
    pub async fn open(reader: Arc<dyn Reader>) -> Result<Self> {
        let reader = FileReader::try_new_self_described_from_reader(reader, None).await?;
        let schema = reader.schema();

        let metadata_str = schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading quantization storage: index key {} not found",
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

        let ivf_data = IvfData::load(&reader).await?;

        let metadata = Q::Metadata::load(&reader).await?;
        Ok(Self {
            reader,
            metric_type,
            metadata,
            ivf: ivf_data,
        })
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    pub async fn load_partition(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        Q::Storage::load_partition(&self.reader, range, self.metric_type, &self.metadata).await
    }
}
