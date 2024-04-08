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

use arrow_array::FixedSizeListArray;
use async_trait::async_trait;
use lance_arrow::ArrowFloatType;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::traits::Reader;
use lance_linalg::distance::{Dot, MetricType, L2};
use lance_table::format::SelfDescribingFileReader;
use snafu::{location, Location};

use crate::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

use super::pq::storage::PQ_METADTA_KEY;
use super::pq::ProductQuantizer;
use super::sq::storage::SQ_METADATA_KEY;
use super::{
    graph::VectorStorage,
    ivf::storage::IvfData,
    pq::{
        storage::{ProductQuantizationMetadata, ProductQuantizationStorage},
        ProductQuantizerImpl,
    },
    sq::{
        storage::{ScalarQuantizationMetadata, ScalarQuantizationStorage},
        ScalarQuantizer,
    },
};
use super::{PQ_CODE_COLUMN, SQ_CODE_COLUMN};

pub trait Quantization {
    type Metadata: QuantizerMetadata + Send + Sync;
    type Storage: QuantizerStorage<Metadata = Self::Metadata> + VectorStorage;

    fn code_dim(&self) -> usize;
    fn column(&self) -> &'static str;
    fn metadata_key(&self) -> &'static str;
    fn quantization_type(&self) -> QuantizationType;
    fn metadata(&self, _: Option<QuantizationMetadata>) -> Result<serde_json::Value>;
}

pub enum QuantizationType {
    Product,
    Scalar,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Product => write!(f, "PQ"),
            Self::Scalar => write!(f, "SQ"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Quantizer {
    Product(Arc<dyn ProductQuantizer>),
    Scalar(ScalarQuantizer),
}

impl Quantizer {
    pub fn code_dim(&self) -> usize {
        match self {
            Self::Product(pq) => pq.num_sub_vectors(),
            Self::Scalar(sq) => sq.dim,
        }
    }

    pub fn column(&self) -> &'static str {
        match self {
            Self::Product(pq) => pq.column(),
            Self::Scalar(sq) => sq.column(),
        }
    }

    pub fn metadata_key(&self) -> &'static str {
        match self {
            Self::Product(pq) => pq.metadata_key(),
            Self::Scalar(sq) => sq.metadata_key(),
        }
    }

    pub fn quantization_type(&self) -> QuantizationType {
        match self {
            Self::Product(pq) => pq.quantization_type(),
            Self::Scalar(sq) => sq.quantization_type(),
        }
    }

    pub fn metadata(&self, args: Option<QuantizationMetadata>) -> Result<serde_json::Value> {
        match self {
            Self::Product(pq) => pq.metadata(args),
            Self::Scalar(sq) => sq.metadata(args),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct QuantizationMetadata {
    // For PQ
    pub codebook_position: Option<usize>,
    pub codebook: Option<FixedSizeListArray>,
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

impl Quantization for ScalarQuantizer {
    type Metadata = ScalarQuantizationMetadata;
    type Storage = ScalarQuantizationStorage;

    fn code_dim(&self) -> usize {
        self.dim
    }

    fn column(&self) -> &'static str {
        SQ_CODE_COLUMN
    }

    fn metadata_key(&self) -> &'static str {
        SQ_METADATA_KEY
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Scalar
    }

    fn metadata(&self, _: Option<QuantizationMetadata>) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(ScalarQuantizationMetadata {
            num_bits: self.num_bits(),
            bounds: self.bounds(),
        })?)
    }
}

impl Quantization for dyn ProductQuantizer {
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;

    fn code_dim(&self) -> usize {
        self.num_sub_vectors()
    }

    fn column(&self) -> &'static str {
        PQ_CODE_COLUMN
    }

    fn metadata_key(&self) -> &'static str {
        PQ_METADTA_KEY
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Product
    }

    fn metadata(&self, args: Option<QuantizationMetadata>) -> Result<serde_json::Value> {
        let args = args.unwrap_or_default();

        let codebook_position = args.codebook_position.ok_or(Error::Index {
            message: "codebook_position not found".to_owned(),
            location: location!(),
        })?;
        Ok(serde_json::to_value(ProductQuantizationMetadata {
            codebook_position,
            num_bits: self.num_bits(),
            num_sub_vectors: self.num_sub_vectors(),
            dimension: self.dimension(),
            codebook: args.codebook,
        })?)
    }
}

impl<T: ArrowFloatType + Dot + L2 + 'static> Quantization for ProductQuantizerImpl<T> {
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;

    fn code_dim(&self) -> usize {
        self.num_sub_vectors()
    }

    fn column(&self) -> &'static str {
        PQ_CODE_COLUMN
    }

    fn metadata_key(&self) -> &'static str {
        PQ_METADTA_KEY
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Product
    }

    fn metadata(&self, args: Option<QuantizationMetadata>) -> Result<serde_json::Value> {
        let args = args.unwrap_or_default();

        let codebook_position = args.codebook_position.ok_or(Error::Index {
            message: "codebook_position not found".to_owned(),
            location: location!(),
        })?;
        Ok(serde_json::to_value(ProductQuantizationMetadata {
            codebook_position,
            num_bits: self.num_bits(),
            num_sub_vectors: self.num_sub_vectors(),
            dimension: self.dimension(),
            codebook: args.codebook,
        })?)
    }
}

/// Loader to load partitioned PQ storage from disk.
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    metric_type: MetricType,
    metadata: Q::Metadata,

    ivf: IvfData,
}

impl<Q: Quantization> Clone for IvfQuantizationStorage<Q> {
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
impl<Q: Quantization> IvfQuantizationStorage<Q> {
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
