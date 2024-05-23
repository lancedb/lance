// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::Float32Type;
use arrow_array::{FixedSizeListArray, Float32Array};
use async_trait::async_trait;
use lance_arrow::ArrowFloatType;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::traits::Reader;
use lance_linalg::distance::{DistanceType, Dot, MetricType, L2};
use lance_table::format::SelfDescribingFileReader;
use snafu::{location, Location};

use crate::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

use super::pq::storage::PQ_METADTA_KEY;
use super::pq::ProductQuantizer;
use super::sq::storage::SQ_METADATA_KEY;
use super::{
    ivf::storage::IvfData,
    pq::{
        storage::{ProductQuantizationMetadata, ProductQuantizationStorage},
        ProductQuantizerImpl,
    },
    sq::{
        storage::{ScalarQuantizationMetadata, ScalarQuantizationStorage},
        ScalarQuantizer,
    },
    v3::storage::VectorStore,
};
use super::{PQ_CODE_COLUMN, SQ_CODE_COLUMN};

pub trait Quantization {
    type Metadata: QuantizerMetadata + Send + Sync;
    type Storage: QuantizerStorage<Metadata = Self::Metadata> + VectorStore;

    fn code_dim(&self) -> usize;
    fn column(&self) -> &'static str;
    fn metadata_key(&self) -> &'static str;
    fn quantization_type(&self) -> QuantizationType;
    fn metadata(&self, _: Option<QuantizationMetadata>) -> Result<serde_json::Value>;
    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer>;
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

impl From<Arc<dyn ProductQuantizer>> for Quantizer {
    fn from(pq: Arc<dyn ProductQuantizer>) -> Self {
        Self::Product(pq)
    }
}

impl From<ScalarQuantizer> for Quantizer {
    fn from(sq: ScalarQuantizer) -> Self {
        Self::Scalar(sq)
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
            dim: self.dim,
            num_bits: self.num_bits(),
            bounds: self.bounds(),
        })?)
    }

    fn from_metadata(metadata: &Self::Metadata, _: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Scalar(Self::with_bounds(
            metadata.num_bits,
            metadata.dim,
            metadata.bounds.clone(),
        )))
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

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Product(Arc::new(ProductQuantizerImpl::<
            Float32Type,
        >::new(
            metadata.num_sub_vectors,
            metadata.num_bits,
            metadata.dimension,
            Arc::new(
                metadata
                    .codebook
                    .as_ref()
                    .unwrap()
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .clone(),
            ),
            distance_type,
        ))))
    }
}

impl<T: ArrowFloatType + 'static> Quantization for ProductQuantizerImpl<T>
where
    T::Native: Dot + L2,
{
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

    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer> {
        Ok(Quantizer::Product(Arc::new(Self::new(
            metadata.num_sub_vectors,
            metadata.num_bits,
            metadata.dimension,
            Arc::new(
                metadata
                    .codebook
                    .as_ref()
                    .unwrap()
                    .values()
                    .as_any()
                    .downcast_ref::<T::ArrayType>()
                    .unwrap()
                    .clone(),
            ),
            distance_type,
        ))))
    }
}

/// Loader to load partitioned PQ storage from disk.
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    metric_type: MetricType,
    quantizer: Quantizer,
    metadata: Q::Metadata,

    ivf: IvfData,
}

impl<Q: Quantization> Clone for IvfQuantizationStorage<Q> {
    fn clone(&self) -> Self {
        Self {
            reader: self.reader.clone(),
            metric_type: self.metric_type,
            quantizer: self.quantizer.clone(),
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
        let quantizer = Q::from_metadata(&metadata, metric_type)?;
        Ok(Self {
            reader,
            metric_type,
            quantizer,
            metadata,
            ivf: ivf_data,
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
        Q::Storage::load_partition(&self.reader, range, self.metric_type, &self.metadata).await
    }
}
