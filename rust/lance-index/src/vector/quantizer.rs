// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use lance_table::format::SelfDescribingFileReader;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

use super::flat::index::FlatQuantizer;
use super::pq::ProductQuantizer;
use super::{ivf::storage::IvfModel, sq::ScalarQuantizer, storage::VectorStore};

pub trait Quantization: Send + Sync + Debug + DeepSizeOf + Into<Quantizer> {
    type BuildParams: QuantizerBuildParams;
    type Metadata: QuantizerMetadata + Send + Sync;
    type Storage: QuantizerStorage<Metadata = Self::Metadata> + VectorStore + Debug;

    fn build(
        data: &dyn Array,
        distance_type: DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self>;
    fn code_dim(&self) -> usize;
    fn column(&self) -> &'static str;
    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef>;
    fn metadata_key() -> &'static str;
    fn quantization_type() -> QuantizationType;
    fn metadata(&self, _: Option<QuantizationMetadata>) -> Result<serde_json::Value>;
    fn from_metadata(metadata: &Self::Metadata, distance_type: DistanceType) -> Result<Quantizer>;
}

pub enum QuantizationType {
    Flat,
    Product,
    Scalar,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flat => write!(f, "FLAT"),
            Self::Product => write!(f, "PQ"),
            Self::Scalar => write!(f, "SQ"),
        }
    }
}

pub trait QuantizerBuildParams {
    fn sample_size(&self) -> usize;
}

impl QuantizerBuildParams for () {
    fn sample_size(&self) -> usize {
        0
    }
}

/// Quantization Method.
///
/// <section class="warning">
/// Internal use only. End-user does not use this directly.
/// </section>
#[derive(Debug, Clone, DeepSizeOf)]
pub enum Quantizer {
    Flat(FlatQuantizer),
    Product(ProductQuantizer),
    Scalar(ScalarQuantizer),
}

impl Quantizer {
    pub fn code_dim(&self) -> usize {
        match self {
            Self::Flat(fq) => fq.code_dim(),
            Self::Product(pq) => pq.code_dim(),
            Self::Scalar(sq) => sq.code_dim(),
        }
    }

    pub fn column(&self) -> &'static str {
        match self {
            Self::Flat(fq) => fq.column(),
            Self::Product(pq) => pq.column(),
            Self::Scalar(sq) => sq.column(),
        }
    }

    pub fn metadata_key(&self) -> &'static str {
        match self {
            Self::Flat(_) => FlatQuantizer::metadata_key(),
            Self::Product(_) => ProductQuantizer::metadata_key(),
            Self::Scalar(_) => ScalarQuantizer::metadata_key(),
        }
    }

    pub fn quantization_type(&self) -> QuantizationType {
        match self {
            Self::Flat(_) => QuantizationType::Flat,
            Self::Product(_) => QuantizationType::Product,
            Self::Scalar(_) => QuantizationType::Scalar,
        }
    }

    pub fn metadata(&self, args: Option<QuantizationMetadata>) -> Result<serde_json::Value> {
        match self {
            Self::Flat(fq) => fq.metadata(args),
            Self::Product(pq) => pq.metadata(args),
            Self::Scalar(sq) => sq.metadata(args),
        }
    }
}

impl From<ProductQuantizer> for Quantizer {
    fn from(pq: ProductQuantizer) -> Self {
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
pub trait QuantizerMetadata:
    fmt::Debug + Clone + Sized + DeepSizeOf + for<'a> Deserialize<'a> + Serialize
{
    async fn load(reader: &FileReader) -> Result<Self>;
}

#[async_trait::async_trait]
pub trait QuantizerStorage: Clone + Sized + DeepSizeOf + VectorStore {
    type Metadata: QuantizerMetadata;

    async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
    ) -> Result<Self>;
}

/// Loader to load partitioned [VectorStore] from disk.
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    quantizer: Quantizer,
    metadata: Q::Metadata,

    ivf: IvfModel,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.reader.deep_size_of_children(context)
            + self.quantizer.deep_size_of_children(context)
            + self.metadata.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
    }
}

impl<Q: Quantization> Clone for IvfQuantizationStorage<Q> {
    fn clone(&self) -> Self {
        Self {
            reader: self.reader.clone(),
            distance_type: self.distance_type,
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
        let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let ivf_data = IvfModel::load(&reader).await?;

        let metadata = Q::Metadata::load(&reader).await?;
        let quantizer = Q::from_metadata(&metadata, distance_type)?;
        Ok(Self {
            reader,
            distance_type,
            quantizer,
            metadata,
            ivf: ivf_data,
        })
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
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

    /// Load one partition of vector storage.
    ///
    /// # Parameters
    /// - `part_id`, partition id
    ///
    ///
    pub async fn load_partition(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        Q::Storage::load_partition(&self.reader, range, self.distance_type, &self.metadata).await
    }
}
