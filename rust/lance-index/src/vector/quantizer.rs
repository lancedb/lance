// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::fmt;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_arrow::ArrowFloatType;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::traits::Reader;
use lance_linalg::distance::{DistanceType, Dot, L2};
use lance_table::format::SelfDescribingFileReader;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};

use super::flat::index::FlatQuantizer;
use super::pq::storage::PQ_METADTA_KEY;
use super::pq::ProductQuantizer;
use super::sq::builder::SQBuildParams;
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
    storage::VectorStore,
};
use super::{PQ_CODE_COLUMN, SQ_CODE_COLUMN};

pub trait Quantization: Send + Sync + DeepSizeOf + Into<Quantizer> {
    type BuildParams: QuantizerBuildParams;
    type Metadata: QuantizerMetadata + Send + Sync;
    type Storage: QuantizerStorage<Metadata = Self::Metadata> + VectorStore;

    fn build(
        data: &dyn Array,
        distance_type: DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self>;
    fn code_dim(&self) -> usize;
    fn column(&self) -> &'static str;
    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef>;
    fn metadata_key() -> &'static str;
    fn quantization_type(&self) -> QuantizationType;
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
    Product(Arc<dyn ProductQuantizer>),
    Scalar(ScalarQuantizer),
}

impl Quantizer {
    pub fn code_dim(&self) -> usize {
        match self {
            Self::Flat(fq) => fq.code_dim(),
            Self::Product(pq) => pq.num_sub_vectors(),
            Self::Scalar(sq) => sq.dim,
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
            Self::Product(_) => ProductQuantizerImpl::<Float32Type>::metadata_key(),
            Self::Scalar(_) => ScalarQuantizer::metadata_key(),
        }
    }

    pub fn quantization_type(&self) -> QuantizationType {
        match self {
            Self::Flat(fq) => fq.quantization_type(),
            Self::Product(pq) => pq.quantization_type(),
            Self::Scalar(sq) => sq.quantization_type(),
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

impl<T: ArrowFloatType + 'static> From<ProductQuantizerImpl<T>> for Quantizer
where
    T::Native: Dot + L2,
{
    fn from(pq: ProductQuantizerImpl<T>) -> Self {
        Self::Product(Arc::new(pq))
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

impl Quantization for ScalarQuantizer {
    type BuildParams = SQBuildParams;
    type Metadata = ScalarQuantizationMetadata;
    type Storage = ScalarQuantizationStorage;

    fn build(data: &dyn Array, _: DistanceType, params: &Self::BuildParams) -> Result<Self> {
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "SQ builder: input is not a FixedSizeList: {}",
                data.data_type()
            ),
            location: location!(),
        })?;

        let mut quantizer = Self::new(params.num_bits, fsl.value_length() as usize);

        match fsl.value_type() {
            DataType::Float16 => {
                quantizer.update_bounds::<Float16Type>(fsl)?;
            }
            DataType::Float32 => {
                quantizer.update_bounds::<Float32Type>(fsl)?;
            }
            DataType::Float64 => {
                quantizer.update_bounds::<Float64Type>(fsl)?;
            }
            _ => {
                return Err(Error::Index {
                    message: format!("SQ builder: unsupported data type: {}", fsl.value_type()),
                    location: location!(),
                })
            }
        }

        Ok(quantizer)
    }

    fn code_dim(&self) -> usize {
        self.dim
    }

    fn column(&self) -> &'static str {
        SQ_CODE_COLUMN
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        match vectors.as_fixed_size_list().value_type() {
            DataType::Float16 => self.transform::<Float16Type>(vectors),
            DataType::Float32 => self.transform::<Float32Type>(vectors),
            DataType::Float64 => self.transform::<Float64Type>(vectors),
            value_type => Err(Error::invalid_input(
                format!("unsupported data type {} for scalar quantizer", value_type),
                location!(),
            )),
        }
    }

    fn metadata_key() -> &'static str {
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

impl Quantization for Arc<dyn ProductQuantizer> {
    type BuildParams = ();
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;

    fn build(_: &dyn Array, _: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        unimplemented!("ProductQuantizer cannot be built with new index builder")
    }

    fn code_dim(&self) -> usize {
        self.num_sub_vectors()
    }

    fn column(&self) -> &'static str {
        PQ_CODE_COLUMN
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        let code_array = self.transform(vectors)?;
        Ok(code_array)
    }

    fn metadata_key() -> &'static str {
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
            codebook_tensor: Vec::new(),
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
    type BuildParams = ();
    type Metadata = ProductQuantizationMetadata;
    type Storage = ProductQuantizationStorage;

    fn build(_: &dyn Array, _: DistanceType, _: &Self::BuildParams) -> Result<Self> {
        unimplemented!("ProductQuantizer cannot be built with new index builder")
    }

    fn code_dim(&self) -> usize {
        self.num_sub_vectors()
    }

    fn column(&self) -> &'static str {
        PQ_CODE_COLUMN
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<ArrayRef> {
        let code_array = self.transform(vectors)?;
        Ok(code_array)
    }

    fn metadata_key() -> &'static str {
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
            codebook_tensor: Vec::new(),
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

/// Loader to load partitioned [VectorStore] from disk.
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    quantizer: Quantizer,
    metadata: Q::Metadata,

    ivf: IvfData,
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

        let ivf_data = IvfData::load(&reader).await?;

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
