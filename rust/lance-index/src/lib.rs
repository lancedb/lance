// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance secondary index library
//!
//! <section class="warning">
//! This is internal crate used by <a href="https://github.com/lancedb/lance">the lance project</a>.
//! <br/>
//! API stability is not guaranteed.
//! </section>

use std::{any::Any, sync::Arc};

use crate::frag_reuse::FRAG_REUSE_INDEX_NAME;
use crate::mem_wal::MEM_WAL_INDEX_NAME;
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::convert::TryFrom;

pub mod frag_reuse;
pub mod mem_wal;
pub mod metrics;
pub mod optimize;
pub mod prefilter;
pub mod scalar;
pub mod traits;
pub mod vector;

pub use crate::traits::*;

pub const INDEX_FILE_NAME: &str = "index.idx";
/// The name of the auxiliary index file.
///
/// This file is used to store additional information about the index, to improve performance.
/// - For 'IVF_HNSW' index, it stores the partitioned PQ Storage.
pub const INDEX_AUXILIARY_FILE_NAME: &str = "auxiliary.idx";
pub const INDEX_METADATA_SCHEMA_KEY: &str = "lance:index";

// Currently all vector indexes are version 1
pub const VECTOR_INDEX_VERSION: u32 = 1;

/// The factor of threshold to trigger split / join for vector index.
///
/// If the number of rows in the single partition is greater than `MAX_PARTITION_SIZE_FACTOR * target_partition_size`,
/// the partition will be split.
/// If the number of rows in the single partition is less than `MIN_PARTITION_SIZE_PERCENT *target_partition_size / 100`,
/// the partition will be joined.
pub const MAX_PARTITION_SIZE_FACTOR: usize = 4;
pub const MIN_PARTITION_SIZE_PERCENT: usize = 25;

pub mod pb {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.index.pb.rs"));
}

pub mod pbold {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.table.rs"));
}

/// Generic methods common across all types of secondary indices
///
#[async_trait]
pub trait Index: Send + Sync + DeepSizeOf {
    /// Cast to [Any].
    fn as_any(&self) -> &dyn Any;

    /// Cast to [Index]
    fn as_index(self: Arc<Self>) -> Arc<dyn Index>;

    /// Cast to [vector::VectorIndex]
    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn vector::VectorIndex>>;

    /// Retrieve index statistics as a JSON Value
    fn statistics(&self) -> Result<serde_json::Value>;

    /// Prewarm the index.
    ///
    /// This will load the index into memory and cache it.
    async fn prewarm(&self) -> Result<()>;

    /// Get the type of the index
    fn index_type(&self) -> IndexType;

    /// Read through the index and determine which fragment ids are covered by the index
    ///
    /// This is a kind of slow operation.  It's better to use the fragment_bitmap.  This
    /// only exists for cases where the fragment_bitmap has become corrupted or missing.
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap>;
}

/// Index Type
#[derive(Debug, PartialEq, Eq, Copy, Hash, Clone, DeepSizeOf)]
pub enum IndexType {
    // Preserve 0-100 for simple indices.
    Scalar = 0, // Legacy scalar index, alias to BTree

    BTree = 1, // BTree

    Bitmap = 2, // Bitmap

    LabelList = 3, // LabelList

    Inverted = 4, // Inverted

    NGram = 5, // NGram

    FragmentReuse = 6,

    MemWal = 7,

    ZoneMap = 8, // ZoneMap

    BloomFilter = 9, // Bloom filter

    // 100+ and up for vector index.
    /// Flat vector index.
    Vector = 100, // Legacy vector index, alias to IvfPq
    IvfFlat = 101,
    IvfSq = 102,
    IvfPq = 103,
    IvfHnswSq = 104,
    IvfHnswPq = 105,
    IvfHnswFlat = 106,
    IvfRq = 107,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Scalar | Self::BTree => write!(f, "BTree"),
            Self::Bitmap => write!(f, "Bitmap"),
            Self::LabelList => write!(f, "LabelList"),
            Self::Inverted => write!(f, "Inverted"),
            Self::NGram => write!(f, "NGram"),
            Self::FragmentReuse => write!(f, "FragmentReuse"),
            Self::MemWal => write!(f, "MemWal"),
            Self::ZoneMap => write!(f, "ZoneMap"),
            Self::BloomFilter => write!(f, "BloomFilter"),
            Self::Vector | Self::IvfPq => write!(f, "IVF_PQ"),
            Self::IvfFlat => write!(f, "IVF_FLAT"),
            Self::IvfSq => write!(f, "IVF_SQ"),
            Self::IvfHnswSq => write!(f, "IVF_HNSW_SQ"),
            Self::IvfHnswPq => write!(f, "IVF_HNSW_PQ"),
            Self::IvfHnswFlat => write!(f, "IVF_HNSW_FLAT"),
            Self::IvfRq => write!(f, "IVF_RQ"),
        }
    }
}

impl TryFrom<i32> for IndexType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            v if v == Self::Scalar as i32 => Ok(Self::Scalar),
            v if v == Self::BTree as i32 => Ok(Self::BTree),
            v if v == Self::Bitmap as i32 => Ok(Self::Bitmap),
            v if v == Self::LabelList as i32 => Ok(Self::LabelList),
            v if v == Self::NGram as i32 => Ok(Self::NGram),
            v if v == Self::Inverted as i32 => Ok(Self::Inverted),
            v if v == Self::FragmentReuse as i32 => Ok(Self::FragmentReuse),
            v if v == Self::MemWal as i32 => Ok(Self::MemWal),
            v if v == Self::ZoneMap as i32 => Ok(Self::ZoneMap),
            v if v == Self::BloomFilter as i32 => Ok(Self::BloomFilter),
            v if v == Self::Vector as i32 => Ok(Self::Vector),
            v if v == Self::IvfFlat as i32 => Ok(Self::IvfFlat),
            v if v == Self::IvfSq as i32 => Ok(Self::IvfSq),
            v if v == Self::IvfPq as i32 => Ok(Self::IvfPq),
            v if v == Self::IvfHnswSq as i32 => Ok(Self::IvfHnswSq),
            v if v == Self::IvfHnswPq as i32 => Ok(Self::IvfHnswPq),
            v if v == Self::IvfHnswFlat as i32 => Ok(Self::IvfHnswFlat),
            _ => Err(Error::InvalidInput {
                source: format!("the input value {} is not a valid IndexType", value).into(),
                location: location!(),
            }),
        }
    }
}

impl TryFrom<&str> for IndexType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "BTree" => Ok(Self::BTree),
            "Bitmap" => Ok(Self::Bitmap),
            "LabelList" => Ok(Self::LabelList),
            "Inverted" => Ok(Self::Inverted),
            "NGram" => Ok(Self::NGram),
            "FragmentReuse" => Ok(Self::FragmentReuse),
            "MemWal" => Ok(Self::MemWal),
            "ZoneMap" => Ok(Self::ZoneMap),
            "Vector" => Ok(Self::Vector),
            "IVF_FLAT" => Ok(Self::IvfFlat),
            "IVF_SQ" => Ok(Self::IvfSq),
            "IVF_PQ" => Ok(Self::IvfPq),
            "IVF_RQ" => Ok(Self::IvfRq),
            "IVF_HNSW_FLAT" => Ok(Self::IvfHnswFlat),
            "IVF_HNSW_SQ" => Ok(Self::IvfHnswSq),
            "IVF_HNSW_PQ" => Ok(Self::IvfHnswPq),
            _ => Err(Error::invalid_input(
                format!("invalid index type: {}", value),
                location!(),
            )),
        }
    }
}

impl IndexType {
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            Self::Scalar
                | Self::BTree
                | Self::Bitmap
                | Self::LabelList
                | Self::Inverted
                | Self::NGram
                | Self::ZoneMap
                | Self::BloomFilter
        )
    }

    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            Self::Vector
                | Self::IvfPq
                | Self::IvfHnswSq
                | Self::IvfHnswPq
                | Self::IvfHnswFlat
                | Self::IvfFlat
                | Self::IvfSq
                | Self::IvfRq
        )
    }

    pub fn is_system(&self) -> bool {
        matches!(self, Self::FragmentReuse | Self::MemWal)
    }

    /// Returns the current format version of the index type,
    /// bump this when the index format changes.
    /// Indices which higher version than these will be ignored for compatibility,
    /// This would happen when creating index in a newer version of Lance,
    /// but then opening the index in older version of Lance
    pub fn version(&self) -> i32 {
        match self {
            Self::Scalar => 0,
            Self::BTree => 0,
            Self::Bitmap => 0,
            Self::LabelList => 0,
            Self::Inverted => 0,
            Self::NGram => 0,
            Self::FragmentReuse => 0,
            Self::MemWal => 0,
            Self::ZoneMap => 0,
            Self::BloomFilter => 0,

            // for now all vector indices are built by the same builder,
            // so they share the same version.
            Self::Vector
            | Self::IvfFlat
            | Self::IvfSq
            | Self::IvfPq
            | Self::IvfHnswSq
            | Self::IvfHnswPq
            | Self::IvfHnswFlat
            | Self::IvfRq => 1,
        }
    }

    /// Returns the target partition size for the index type.
    ///
    /// This is used to compute the number of partitions for the index.
    /// The partition size is optimized for the best performance of the index.
    ///
    /// This is for vector indices only.
    pub fn target_partition_size(&self) -> usize {
        match self {
            Self::Vector => 8192,
            Self::IvfFlat => 4096,
            Self::IvfSq => 8192,
            Self::IvfPq => 8192,
            Self::IvfHnswFlat => 1 << 20,
            Self::IvfHnswSq => 1 << 20,
            Self::IvfHnswPq => 1 << 20,
            _ => 8192,
        }
    }
}

pub trait IndexParams: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn index_name(&self) -> &str;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IndexMetadata {
    #[serde(rename = "type")]
    pub index_type: String,
    pub distance_type: String,
}

pub fn is_system_index(index_meta: &lance_table::format::IndexMetadata) -> bool {
    index_meta.name == FRAG_REUSE_INDEX_NAME || index_meta.name == MEM_WAL_INDEX_NAME
}

pub fn infer_system_index_type(
    index_meta: &lance_table::format::IndexMetadata,
) -> Option<IndexType> {
    if index_meta.name == FRAG_REUSE_INDEX_NAME {
        Some(IndexType::FragmentReuse)
    } else if index_meta.name == MEM_WAL_INDEX_NAME {
        Some(IndexType::MemWal)
    } else {
        None
    }
}
