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

use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::convert::TryFrom;

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

pub mod pb {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.index.pb.rs"));
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

    // 100+ and up for vector index.
    /// Flat vector index.
    Vector = 100, // Legacy vector index, alias to IvfPq
    IvfFlat = 101,
    IvfSq = 102,
    IvfPq = 103,
    IvfHnswSq = 104,
    IvfHnswPq = 105,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Scalar | Self::BTree => write!(f, "BTree"),
            Self::Bitmap => write!(f, "Bitmap"),
            Self::LabelList => write!(f, "LabelList"),
            Self::Inverted => write!(f, "Inverted"),
            Self::Vector | Self::IvfPq => write!(f, "IVF_PQ"),
            Self::IvfFlat => write!(f, "IVF_FLAT"),
            Self::IvfSq => write!(f, "IVF_SQ"),
            Self::IvfHnswSq => write!(f, "IVF_HNSW_SQ"),
            Self::IvfHnswPq => write!(f, "IVF_HNSW_PQ"),
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
            v if v == Self::Inverted as i32 => Ok(Self::Inverted),
            v if v == Self::Vector as i32 => Ok(Self::Vector),
            v if v == Self::IvfFlat as i32 => Ok(Self::IvfFlat),
            v if v == Self::IvfSq as i32 => Ok(Self::IvfSq),
            v if v == Self::IvfPq as i32 => Ok(Self::IvfPq),
            v if v == Self::IvfHnswSq as i32 => Ok(Self::IvfHnswSq),
            v if v == Self::IvfHnswPq as i32 => Ok(Self::IvfHnswPq),
            _ => Err(Error::InvalidInput {
                source: format!("the input value {} is not a valid IndexType", value).into(),
                location: location!(),
            }),
        }
    }
}

impl IndexType {
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            Self::Scalar | Self::BTree | Self::Bitmap | Self::LabelList | Self::Inverted
        )
    }

    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            Self::Vector
                | Self::IvfPq
                | Self::IvfHnswSq
                | Self::IvfHnswPq
                | Self::IvfFlat
                | Self::IvfSq
        )
    }
}

pub trait IndexParams: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn index_type(&self) -> IndexType;

    fn index_name(&self) -> &str;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IndexMetadata {
    #[serde(rename = "type")]
    pub index_type: String,
    pub distance_type: String,
}
