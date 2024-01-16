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

//! Lance secondary index library

#![cfg_attr(nightly, feature(stdsimd))]

use std::{any::Any, sync::Arc};

use async_trait::async_trait;
use lance_core::Result;
use roaring::RoaringBitmap;

pub mod optimize;
pub mod scalar;
pub mod traits;
pub mod vector;
pub use crate::traits::*;

pub const INDEX_FILE_NAME: &str = "index.idx";

pub mod pb {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.index.pb.rs"));
}

/// Generic methods common across all types of secondary indices
///
#[async_trait]
pub trait Index: Send + Sync {
    /// Cast to [Any].
    fn as_any(&self) -> &dyn Any;

    /// Cast to [Index]
    fn as_index(self: Arc<Self>) -> Arc<dyn Index>;

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
pub enum IndexType {
    // Preserve 0-100 for simple indices.
    Scalar = 0,
    // 100+ and up for vector index.
    /// Flat vector index.
    Vector = 100,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "Scalar"),
            Self::Vector => write!(f, "Vector"),
        }
    }
}

pub trait IndexParams: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}
