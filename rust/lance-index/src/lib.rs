// Copyright 2023 Lance Developers.
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

use std::any::Any;
use std::fmt;

use serde_json;

use lance_core::Result;

pub mod scalar;
pub mod vector;

pub mod pb {
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.index.pb.rs"));
}


/// Trait of a secondary index.
pub trait Index: Send + Sync {
    /// Cast to [Any].
    fn as_any(&self) -> &dyn Any;

    // TODO: if we ever make this public, do so in such a way that `serde_json`
    // isn't exposed at the interface. That way mismatched versions isn't an issue.
    fn statistics(&self) -> Result<serde_json::Value>;
}

/// Index Type
pub enum IndexType {
    // Preserve 0-100 for simple indices.

    // 100+ and up for vector index.
    /// Flat vector index.
    Vector = 100,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Vector => write!(f, "Vector"),
        }
    }
}