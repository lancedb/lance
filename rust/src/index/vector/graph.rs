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

//! Graph-based vector index.
//!

use crate::Result;
use std::any::Any;

mod memory;
mod persisted;

/// Vertex (metadata). It does not include the actual data.
pub trait Vertex: Sized {
    fn byte_length(&self) -> usize;

    fn from_bytes(data: &[u8]) -> Result<Self>;

    fn as_any(&self) -> &dyn Any;
}
