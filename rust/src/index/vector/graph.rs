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

use async_trait::async_trait;

use crate::io::object_writer::ObjectWriter;
use crate::Result;

/// A vertex in graph.
#[derive(Debug)]
pub struct Vertex<T> {
    /// Vertex ID
    pub id: u32,

    /// neighbors
    pub neighbors: Vec<u32>,

    pub(crate) aux_data: T,
}

#[async_trait]
pub(crate) trait Graph {
    /// Distance between two vertices.
    fn distance(&self, a: usize, b: usize) -> Result<f32>;

    /// Serialize to disk.
    fn serialize(&self, writer: &ObjectWriter) -> Result<()>;
}
