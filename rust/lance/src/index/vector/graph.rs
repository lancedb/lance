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

use std::any::Any;
use std::sync::Arc;

use arrow_array::UInt32Array;
use async_trait::async_trait;
use ordered_float::OrderedFloat;

pub mod builder;
pub mod persisted;

use crate::Result;
pub use persisted::*;

/// Graph
#[async_trait]
pub trait Graph {
    /// Distance between two vertices, specified by their IDs.
    async fn distance(&self, a: usize, b: usize) -> Result<f32>;

    /// Distance from query vector to a vertex identified by the idx.
    async fn distance_to(&self, query: &[f32], idx: usize) -> Result<f32>;

    /// Return the neighbor IDs.
    async fn neighbors(&self, id: usize) -> Result<Arc<UInt32Array>>;
}

/// Vertex (metadata). It does not include the actual data.
pub trait Vertex: Clone {
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn vector(&self) -> &[f32];
}

/// Vertex SerDe. Used for serializing and deserializing the vertex.
pub trait VertexSerDe<V: Vertex> {
    /// The size of the serialized vertex, in bytes.
    fn size(&self) -> usize;

    /// Serialize the vertex into a buffer.
    fn serialize(&self, vertex: &V) -> Vec<u8>;

    /// Deserialize the vertex from the buffer.
    fn deserialize(&self, data: &[u8]) -> Result<V>;
}

/// Vertex With Distance. Used for traversing the graph.
pub struct VertexWithDistance {
    /// Vertex ID.
    pub id: usize,

    /// Distance to the query.
    pub distance: OrderedFloat<f32>,
}

impl VertexWithDistance {
    pub fn new(id: usize, distance: f32) -> Self {
        Self {
            id,
            distance: OrderedFloat(distance),
        }
    }
}

impl PartialEq for VertexWithDistance {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for VertexWithDistance {}

impl PartialOrd for VertexWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VertexWithDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}
