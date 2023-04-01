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
use ordered_float::OrderedFloat;

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

/// A Graph-backed by Lance dataset and index.

#[async_trait]
pub(crate) trait Graph {
    /// Distance between two vertices.
    async fn distance(&self, a: usize, b: usize) -> Result<f32>;

    /// Get neighbors of a vertex, specified by its ID.
    async fn neighbors(&self, id: usize) -> Result<Vec<usize>>;
}

pub(crate) struct VertexWithDistance {
    pub id: usize,
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
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for VertexWithDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}
