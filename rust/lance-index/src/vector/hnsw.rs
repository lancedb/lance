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

//! HNSW graph implementation.
//!
//! Hierarchical Navigable Small World (HNSW).
//!

use std::collections::BTreeMap;
use std::sync::Arc;

use lance_core::Result;
use lance_linalg::distance::MetricType;

use super::graph::{Graph, OrderedFloat};
use crate::vector::graph::beam_search;

pub mod builder;

pub struct HNSW {
    layers: Vec<Arc<dyn Graph>>,
    metric_type: MetricType,
    entry_point: u32,
}

impl HNSW {
    fn from_builder(
        layers: Vec<Arc<dyn Graph>>,
        entry_point: u32,
        metric_type: MetricType,
    ) -> Self {
        Self {
            layers,
            metric_type,
            entry_point,
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u32>> {
        let mut ep = vec![self.entry_point];
        let num_layers = self.layers.len();
        for layer in self.layers.iter().rev().take(num_layers - 1) {
            let candidates = beam_search(layer.as_ref(), &ep, query, 1)?;
            ep = select_neighbors(&candidates, 1);
        }
        let candidates = beam_search(self.layers[0].as_ref(), &ep, query, ef)?;
        Ok(select_neighbors(&candidates, k))
    }
}

/// Select neighbors from the ordered candidate list.
/// Algorithm 3 in the HNSW paper.
fn select_neighbors(orderd_candidates: &BTreeMap<OrderedFloat, u32>, k: usize) -> Vec<u32> {
    orderd_candidates
        .iter()
        .take(k)
        .map(|(_, id)| *id)
        .collect()
}
