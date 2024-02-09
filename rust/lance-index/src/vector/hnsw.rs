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

//! Generic Graph implementation.
//!

use std::collections::BTreeMap;

use super::graph::{Graph, OrderedFloat};

pub mod builder;

pub struct HNSW {
    pub layers: Vec<Box<dyn Graph>>,
    pub entry: u32,
}

/// Select neighbors from the ordered candidate list.
/// Algorithm 3 in the HNSW paper.
fn select_neighbors(
    orderd_candidates: &BTreeMap<OrderedFloat, u32>,
    k: usize,
) -> Vec<(OrderedFloat, u32)> {
    orderd_candidates
        .iter()
        .take(k)
        .map(|(&d, &id)| (d, id))
        .collect()
}
