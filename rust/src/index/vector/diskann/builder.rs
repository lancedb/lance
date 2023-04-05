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

use std::sync::Arc;
use std::collections::{HashSet, BinaryHeap};

use arrow_array::{Float32Array, cast::as_primitive_array, types::UInt64Type};
use arrow_select::concat::concat_batches;
use futures::stream::TryStreamExt;

use crate::dataset::{Dataset, ROW_ID};
use crate::{Result, Error};
use crate::arrow::*;

/// From the paper
const DEFAULT_R: usize = 90;
const DEFAULT_ALPHA: f32 = 1.4;
const DEFAULT_L: usize = 110;

/// A builder for DiskANN index.
pub struct Builder {
    dataset: Arc<Dataset>,

    column: String,

    /// out-degree bound (R)
    r: usize,

    /// Distance threshold
    alpha: f32,

    /// Search list size
    l: usize,
}

impl Builder {
    pub fn new(dataset: Arc<Dataset>, column: &str) -> Self {
        Self {
            dataset,
            column: column.to_string(),
            r: DEFAULT_R,
            alpha: DEFAULT_ALPHA,
            l: DEFAULT_L,
        }
    }

    /// Set the out-degree bound (R) for DiskANN.
    pub fn r(&mut self, r: usize) -> &mut Self {
        self.r = r;
        self
    }

    /// Set the distance threshold, `alpha`, in DiskANN.
    pub fn alpha(&mut self, alpha: f32) -> &mut Self {
        self.alpha = alpha;
        self
    }

    /// Set the search list size `L` in DiskANN.
    pub fn l(&mut self, l: usize) -> &mut Self {
        self.l = l;
        self
    }

    pub async fn build(&self) -> Result<()> {
        todo!()
    }
}


/// Algorithm 2 in the paper.
async fn robust_prune(
    graph: &Builder,
    id: usize,
    mut visited: HashSet<usize>,
    alpha: f32,
    r: usize,
) -> Result<Vec<u32>> {
    visited.remove(&id);
    let neighbors = graph.neighbors(id)?;
    visited.extend(neighbors.iter().map(|id| *id as usize));

    let mut heap: BinaryHeap<VertexWithDistance> = BinaryHeap::new();
    for p in visited.iter() {
        let dist = graph.distance(id, *p)?;
        heap.push(VertexWithDistance {
            id: *p,
            distance: OrderedFloat(dist),
        });
    }

    let vectors = graph.vectors.clone();
    let dim = graph.dimension;
    let new_neighbours = tokio::task::spawn_blocking(move || {
        let mut new_neighbours: Vec<usize> = vec![];
        while !visited.is_empty() {
            let mut p = heap.pop().unwrap();
            while !visited.contains(&p.id) {
                // Because we are using a heap for `argmin(Visited)` in the original
                // algorithm, we need to pop out the vertices that are not in `visited` anymore.
                p = heap.pop().unwrap();
            }

            new_neighbours.push(p.id);
            if new_neighbours.len() >= r {
                break;
            }

            let mut to_remove: HashSet<usize> = HashSet::new();
            for pv in visited.iter() {
                let dist_prime = distance(vectors.as_ref(), dim, p.id, *pv)?;
                let dist_query = distance(vectors.as_ref(), dim, id, *pv)?;
                if alpha * dist_prime <= dist_query {
                    to_remove.insert(*pv);
                }
            }
            for pv in to_remove.iter() {
                visited.remove(pv);
            }
        }
        Ok::<_, Error>(new_neighbours)
    })
    .await??;

    Ok(new_neighbours.iter().map(|id| *id as u32).collect())
}

