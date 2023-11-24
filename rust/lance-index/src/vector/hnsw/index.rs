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
#![allow(clippy::derive_ord_xor_partial_ord)]

use std::{
    cmp::min,
    collections::{BTreeSet, HashMap, HashSet},
};

use lance_core::Result;
use moka::sync::Cache;
use rand::Rng;

use super::storage::{GraphNode, HNSWGraphStorage, VectorStorage};

#[derive(Debug)]
pub struct HNSWIndex<S>
where
    S: HNSWGraphStorage + VectorStorage,
{
    storage: S,

    // construction params
    m_l: f32,
    l_max: u8,
    m_max: usize,
    ef_construction: usize,

    dist_cache: Cache<(u64, u64), f32>,

    indexing: bool,

    build_time_nodes: HashMap<(u8, u64), GraphNode>,

    visited: HashSet<u64>,
}

pub struct VecAndRowId<'a>(pub &'a [f32], pub u64);

#[derive(Clone, Debug, PartialEq, PartialOrd)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// see https://arxiv.org/pdf/1603.09320.pdf
// for the original paperx
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeWithDist(OrderedFloat, GraphNode);

impl<S> HNSWIndex<S>
where
    S: HNSWGraphStorage + VectorStorage,
{
    #[allow(dead_code)]
    pub fn new(
        storage: S,
        m_l: f32,
        l_max: u8,
        m_max: usize,
        ef_construction: usize,
        total_vecs: Option<usize>,
    ) -> Self {
        let tem_storage_size = total_vecs.map(|x| x as f32 * m_l * 1.5);
        Self {
            storage,
            m_l,
            l_max,
            m_max,
            ef_construction,
            dist_cache: Cache::new(5_000_000),
            indexing: true,
            build_time_nodes: HashMap::with_capacity(tem_storage_size.unwrap_or(0.0) as usize),
            visited: HashSet::with_capacity(ef_construction * l_max as usize),
        }
    }

    pub fn query_mode(&mut self) {
        self.indexing = false;
    }

    fn random_l(&self) -> u8 {
        let mut rng = rand::thread_rng();

        let f: f32 = rng.gen_range(0.0..1.0);

        let l = (-1.0 * f.ln() * self.m_l).floor() as u8;

        // clip to a maximum layer number
        min(l, self.l_max)
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;

        for i in 0..a.len() {
            sum += (a[i] - b[i]).powi(2);
        }

        sum.sqrt()
    }

    // TODO: add caching to this function
    fn get_distance(&self, from: u64, to: u64) -> f32 {
        let key = if from > to { (to, from) } else { (from, to) };

        self.dist_cache.get_with(key, || {
            let vectors = self.storage.get_vectors(&[from, to]).unwrap();
            self.distance(&vectors[..vectors.len() / 2], &vectors[vectors.len() / 2..])
        })
    }

    fn traverse(&self, from: &GraphNode, level: &u8, to: &[u64]) -> Result<Vec<GraphNode>> {
        if self.indexing {
            let mut ret = Vec::with_capacity(to.len() + 1);
            for node in to {
                let n = self
                    .build_time_nodes
                    .get(&(*level, *node))
                    .expect("should always have nodes at build time");
                ret.push(n.clone());
            }
            return Ok(ret);
        }
        self.storage.traverse(from, level, to)
    }

    fn put_node(&mut self, node: &GraphNode) -> Result<()> {
        if self.indexing {
            self.build_time_nodes
                .insert((node.level, node.id), node.clone());
            return Ok(());
        }
        self.storage.put_node(node)
    }

    fn search_layer(
        &mut self,
        q: &[f32],
        ep: GraphNode,
        ef: usize,
        l_c: u8,
    ) -> BTreeSet<NodeWithDist> {
        let mut c: BTreeSet<NodeWithDist> = BTreeSet::new();
        let mut w: BTreeSet<NodeWithDist> = BTreeSet::new();

        let ep = if l_c != ep.level {
            self.traverse(&ep, &l_c, &[ep.id]).unwrap().pop().unwrap()
        } else {
            ep
        };

        c.insert(NodeWithDist(
            OrderedFloat(self.distance(q, self.storage.get_vectors(&[ep.id]).unwrap().as_ref())),
            ep,
        ));

        while let Some(elem) = c.pop_first() {
            if let Some(f) = w.last() {
                if f.0 < elem.0 {
                    break;
                }
            }

            let to_fetch = &elem.1.neighbors;

            // this awkward code is to avoid malloc from vec resize
            let mut neighbor_nodes = self.traverse(&elem.1, &l_c, to_fetch).unwrap();
            neighbor_nodes.extend(self.traverse(&elem.1, &l_c, &[elem.1.id]).unwrap());
            // add the center to the neighbors

            for neighbor in neighbor_nodes {
                if self.visited.contains(&neighbor.id) {
                    continue;
                }

                self.visited.insert(neighbor.id);

                if let Some(f) = w.last() {
                    // have found enough elements and this one is too far aways
                    if w.len() >= ef && f.0 < elem.0 {
                        continue;
                    }
                }

                let node = NodeWithDist(
                    OrderedFloat(self.distance(
                        q,
                        self.storage.get_vectors(&[neighbor.id]).unwrap().as_ref(),
                    )),
                    neighbor,
                );

                c.insert(node.clone());
                w.insert(node);

                if w.len() > ef {
                    w.pop_last();
                }
            }
        }

        self.visited.clear();

        w
    }

    fn select_neighbors_simple(
        &self,
        mut c: BTreeSet<NodeWithDist>,
        m: usize,
    ) -> BTreeSet<NodeWithDist> {
        while c.len() > m {
            c.pop_last();
        }

        c
    }

    fn prune_connections(&self, node: &mut GraphNode) {
        if node.neighbors.len() <= self.m_max {
            return;
        }

        let mut r: Vec<(OrderedFloat, u64)> = Vec::new();
        let mut w = BTreeSet::new();

        for neighbor in &node.neighbors {
            w.insert((
                OrderedFloat(self.get_distance(node.id, *neighbor)),
                *neighbor,
            ));
        }

        while !w.is_empty() && r.len() < self.m_max {
            let e = w.pop_first().unwrap();
            let mut should_include = true;
            for r_elem in &r {
                if self.get_distance(e.1, r_elem.1) < self.get_distance(e.1, node.id) {
                    should_include = false;
                    break;
                }
            }
            if should_include {
                r.push(e);
            }
        }

        node.neighbors.clear();
        node.neighbors.extend(r.iter().map(|n| n.1));
    }

    pub fn index<'a>(&mut self, vectors: impl IntoIterator<Item = VecAndRowId<'a>>) -> Result<()> {
        let mut first = true;
        for VecAndRowId(vec, id) in vectors {
            self.storage.put_vectors(&[id], &vec.to_vec(), vec.len())?;
            let l = self.random_l();

            if first {
                for l_c in (0..=l).rev() {
                    let node = GraphNode::new(id, l_c);

                    self.put_node(&node)?;

                    if l_c == l {
                        self.storage.set_entry_points(&node)?;
                    }
                }

                first = false;
                continue;
            }

            let mut ep = self.storage.get_entry_points()?;
            let l_start = ep.level;
            let mut w: BTreeSet<NodeWithDist> = BTreeSet::new();
            for l_c in (l + 1..=l_start).rev() {
                let mut result = self.search_layer(vec, ep, 1, l_c);
                let next_ep = result.pop_last().expect("should have at least one element");
                w.insert(next_ep.clone());
                ep = next_ep.1;
            }

            // l could be greater than l_start
            // but we won't be able to insert to those layers yet, since they don't exist
            for l_c in (0..=min(l, l_start)).rev() {
                let result = self.search_layer(vec, ep.clone(), self.ef_construction, l_c);
                let result = self.select_neighbors_simple(result, self.m_max);

                let mut node = GraphNode::new(id, l_c);

                node.neighbors = result.iter().map(|n| n.1.id).collect();

                self.put_node(&node)?;

                for NodeWithDist(_dist, mut neighbor) in result {
                    neighbor.neighbors.push(id);
                    self.prune_connections_smart(&mut neighbor);
                    self.put_node(&neighbor)?;
                }
            }

            for l_c in min(l, l_start) + 1..=l {
                let node = GraphNode::new(id, l_c);
                self.put_node(&node)?;

                self.storage.set_entry_points(&node)?;
            }
        }

        for node in self.build_time_nodes.values() {
            self.storage.put_node(node)?;
        }

        Ok(())
    }

    pub fn search(&mut self, q: &[f32], k: usize, ef: usize) -> Result<Vec<u64>> {
        let mut ep = self.storage.get_entry_points()?;

        for l_c in (1..=ep.level).rev() {
            let next_ep = self
                .search_layer(q, ep, 1, l_c)
                .pop_first()
                .expect("should have at least one element");
            ep = next_ep.1;
        }

        let w = self.search_layer(q, ep, ef, 0);

        Ok(self
            .select_neighbors_simple(w, k)
            .iter()
            .map(|n| n.1.id)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::hnsw::storage::RocksDBGraphStorage;

    use super::*;

    #[test]
    fn test_index() {
        let dir = tempfile::tempdir().unwrap();

        let storage = RocksDBGraphStorage::try_new(dir.path().to_str().unwrap()).unwrap();
        let mut index = HNSWIndex::new(storage, 0.33, 5, 12, 400, Some(2000));

        let mut rng = rand::thread_rng();

        let vecs: Vec<_> = (0..2000)
            .map(|_| (0..64).map(|_| rng.gen::<f32>()).collect::<Vec<_>>())
            .collect();

        index
            .index((0..2000).map(|id| VecAndRowId(&vecs[id], id as u64)))
            .unwrap();

        let mut hits = 0;
        for _ in 0..300 {
            let id = rng.gen_range(0..2000);
            if index.search(&vecs[id], 1, 200).unwrap() == vec![id as u64] {
                hits += 1;
            }
        }

        // Top1 recall > 90%
        assert!(hits > 270);
    }
}
