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

//! HNSW storage - Storage implementations for HNSW graph nodes.
#![allow(clippy::ptr_arg)]

use std::sync::Arc;

use bincode::{config, Decode, Encode};
use lance_core::Result;
use rocksdb::{Options, WriteBatchWithTransaction, DB};

#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphNode {
    pub id: u64,
    pub level: u8,
    pub neighbors: Vec<u64>,

    // private data to the storage engine
    metadata: Vec<u8>,
}

impl GraphNode {
    pub fn new(id: u64, level: u8) -> Self {
        Self {
            id,
            level,
            neighbors: vec![],
            metadata: vec![],
        }
    }
}

// use a macro because the return type has generics
// if we change the settings we need to change the return type
macro_rules! bincode_config {
    () => {
        config::standard()
            .with_little_endian()
            .with_fixed_int_encoding()
            .with_no_limit()
    };
}

impl GraphNode {
    fn to_bytes(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, bincode_config!()).expect("should be able to encode")
    }

    fn try_from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::decode_from_slice(bytes, bincode_config!())
            .unwrap()
            .0)
    }
}

/// A trait for storing and retrieving graph nodes.
/// We abstract away the storage of the graph nodes so that we can
/// implement different storage backends and decide which layout is the
/// most efficient for our use case.
pub trait HNSWGraphStorage {
    fn get_entry_points(&self) -> Result<GraphNode>;

    fn set_entry_points(&self, node: &GraphNode) -> Result<()>;

    fn traverse(&self, from: &GraphNode, level: &u8, to: &[u64]) -> Result<Vec<GraphNode>>;

    fn put_node(&self, node: &GraphNode) -> Result<()>;

    fn put_nodes(&self, nodes: &[GraphNode]) -> Result<()>;
}

pub trait VectorStorage {
    fn get_vectors(&self, id: &[u64]) -> Result<Arc<Vec<f32>>>;

    fn put_vectors(&self, id: &[u64], vector: &Vec<f32>, dims: usize) -> Result<()>;
}

// storage implementation meant PoC
pub struct RocksDBGraphStorage {
    db: Box<DB>,
}

impl RocksDBGraphStorage {
    pub fn try_new(path: &str) -> Result<Self> {
        let mut db_options = Options::default();
        db_options.create_if_missing(true);
        db_options.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
        db_options.optimize_universal_style_compaction(8 * 1024 * 1024 * 1024);
        db_options.increase_parallelism(4);

        let db = Box::new(DB::open(&db_options, path).expect("should be able to open db"));
        Ok(Self { db })
    }

    fn make_key(&self, level: &u8, id: &u64) -> Vec<u8> {
        format!("{:02}:{:08}", level, id).into_bytes()
    }

    pub fn dump(&self) {
        for it in self.db.iterator(rocksdb::IteratorMode::Start) {
            let (k, v) = it.unwrap();
            let key = String::from_utf8(k.to_vec()).unwrap();
            if key == "__EP" {
                println!("ENTRY POINT: {}", String::from_utf8(v.to_vec()).unwrap());

                continue;
            }

            if key.starts_with("v:") {
                let (val, _): (Vec<f32>, _) =
                    bincode::decode_from_slice(&v[..], bincode_config!()).unwrap();
                println!("{}: {:?}", key, val);

                continue;
            }

            println!("{}: {:?}", key, GraphNode::try_from_bytes(&v).unwrap(),);
        }
    }
}

impl HNSWGraphStorage for RocksDBGraphStorage {
    fn get_entry_points(&self) -> Result<GraphNode> {
        let ep_key = self.db.get("__EP").unwrap().unwrap();
        GraphNode::try_from_bytes(&self.db.get(ep_key).unwrap().unwrap())
    }

    fn set_entry_points(&self, node: &GraphNode) -> Result<()> {
        let ep_key = self.make_key(&node.level, &node.id);
        self.db.put("__EP", ep_key).unwrap();
        self.put_node(node)?;
        Ok(())
    }

    fn traverse(&self, _: &GraphNode, level: &u8, to: &[u64]) -> Result<Vec<GraphNode>> {
        let keys = to.iter().map(|id| self.make_key(level, id));

        let mut ret = Vec::with_capacity(to.len());

        for res in self.db.multi_get(keys) {
            let node = GraphNode::try_from_bytes(&res.unwrap().unwrap()).unwrap();
            ret.push(node);
        }

        Ok(ret)
    }

    fn put_node(&self, node: &GraphNode) -> Result<()> {
        self.db
            .put(self.make_key(&node.level, &node.id), node.to_bytes())
            .unwrap();

        Ok(())
    }

    fn put_nodes(&self, nodes: &[GraphNode]) -> Result<()> {
        let mut batch = WriteBatchWithTransaction::<false>::default();
        for node in nodes {
            batch.put(self.make_key(&node.level, &node.id), node.to_bytes());
        }
        self.db.write(batch).unwrap();
        Ok(())
    }
}

impl VectorStorage for RocksDBGraphStorage {
    fn get_vectors(&self, id: &[u64]) -> Result<Arc<Vec<f32>>> {
        let mut ret = Vec::with_capacity(id.len());

        for res in self
            .db
            .multi_get(id.iter().map(|id| format!("v:{}", id).into_bytes()))
        {
            let bytes = res.expect("no error").expect("vec should exist");
            let vec: Vec<f32> = bincode::decode_from_slice(&bytes[..], bincode_config!())
                .unwrap()
                .0;

            ret.extend(vec);
        }

        Ok(Arc::new(ret))
    }

    fn put_vectors(&self, id: &[u64], vector: &Vec<f32>, dims: usize) -> Result<()> {
        for (vec, idx) in vector.chunks(dims).zip(id) {
            self.db
                .put(
                    format!("v:{}", idx).into_bytes(),
                    bincode::encode_to_vec(vec, bincode_config!())
                        .expect("should be able to encode"),
                )
                .unwrap();
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tempfile;

    #[test]
    fn test_serde() {
        let node = GraphNode {
            id: 1,
            level: 2,
            neighbors: vec![4, 5, 6],
            metadata: vec![4, 5, 6],
        };

        let bytes = node.to_bytes();
        let node2 = GraphNode::try_from_bytes(&bytes).unwrap();
        assert_eq!(node, node2);
    }

    #[test]
    fn test_db() {
        let node = GraphNode {
            id: 1,
            level: 2,
            neighbors: vec![4, 5, 6],
            metadata: vec![4, 5, 6],
        };

        let dir = tempfile::tempdir().unwrap();

        let storage = RocksDBGraphStorage::try_new(dir.path().to_str().unwrap()).unwrap();

        storage.put_node(&node).unwrap();

        storage.traverse(&node, &2, &[1]).unwrap();
    }
}
