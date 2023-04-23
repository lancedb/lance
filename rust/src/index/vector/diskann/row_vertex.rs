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

use arrow_array::Float32Array;
use byteorder::{ByteOrder, LE};

use super::{Vertex, VertexSerDe};
use crate::Result;

/// Vertex with only Row ID.
#[derive(Clone, Debug)]
pub(crate) struct RowVertex {
    pub(crate) row_id: u64,

    #[allow(dead_code)]
    pub(crate) vector: Option<Float32Array>,
}

impl RowVertex {
    pub(crate) fn new(row_id: u64, vector: Option<Float32Array>) -> Self {
        Self { row_id, vector }
    }
}

impl Vertex for RowVertex {}

pub(crate) struct RowVertexSerDe {}

impl VertexSerDe<RowVertex> for RowVertexSerDe {
    fn size(&self) -> usize {
        8
    }

    fn serialize(&self, vertex: &RowVertex) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        buf.copy_from_slice(&vertex.row_id.to_le_bytes());
        buf
    }

    fn deserialize(&self, data: &[u8]) -> Result<RowVertex> {
        let row_id = LE::read_u64(data);
        Ok(RowVertex {
            row_id,
            vector: None,
        })
    }
}
