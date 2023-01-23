// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_array::{cast::as_primitive_array, Array, FixedSizeListArray, Float32Array};

use crate::arrow::*;
use crate::index::pb;
use crate::{Error, Result};

/// Ivf Model
#[derive(Debug)]
struct Ivf {
    // A 2-D (num_partitions * dimension) of float32 array.
    // It is 64-bit aligned via Arrow.
    centroids: FixedSizeListArray,

    // Offset of each partition in the file.
    offsets: Vec<usize>,

    // Number of vectors in each partition.
    lengths: Vec<u32>,
}

/// Convert IvfModel to protobuf.
impl TryFrom<&Ivf> for pb::Ivf {
    type Error = Error;

    fn try_from(ivf: &Ivf) -> Result<Self> {
        if ivf.offsets.len() != ivf.centroids.len() {
            return Err(Error::IO("Ivf model has not been populated".to_string()));
        }
        let centroids_arr = ivf.centroids.values();
        let f32_centroids: &Float32Array = as_primitive_array(&centroids_arr);
        Ok(Self {
            centroids: f32_centroids.iter().map(|v| v.unwrap()).collect(),
            offsets: ivf.offsets.iter().map(|o| *o as u64).collect(),
            lengths: ivf.lengths.clone(),
        })
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&pb::Ivf> for Ivf {
    type Error = Error;

    fn try_from(proto: &pb::Ivf) -> Result<Self> {
        // let centroids_arr = proto.centroids.values();
        let f32_centroids: Float32Array = Float32Array::from(proto.centroids.clone());
        let dimension = f32_centroids.len() / proto.offsets.len();
        let centroids = FixedSizeListArray::try_new(f32_centroids, dimension as i32)?;
        Ok(Self {
            centroids,
            offsets: proto.offsets.iter().map(|o| *o as usize).collect(),
            lengths: proto.lengths.clone(),
        })
    }
}
