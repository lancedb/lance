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

//! Testing utilities

use rand::Rng;
use std::iter::repeat_with;

use arrow_array::Float32Array;

/// Create a random float32 array.
pub fn generate_random_array(n: usize) -> Float32Array {
    let mut rng = rand::thread_rng();
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}
