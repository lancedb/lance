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

//! dot product

use arrow_array::Float32Array;

pub trait Dot {
    /// Dot product
    fn dot(&self, other: &Self) -> f32;
}

impl Dot for [f32] {
    fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..self.len() {
            sum += self[i] * other[i];
        }
        sum
    }
}

impl Dot for Float32Array {
    fn dot(&self, other: &Self) -> f32 {
        self.values().dot(other.values())
    }
}
