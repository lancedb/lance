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

//! Dot product.

pub fn dot<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter().zip(to.iter()).map(|(x, y)| x * y).sum()
}
pub trait Dot {
    type Output;

    /// Dot product.
    fn dot(&self, other: &Self) -> Self::Output;
}

impl Dot for [f32] {
    type Output = f32;

    fn dot(&self, other: &[f32]) -> f32 {
        self.iter().zip(other.iter()).map(|(x, y)| x * y).sum()
    }
}