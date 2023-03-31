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

/// A vertex in graph.
#[derive(Debug)]
pub struct Vertex<T> {
    /// Vertex ID
    pub id: u32,

    /// neighbors
    pub neighbors: Vec<u32>,

    pub auxilary: T,
}

pub(crate) trait Graph<T> {
    fn vertex(&self, id: u32) -> &Vertex<T>;

    fn distance(&self, from: u32, to: u32) -> f32;
}
