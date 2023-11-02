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

//! High-performance [Apache Arrow](https://docs.rs/arrow/latest/arrow/) native Linear Algebra algorithms.

#![cfg_attr(feature = "avx512", feature(stdsimd))]

pub mod distance;
pub mod kernels;
pub mod kmeans;
pub mod matrix;
pub mod simd;

pub use matrix::MatrixView;

use arrow_schema::ArrowError;

type Error = ArrowError;
pub type Result<T> = std::result::Result<T, Error>;
