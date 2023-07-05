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

//! Lance Columnar Data Format
//!
//! Lance columnar data format is an alternative to Parquet. It provides 100x faster for random access,
//! automatic versioning, optimized for computer vision, bio-informatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.
//!
//! ## Features
//! - Fast random access
//! - Vector search
//! - Zero-copy, automatic versioning.
//!
//! ## Examples
//!
//! Lance API is native to [Apache Arrow](https://arrow.apache.org/).
//!
//! ```rust
//! # use std::sync::Arc;
//! use lance::Dataset;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
//! # use arrow_schema::{Field, Schema, DataType};
//!
//! # std::fs::remove_dir_all("/tmp/test.lance").unwrap();
//! # let rt = Runtime::new().unwrap();
//! let schema = Arc::new(Schema::new(vec![Field::new(
//!     "i", DataType::Int32, false)]));
//! let batches: Vec<RecordBatch> = (0..20)
//!     .map(|i| { RecordBatch::try_new(schema.clone(),
//!         vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
//!     ).unwrap() }).collect();
//! let mut reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
//!
//! # rt.block_on(async {
//!     Dataset::write(reader, "/tmp/test.lance", None).await.unwrap();
//! # })
//! ```
//!
//! Read the dataset:
//!
//! ```rust
//! # use tokio::runtime::Runtime;
//! # use lance::Dataset;
//! # let rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//!     let dataset = Dataset::open("/tmp/test.lance").await.unwrap();
//!     println!("Total records: {}", dataset.count_rows().await.unwrap());
//! # });
//! ```

pub mod arrow;
pub mod datafusion;
pub mod dataset;
pub mod datatypes;
pub mod encodings;
pub mod error;
pub mod format;
pub mod index;
pub mod io;
pub mod linalg;
pub mod session;
pub mod utils;

pub use dataset::Dataset;
pub use error::{Error, Result};
