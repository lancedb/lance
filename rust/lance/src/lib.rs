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
//! automatic versioning, optimized for computer vision, bioinformatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.
//!
//!
//! # Create a Dataset
//!
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset};
//!
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #
//! # let test_dir = tempfile::tempdir().unwrap();
//! # let uri = test_dir.path().to_str().unwrap().to_string();
//! let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Int64, false)]));
//! let batches = vec![RecordBatch::new_empty(schema.clone())];
//! let reader = RecordBatchIterator::new(
//!     batches.into_iter().map(Ok), schema
//! );
//!
//! let write_params = WriteParams::default();
//! Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
//! # })
//! ```
//!
//! # Scan a Dataset
//!
//! ```rust
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, Int32Array, RecordBatchIterator, ArrayRef};
//! # use tokio::runtime::Runtime;
//! use futures::StreamExt;
//! use lance::Dataset;
//! # use lance::dataset::WriteParams;
//!
//! # let array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
//! # let batches = vec![RecordBatch::try_from_iter(vec![("test", array)]).unwrap()];
//! # let test_dir = tempfile::tempdir().unwrap();
//! # let path = test_dir.path().to_str().unwrap().to_string();
//! # let schema = batches[0].schema();
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #   let write_params = WriteParams::default();
//! #   let reader = RecordBatchIterator::new(
//! #       batches.into_iter().map(Ok), schema
//! #   );
//! #   Dataset::write(reader, &path, Some(write_params)).await.unwrap();
//! let dataset = Dataset::open(&path).await.unwrap();
//! let mut scanner = dataset.scan();
//! let batches: Vec<RecordBatch> = scanner
//!     .try_into_stream()
//!     .await
//!     .unwrap()
//!     .map(|b| b.unwrap())
//!     .collect::<Vec<RecordBatch>>()
//!     .await;
//! # })
//!
//! ```
//!
use dataset::builder::DatasetBuilder;
pub use lance_core::{datatypes, error};
pub use lance_core::{Error, Result};

pub mod arrow;
pub mod datafusion;
pub mod dataset;
pub mod index;
pub mod io;
pub mod session;
pub mod table;
pub mod utils;

pub use dataset::Dataset;

/// Creates and loads a [`Dataset`] from the given path.
/// Infers the storage backend to use from the scheme in the given table path.
///
/// For more advanced configurations use [`DatasetBuilder`].
pub async fn open_dataset<T: AsRef<str>>(table_uri: T) -> Result<Dataset> {
    DatasetBuilder::from_uri(table_uri.as_ref()).load().await
}
