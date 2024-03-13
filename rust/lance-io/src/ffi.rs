// Copyright 2024 Lance Developers.
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

use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SchemaRef};
use futures::StreamExt;
use lance_core::Result;

use crate::stream::RecordBatchStream;

#[pin_project::pin_project]
struct RecordBatchIteratorAdaptor<S: RecordBatchStream> {
    schema: SchemaRef,

    #[pin]
    stream: S,

    rt: tokio::runtime::Runtime,
}

impl<S: RecordBatchStream> RecordBatchIteratorAdaptor<S> {
    fn new(stream: S, schema: SchemaRef, rt: tokio::runtime::Runtime) -> Self {
        Self { schema, stream, rt }
    }
}

impl<S: RecordBatchStream + Unpin> arrow::record_batch::RecordBatchReader
    for RecordBatchIteratorAdaptor<S>
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl<S: RecordBatchStream + Unpin> Iterator for RecordBatchIteratorAdaptor<S> {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rt
            .block_on(async { self.stream.next().await })
            .map(|r| r.map_err(|e| ArrowError::ExternalError(Box::new(e))))
    }
}

/// Wrap a [`RecordBatchStream`] into an [FFI_ArrowArrayStream].
pub fn to_ffi_arrow_array_stream(
    stream: impl RecordBatchStream + std::marker::Unpin + 'static,
    rt: tokio::runtime::Runtime,
) -> Result<FFI_ArrowArrayStream> {
    let schema = stream.schema();
    let arrow_stream = RecordBatchIteratorAdaptor::new(stream, schema, rt);
    let reader = FFI_ArrowArrayStream::new(Box::new(arrow_stream));

    Ok(reader)
}
