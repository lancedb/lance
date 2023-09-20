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

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::{stream::BoxStream, Stream, StreamExt};

use crate::Result;

/// RecordBatch Stream trait.
pub trait RecordBatchStream<'a>: Stream<Item = Result<RecordBatch>> + Send + 'a {
    /// Returns the schema of the stream.
    fn schema(&self) -> SchemaRef;
}

pub struct BoxedRecordBatchStream<'a> {
    inner: BoxStream<'a, Result<RecordBatch>>,
    schema: SchemaRef,
}

impl<'a> BoxedRecordBatchStream<'a> {
    pub fn new(inner: BoxStream<'a, Result<RecordBatch>>, schema: SchemaRef) -> Self {
        Self { inner, schema }
    }
}

impl<'a> RecordBatchStream<'a> for BoxedRecordBatchStream<'a> {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for BoxedRecordBatchStream<'_> {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).inner.poll_next_unpin(cx)
    }
}
