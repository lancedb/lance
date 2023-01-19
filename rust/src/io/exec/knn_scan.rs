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

//! I/O execution pipeline

use std::pin::Pin;

use arrow_array::RecordBatch;
use futures::stream::{BoxStream, Stream};
use futures::{FutureExt, StreamExt};

use crate::dataset::Dataset;
use crate::index::vector::flat::FlatIndex;
use crate::index::vector::{Query, VectorIndex};
use crate::io::exec::ExecNode;
use crate::Result;

use super::Type;

/// K-nearest-neighbors Scan
pub(crate) struct KNNScan<'a> {
    // dataset: &'a Dataset,
    index: Box<dyn VectorIndex>,
    query: &'a Query,
    stream: Option<BoxStream<'a, dyn Stream<Item = Result<RecordBatch>>>>,
}

impl<'a> KNNScan<'a> {
    /// Create a KNNScan exec node.
    pub fn try_new(dataset: &'static Dataset, column: &str, query: &'a Query) -> Result<Self> {
        Ok(Self {
            index: Box::new(FlatIndex::try_new(dataset, column)?),
            query,
            stream: None,
        })
    }
}

impl ExecNode for KNNScan<'_> {
    const TYPE: Type = Type::KNNScan;
}

impl Stream for KNNScan<'_> {
    type Item = Result<RecordBatch>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = Pin::into_inner(self);
        if this.stream.is_none() {
            let mut stream = this.index.search(this.query).into_stream();

            this.stream = Some(Box::pin(stream)?);
        }
        let mut stream = this.index.search(this.query).into_stream();
        stream.poll_next_unpin(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_scan() {}
}
