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

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use futures::stream::Stream;

use super::{ExecNode, NodeType};
use crate::index::vector::Query;
use crate::Result;

/// KNN node for post-filtering.
pub struct KNN {
    child: Box<dyn ExecNode + Unpin + Send>,

    /// Column to search.
    column: String,

    query: Query,
}

impl KNN {
    pub(crate) fn new(child: Box<dyn ExecNode + Unpin + Send>, column: &str, query: Query) -> Self {
        // assert_eq!(child.node_type(), NodeType::Scan, "")
        Self {
            child,
            column: column.to_string(),
            query,
        }
    }
}

impl ExecNode for KNN {
    fn node_type(&self) -> NodeType {
        NodeType::KnnFlat
    }
}

impl Stream for KNN {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        //Pin::into_inner(self).rx.poll_recv(cx)
        todo!()
    }
}
