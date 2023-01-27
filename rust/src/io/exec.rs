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

use arrow_array::RecordBatch;
use futures::stream::Stream;

mod knn;
mod limit;
mod scan;
mod take;

use crate::Result;
pub(crate) use knn::*;
pub(crate) use limit::Limit;
pub(crate) use scan::Scan;
pub(crate) use take::Take;

#[derive(Debug)]
pub enum NodeType {
    /// Dataset Scan
    Scan = 1,
    /// Dataset Take (row_ids).
    Take = 2,
    /// Limit / offset
    Limit = 4, // Filter can be 3
    /// Knn Flat Scan
    KnnFlat = 10,
    /// Knn Index Scan
    Knn = 11,
}

/// I/O Exec Node
pub(crate) trait ExecNode: Stream<Item = Result<RecordBatch>> {
    fn node_type(&self) -> NodeType;
}

pub(crate) type ExecNodeBox = Box<dyn ExecNode<Item = Result<RecordBatch>> + Unpin + Send>;
