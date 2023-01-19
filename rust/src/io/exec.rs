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

mod knn_scan;
mod scan;

use arrow_array::RecordBatch;
use futures::stream::Stream;

use crate::Result;

/// Execution Node Type
pub enum Type {
    Scan = 1,
    Project = 2,
    Filter = 3,
    Limit = 4,
    Take = 5,
    KNNScan = 6,
}

pub trait ExecNode: Stream<Item = Result<RecordBatch>> {
    /// execution node type.
    const TYPE: Type;
}
