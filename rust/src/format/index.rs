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

//! Metadata for index

use super::pb;

/// Index metadata
#[derive(Debug, Clone)]
pub struct Index {
    /// Unique ID across all dataset versions.
    id: u64,

    /// Fields to build the index.
    fields: Vec<i32>,

    /// Human readable index name
    name: String,
}

impl From<&pb::Index> for Index {
    fn from(proto: &pb::Index) -> Self {
        Self {
            id: proto.id,
            name: proto.name.clone(),
            fields: proto.fields.clone(),
        }
    }
}

impl From<&Index> for pb::Index {
    fn from(idx: &Index) -> Self {
        Self {
            id: idx.id,
            name: idx.name.clone(),
            fields: idx.fields.clone(),
        }
    }
}
