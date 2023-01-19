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

use super::Fragment;
use crate::datatypes::Schema;
use crate::format::{pb, ProtoStruct};
use std::collections::HashMap;

/// Manifest of a dataset
///
///  * Schema
///  * Version
///  * Fragments.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Dataset schema.
    pub schema: Schema,

    /// Dataset version
    pub version: u64,

    /// Fragments, the pieces to build the dataset.
    pub fragments: Vec<Fragment>,
}

impl Manifest {
    pub fn new(schema: &Schema) -> Self {
        Self {
            schema: schema.clone(),
            version: 1,
            fragments: vec![],
        }
    }
}

impl ProtoStruct for Manifest {
    type Proto = pb::Manifest;
}

impl From<pb::Manifest> for Manifest {
    fn from(p: pb::Manifest) -> Self {
        Self {
            schema: Schema::from(&p.fields),
            version: p.version,
            fragments: p.fragments.iter().map(Fragment::from).collect(),
        }
    }
}

impl From<&Manifest> for pb::Manifest {
    fn from(m: &Manifest) -> Self {
        Self {
            fields: (&m.schema).into(),
            version: m.version,
            fragments: m.fragments.iter().map(pb::DataFragment::from).collect(),
            metadata: HashMap::default(),
            version_aux_data: 0,
        }
    }
}
