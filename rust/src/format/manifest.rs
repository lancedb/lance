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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use chrono::prelude::*;
use prost_types::Timestamp;

use super::Fragment;
use crate::datatypes::Schema;
use crate::format::{pb, ProtoStruct};

/// Manifest of a dataset
///
///  * Schema
///  * Version
///  * Fragments.
///  * Indices.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Dataset schema.
    pub schema: Schema,

    /// Dataset version
    pub version: u64,

    /// Fragments, the pieces to build the dataset.
    pub fragments: Arc<Vec<Fragment>>,

    /// The file position of the version aux data.
    pub version_aux_data: usize,

    /// The file position of the index metadata.
    pub index_section: Option<usize>,

    /// The creation timestamp with nanosecond resolution as 128-bit integer
    pub timestamp_nanos: u128,

    /// An optional string tag for this version
    pub tag: Option<String>,
}

impl Manifest {
    pub fn new(schema: &Schema, fragments: Arc<Vec<Fragment>>) -> Self {
        let duration_since_epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();
        let timestamp_nanos = duration_since_epoch.as_nanos(); // u128

        Self {
            schema: schema.clone(),
            version: 1,
            fragments,
            version_aux_data: 0,
            index_section: None,
            timestamp_nanos,
            tag: None,
        }
    }

    /// Return the `timestamp_nanos` value as a Utc DateTime
    pub fn timestamp(&self) -> DateTime<Utc> {
        let nanos = self.timestamp_nanos % 1_000_000_000;
        let seconds = ((self.timestamp_nanos - nanos) / 1_000_000_000) as i64;
        DateTime::<Utc>::from_utc(
            NaiveDateTime::from_timestamp_opt(seconds, nanos as u32).unwrap_or(NaiveDateTime::MIN),
            Utc,
        )
    }
}

impl ProtoStruct for Manifest {
    type Proto = pb::Manifest;
}

impl From<pb::Manifest> for Manifest {
    fn from(p: pb::Manifest) -> Self {
        let timestamp_nanos = p.timestamp.map(|ts| {
            let sec = ts.seconds as u128 * 1e9 as u128;
            let nanos = ts.nanos as u128;
            sec + nanos
        });
        Self {
            schema: Schema::from(&p.fields),
            version: p.version,
            fragments: Arc::new(p.fragments.iter().map(Fragment::from).collect()),
            version_aux_data: p.version_aux_data as usize,
            index_section: p.index_section.map(|i| i as usize),
            timestamp_nanos: timestamp_nanos.unwrap_or(0),
            tag: if p.tag.is_empty() { None } else { Some(p.tag) },
        }
    }
}

impl From<&Manifest> for pb::Manifest {
    fn from(m: &Manifest) -> Self {
        let timestamp_nanos = if m.timestamp_nanos == 0 {
            None
        } else {
            let nanos = m.timestamp_nanos % 1e9 as u128;
            let seconds = ((m.timestamp_nanos - nanos) / 1e9 as u128) as i64;
            Some(Timestamp {
                seconds,
                nanos: nanos as i32,
            })
        };
        Self {
            fields: (&m.schema).into(),
            version: m.version,
            fragments: m.fragments.iter().map(pb::DataFragment::from).collect(),
            metadata: HashMap::default(),
            version_aux_data: m.version_aux_data as u64,
            index_section: m.index_section.map(|i| i as u64),
            timestamp: timestamp_nanos,
            tag: m.tag.clone().unwrap_or("".to_string()),
        }
    }
}
