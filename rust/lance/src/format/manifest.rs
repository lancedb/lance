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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use chrono::prelude::*;
use prost_types::Timestamp;

use super::Fragment;
use crate::datatypes::Schema;
use crate::error::{Error, Result};
use crate::format::{pb, ProtoStruct};
use crate::utils::temporal::SystemTime;
use snafu::{location, Location};
/// Manifest of a dataset
///
///  * Schema
///  * Version
///  * Fragments.
///  * Indices.
#[derive(Debug, Clone, PartialEq)]
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

    /// The reader flags
    pub reader_feature_flags: u64,

    /// The writer flags
    pub writer_feature_flags: u64,

    /// The max fragment id used so far
    pub max_fragment_id: u32,

    /// The path to the transaction file, relative to the root of the dataset
    pub transaction_file: Option<String>,

    /// Index of fragments by id. This only exists in memory and is not persisted.
    ///
    /// Keys are fragment ids and values are the index of the fragment in the `fragments` vector.
    fragment_id_index: BTreeMap<u32, usize>,
}

impl Manifest {
    pub fn new(schema: &Schema, fragments: Arc<Vec<Fragment>>) -> Self {
        let fragment_id_index = fragments
            .iter()
            .enumerate()
            .map(|(i, f)| (f.id as u32, i))
            .collect();
        Self {
            schema: schema.clone(),
            version: 1,
            fragments,
            version_aux_data: 0,
            index_section: None,
            timestamp_nanos: 0,
            tag: None,
            reader_feature_flags: 0,
            writer_feature_flags: 0,
            max_fragment_id: 0,
            transaction_file: None,
            fragment_id_index,
        }
    }

    pub fn new_from_previous(
        previous: &Self,
        schema: &Schema,
        fragments: Arc<Vec<Fragment>>,
    ) -> Self {
        let fragment_id_index = fragments
            .iter()
            .enumerate()
            .map(|(i, f)| (f.id as u32, i))
            .collect();
        Self {
            schema: schema.clone(),
            version: previous.version + 1,
            fragments,
            version_aux_data: 0,
            index_section: None, // Caller should update index if they want to keep them.
            timestamp_nanos: 0,  // This will be set on commit
            tag: None,
            reader_feature_flags: 0, // These will be set on commit
            writer_feature_flags: 0, // These will be set on commit
            max_fragment_id: previous.max_fragment_id,
            transaction_file: None,
            fragment_id_index,
        }
    }

    /// Return the `timestamp_nanos` value as a Utc DateTime
    pub fn timestamp(&self) -> DateTime<Utc> {
        let nanos = self.timestamp_nanos % 1_000_000_000;
        let seconds = ((self.timestamp_nanos - nanos) / 1_000_000_000) as i64;
        Utc.from_utc_datetime(
            &NaiveDateTime::from_timestamp_opt(seconds, nanos as u32).unwrap_or(NaiveDateTime::MIN),
        )
    }

    /// Set the `timestamp_nanos` value from a Utc DateTime
    pub fn set_timestamp(&mut self, timestamp: Option<SystemTime>) {
        let timestamp = timestamp.unwrap_or_else(SystemTime::now);
        let nanos = timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        self.timestamp_nanos = nanos;
    }

    /// Check the current fragment list and update the high water mark
    pub fn update_max_fragment_id(&mut self) {
        let max_fragment_id = self
            .fragments
            .iter()
            .map(|f| f.id)
            .max()
            .unwrap_or_default()
            .try_into()
            .unwrap();

        if max_fragment_id > self.max_fragment_id {
            self.max_fragment_id = max_fragment_id;
        }
    }

    /// Return the max fragment id.
    /// Note this does not support recycling of fragment ids.
    ///
    /// This will return None if there are no fragments.
    pub fn max_fragment_id(&self) -> Option<u64> {
        if self.max_fragment_id == 0 {
            // It might not have been updated, so the best we can do is recompute
            // it from the fragment list.
            self.fragments.iter().map(|f| f.id).max()
        } else {
            self.max_fragment_id.try_into().ok()
        }
    }

    /// Return the fragments that are newer than the given manifest.
    /// Note this does not support recycling of fragment ids.
    pub fn fragments_since(&self, since: &Self) -> Result<Vec<Fragment>> {
        if since.version >= self.version {
            return Err(Error::IO {
                message: format!(
                    "fragments_since: given version {} is newer than manifest version {}",
                    since.version, self.version
                ),
                location: location!(),
            });
        }
        let start = since.max_fragment_id().map(|id| id + 1).unwrap_or_default() as u32;

        Ok(self
            .fragment_id_index
            .range(start..)
            .map(|(_, i)| self.fragments[*i].clone())
            .collect())
    }

    pub fn fragment_by_id(&self, id: u32) -> Option<&Fragment> {
        self.fragment_id_index.get(&id).map(|i| &self.fragments[*i])
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
        let fragments: Arc<Vec<Fragment>> =
            Arc::new(p.fragments.iter().map(Fragment::from).collect());
        let fragment_id_index = fragments
            .iter()
            .enumerate()
            .map(|(i, f)| (f.id as u32, i))
            .collect();
        Self {
            schema: Schema::from((&p.fields, p.metadata)),
            version: p.version,
            fragments,
            version_aux_data: p.version_aux_data as usize,
            index_section: p.index_section.map(|i| i as usize),
            timestamp_nanos: timestamp_nanos.unwrap_or(0),
            tag: if p.tag.is_empty() { None } else { Some(p.tag) },
            reader_feature_flags: p.reader_feature_flags,
            writer_feature_flags: p.writer_feature_flags,
            max_fragment_id: p.max_fragment_id,
            transaction_file: if p.transaction_file.is_empty() {
                None
            } else {
                Some(p.transaction_file)
            },
            fragment_id_index,
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
        let (fields, metadata): (Vec<pb::Field>, HashMap<String, Vec<u8>>) = (&m.schema).into();
        Self {
            fields,
            version: m.version,
            fragments: m.fragments.iter().map(pb::DataFragment::from).collect(),
            metadata,
            version_aux_data: m.version_aux_data as u64,
            index_section: m.index_section.map(|i| i as u64),
            timestamp: timestamp_nanos,
            tag: m.tag.clone().unwrap_or_default(),
            reader_feature_flags: m.reader_feature_flags,
            writer_feature_flags: m.writer_feature_flags,
            max_fragment_id: m.max_fragment_id,
            transaction_file: m.transaction_file.clone().unwrap_or_default(),
        }
    }
}
