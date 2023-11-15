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
use std::sync::{Arc, RwLock};

use chrono::prelude::*;
use prost_types::Timestamp;

use super::Fragment;
use crate::datatypes::Schema;
use crate::error::{Error, Result};
use crate::format::{pb, ProtoStruct};
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

    /// Version of the writer library that wrote this manifest.
    pub writer_version: Option<WriterVersion>,

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
}

impl Manifest {
    pub fn new(schema: &Schema, fragments: Arc<Vec<Fragment>>) -> Self {
        Self {
            schema: schema.clone(),
            version: 1,
            writer_version: Some(WriterVersion::default()),
            fragments,
            version_aux_data: 0,
            index_section: None,
            timestamp_nanos: 0,
            tag: None,
            reader_feature_flags: 0,
            writer_feature_flags: 0,
            max_fragment_id: 0,
            transaction_file: None,
        }
    }

    pub fn new_from_previous(
        previous: &Self,
        schema: &Schema,
        fragments: Arc<Vec<Fragment>>,
    ) -> Self {
        Self {
            schema: schema.clone(),
            version: previous.version + 1,
            writer_version: Some(WriterVersion::default()),
            fragments,
            version_aux_data: 0,
            index_section: None, // Caller should update index if they want to keep them.
            timestamp_nanos: 0,  // This will be set on commit
            tag: None,
            reader_feature_flags: 0, // These will be set on commit
            writer_feature_flags: 0, // These will be set on commit
            max_fragment_id: previous.max_fragment_id,
            transaction_file: None,
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
    pub fn set_timestamp(&mut self, nanos: u128) {
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
        let start = since.max_fragment_id();
        Ok(self
            .fragments
            .iter()
            .filter(|&f| start.map(|s| f.id > s).unwrap_or(true))
            .cloned()
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WriterVersion {
    pub library: String,
    pub version: String,
}

impl WriterVersion {
    /// Try to parse the version string as a semver string. Returns None if
    /// not successful.
    pub fn semver(&self) -> Option<(u32, u32, u32, Option<&str>)> {
        let mut parts = self.version.split('.');
        let major = parts.next().unwrap_or("0").parse().ok()?;
        let minor = parts.next().unwrap_or("0").parse().ok()?;
        let patch = parts.next().unwrap_or("0").parse().ok()?;
        let tag = parts.next();
        Some((major, minor, patch, tag))
    }

    /// Return true if self is older than the given major/minor/patch
    pub fn older_than(&self, major: u32, minor: u32, patch: u32) -> bool {
        let version = self
            .semver()
            .expect(&format!("Invalid writer version: {}", self.version));
        return (version.0, version.1, version.2) < (major, minor, patch);
    }
}

lazy_static::lazy_static! {
    static ref DEFAULT_WRITER_VER: RwLock<String> =
    RwLock::new(env!("CARGO_PKG_VERSION").to_string());
}

// Allows us to adjust the default writer version (useful for testing)
pub(crate) fn set_default_writer_version(value: String) {
    let mut default_val = DEFAULT_WRITER_VER.write().unwrap();
    *default_val = value;
}

impl Default for WriterVersion {
    fn default() -> Self {
        Self {
            library: "lance".to_string(),
            version: DEFAULT_WRITER_VER.read().unwrap().clone(),
        }
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
        // We only use the writer version if it is fully set.
        let writer_version = match p.writer_version {
            Some(pb::manifest::WriterVersion { library, version }) => {
                Some(WriterVersion { library, version })
            }
            _ => None,
        };
        Self {
            schema: Schema::from((&p.fields, p.metadata)),
            version: p.version,
            writer_version,
            fragments: Arc::new(p.fragments.iter().map(Fragment::from).collect()),
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
            writer_version: m
                .writer_version
                .as_ref()
                .map(|wv| pb::manifest::WriterVersion {
                    library: wv.library.clone(),
                    version: wv.version.clone(),
                }),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_version() {
        let wv = WriterVersion::default();
        assert_eq!(wv.library, "lance");
        assert_eq!(wv.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(
            wv.semver(),
            Some((
                env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
                None
            ))
        );
    }
}
