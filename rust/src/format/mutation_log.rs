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

//! Dataset mutation log

use std::sync::Arc;

use super::{Fragment, Manifest};

use crate::{error::Result, format::index::Index, Dataset};

// Table operation log type
// NOTE: Unstable API, only supports append for now
pub enum MutationLog {
    Append { fragments: Vec<Fragment> },
    Overwrite { fragments: Vec<Fragment> },
}

impl MutationLog {
    /// Make a new log entry for append operation
    pub fn append(fragments: Vec<Fragment>) -> Self {
        Self::Append { fragments }
    }

    /// Apply the log to the current manifest
    pub async fn apply(self, dataset: &Dataset) -> Result<(Manifest, Option<Vec<Index>>)> {
        let manifest = dataset.manifest;

        let (mut new_manifest, new_indices) = match self {
            Self::Append { fragments } => {
                let mut fragment_id = manifest.fragments
                .iter()
                .map(|f| f.id)
                .max()
                .map(|id| id + 1)
                .unwrap_or(0);

                let mut manifest = manifest.clone();
                let mut frags = manifest.fragments.as_ref().to_owned();
                frags.reserve_exact(fragments.len());

                // reapply fragment id increments
                for mut new_frag in fragments {
                    new_frag.id = fragment_id;
                    fragment_id += 1;
                    frags.push(new_frag);
                }
                manifest.fragments = Arc::new(frags);
                (manifest, Some(dataset.load_indices().await?))
            }
            Self::Overwrite { fragments } => {
                let mut manifest = manifest.clone();
                manifest.fragments = Arc::new(fragments);
                manifest.index_section;
                // drop indices if we are overwriting
                (manifest, None)
            }
        };

        new_manifest.version += 1;

        Ok((new_manifest, new_indices))
    }
}
