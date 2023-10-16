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

use roaring::RoaringBitmap;
use uuid::Uuid;

use super::*;

use crate::dataset::Dataset;
use crate::{Error, Result};
use lance_core::format::pb;
use snafu::{location, Location};

/// Index metadata
#[derive(Debug, Clone)]
pub struct Index {
    /// Unique ID across all dataset versions.
    pub uuid: Uuid,

    /// Fields to build the index.
    pub fields: Vec<i32>,

    /// Human readable index name
    pub name: String,

    /// The latest version of the dataset this index covers
    pub dataset_version: u64,

    /// The fragment ids this index covers.
    ///
    /// If this is None, then this is unknown.
    pub fragment_bitmap: Option<RoaringBitmap>,
}

impl Index {
    /// Returns the fragment ids that are not indexed by this index.
    pub async fn unindexed_fragments(&self, dataset: &Dataset) -> Result<Vec<Fragment>> {
        if self.dataset_version == dataset.version().version {
            return Ok(vec![]);
        }
        if let Some(bitmap) = self.fragment_bitmap.as_ref() {
            Ok(dataset
                .fragments()
                .iter()
                .filter(|f| !bitmap.contains(f.id as u32))
                .cloned()
                .collect::<Vec<_>>())
        } else {
            let ds = dataset.checkout_version(self.dataset_version).await?;
            let max_fragment_id_idx = ds.manifest.max_fragment_id().ok_or_else(|| Error::IO {
                message: "No fragments in index version".to_string(),
                location: location!(),
            })?;
            let max_fragment_id_ds =
                dataset
                    .manifest
                    .max_fragment_id()
                    .ok_or_else(|| Error::IO {
                        message: "No fragments in dataset version".to_string(),
                        location: location!(),
                    })?;
            if max_fragment_id_idx < max_fragment_id_ds {
                dataset.manifest.fragments_since(&ds.manifest)
            } else {
                Ok(vec![])
            }
        }
    }
}

impl TryFrom<&pb::IndexMetadata> for Index {
    type Error = Error;

    fn try_from(proto: &pb::IndexMetadata) -> Result<Self> {
        let fragment_bitmap = if proto.fragment_bitmap.is_empty() {
            None
        } else {
            Some(RoaringBitmap::deserialize_from(
                &mut proto.fragment_bitmap.as_slice(),
            )?)
        };

        Ok(Self {
            uuid: proto
                .uuid
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| Error::IO {
                    message: "uuid field does not exist in Index metadata".to_string(),
                    location: location!(),
                })??,
            name: proto.name.clone(),
            fields: proto.fields.clone(),
            dataset_version: proto.dataset_version,
            fragment_bitmap,
        })
    }
}

impl From<&Index> for pb::IndexMetadata {
    fn from(idx: &Index) -> Self {
        let mut fragment_bitmap = Vec::new();
        if let Some(bitmap) = &idx.fragment_bitmap {
            if let Err(e) = bitmap.serialize_into(&mut fragment_bitmap) {
                // In theory, this should never error. But if we do, just
                // recover gracefully.
                log::error!("Failed to serialize fragment bitmap: {}", e);
                fragment_bitmap.clear();
            }
        }

        Self {
            uuid: Some((&idx.uuid).into()),
            name: idx.name.clone(),
            fields: idx.fields.clone(),
            dataset_version: idx.dataset_version,
            fragment_bitmap,
        }
    }
}
//
// impl From<&Vec<Index>> for pb::IndexSection {
//     fn from(indices: &Vec<Index>) -> Self {
//         Self {
//             indices: indices.iter().map(pb::IndexMetadata::from).collect(),
//         }
//     }
// }
