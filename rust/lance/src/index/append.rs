// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use lance_core::{Error, Result};
use log::info;
use uuid::Uuid;

use crate::dataset::Dataset;
use crate::format::Index as IndexMetadata;
use crate::index::vector::ivf::IVFIndex;
use crate::index::vector::open_index;

/// Append new data to the index, without re-train.
pub(crate) async fn append_index(
    dataset: Arc<Dataset>,
    old_index: &IndexMetadata,
) -> Result<Option<Uuid>> {
    let unindexed = old_index.unindexed_fragments(dataset.as_ref()).await?;
    if unindexed.is_empty() {
        return Ok(None);
    };

    let column = dataset
        .schema()
        .field_by_id(old_index.fields[0])
        .ok_or(Error::Index {
            message: format!(
                "Append index: column {} does not exist",
                old_index.fields[0]
            ),
        })?;

    let index = open_index(
        dataset.clone(),
        &column.name,
        old_index.uuid.to_string().as_str(),
    )
    .await?;

    let Some(ivf_idx) = index.as_any().downcast_ref::<IVFIndex>() else {
        info!("Index type: {:?} does not support append", index);
        return Ok(None);
    };

    let mut scanner = dataset.scan();
    scanner.with_fragments(unindexed);
    scanner.with_row_id();
    let stream = scanner.try_into_stream().await?;

    let new_index = ivf_idx
        .append(dataset.as_ref(), stream, old_index, &column.name)
        .await?;
    Ok(Some(new_index))
}
