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

//! Secondary Index pre-filter
//!
//! Based on the query, we might have information about which fragment ids and
//! row ids can be excluded from the search.

use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;

use crate::error::Result;
use crate::Dataset;

/// Filter out row ids that we know are not relevant to the query. This currently
/// is just deleted rows.
pub struct PreFilter {
    dataset: Arc<Dataset>,
}

impl PreFilter {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self { dataset }
    }

    /// Check whether a single row id should be included in the query.
    pub async fn check_one(&self, row_id: u64) -> Result<bool> {
        let fragment_id = (row_id >> 32) as u32;
        // If the fragment isn't found, then it must have been deleted.
        let Some(fragment) = self.dataset.get_fragment(fragment_id as usize) else { return Ok(false) };
        // If the fragment has no deletion vector, then the row must be there.
        let Some(deletion_vector) = fragment.get_deletion_vector().await? else { return Ok(true) };
        let local_row_id = row_id as u32;
        Ok(!deletion_vector.contains(local_row_id))
    }

    /// Check whether a slice of row ids should be included in a query.
    ///
    /// Returns a vector of indices into the input slice that should be included,
    /// also known as a selection vector.
    pub async fn filter_row_ids(&self, row_ids: &[u64]) -> Result<Vec<u64>> {
        // Group by the fragment id so we reduce mutex contention on the cache.
        let mut check_tasks = HashMap::new();
        for (i, row_id) in row_ids.iter().enumerate() {
            let fragment_id = (row_id >> 32) as u32;
            let task = check_tasks.entry(fragment_id).or_insert_with(Vec::new);
            task.push((i as u64, row_id));
        }

        let dataset = self.dataset.as_ref();

        let mut stream: BoxStream<_> = futures::stream::iter(check_tasks.drain())
            .map(|(fragment_id, ids)| async move {
                let Some(fragment) = dataset.get_fragment(fragment_id as usize) else {
                        return Ok(vec![]);
                    };
                let deletion_vector = match fragment.get_deletion_vector().await {
                    Ok(Some(deletion_vector)) => deletion_vector,
                    Ok(None) => return Ok(ids.iter().map(|(i, _)| *i).collect()),
                    Err(err) => return Err(err),
                };

                let mut selection = Vec::new();
                for (i, row_id) in ids {
                    let local_row_id = *row_id as u32;
                    if !deletion_vector.contains(local_row_id) {
                        selection.push(i);
                    }
                }

                Ok(selection)
            })
            .buffer_unordered(num_cpus::get())
            .boxed();

        let mut selection_vector = Vec::new();
        while let Some(res) = stream.next().await {
            let selection = res?;
            selection_vector.extend(selection);
        }

        selection_vector.sort();

        Ok(selection_vector)
    }
}
