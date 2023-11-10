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

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

use super::super::utils::make_rowid_capture_stream;
use super::write_fragments_internal;
use arrow_array::RecordBatch;
use datafusion::error::Result as DFResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::prelude::Expr;
use futures::{StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::format::Fragment;
use roaring::RoaringTreemap;
use snafu::{location, Location};

use crate::dataset::transaction::{Operation, Transaction};
use crate::io::commit::commit_transaction;
use crate::{io::exec::Planner, Dataset};
use crate::{Error, Result};

/// Update a dataset
///
/// ```ignore
/// let dataset = UpdateBuilder::new()
///     .update_where("region_id = 10")
///     .set("region_name", "New York")
///     .await?;
/// ```
///
#[derive(Debug, Clone)]
pub struct UpdateBuilder {
    /// The dataset snapshot to update.
    dataset: Arc<Dataset>,
    /// The condition to apply to find matching rows to update. If None, all rows are updated.
    condition: Option<Expr>,
    /// The updates to apply to matching rows.
    updates: HashMap<String, Expr>,
}

impl UpdateBuilder {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self {
            dataset,
            condition: None,
            updates: HashMap::new(),
        }
    }

    pub fn update_where(mut self, filter: &str) -> Result<Self> {
        let planner = Planner::new(Arc::new(self.dataset.schema().into()));
        let expr = planner.parse_filter(filter)?;
        self.condition = Some(planner.optimize_expr(expr)?);
        Ok(self)
    }

    pub fn set(mut self, column: impl AsRef<str>, value: &str) -> Result<Self> {
        if self.dataset.schema().field(column.as_ref()).is_none() {
            return Err(Error::invalid_input(
                format!(
                    "Column '{}' does not exist in dataset schema: {:?}",
                    column.as_ref(),
                    self.dataset.schema()
                ),
                location!(),
            ));
        }

        let planner = Planner::new(Arc::new(self.dataset.schema().into()));
        let expr = planner.parse_expr(value)?;
        let expr = planner.optimize_expr(expr)?;
        self.updates.insert(column.as_ref().to_string(), expr);
        Ok(self)
    }

    // TODO: should we support Expr for value instead of &str?
    // pub fn set_expr(mut self, column: &str, value: Expr) -> Result<Self> { ... }

    // TODO: set write params
    // pub fn with_write_params(mut self, params: WriteParams) -> Self { ... }

    pub fn build(self) -> Result<UpdateJob> {
        let mut updates = HashMap::new();

        let planner = Planner::new(Arc::new(self.dataset.schema().into()));

        for (column, expr) in self.updates {
            let physical_expr = planner.create_physical_expr(&expr)?;
            updates.insert(column, physical_expr);
        }

        let updates = Arc::new(updates);

        Ok(UpdateJob {
            dataset: self.dataset,
            condition: self.condition,
            updates,
        })
    }
}

// TODO: support distributed operation.

#[derive(Debug, Clone)]
pub struct UpdateJob {
    dataset: Arc<Dataset>,
    condition: Option<Expr>,
    updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
}

impl UpdateJob {
    pub async fn execute(self) -> Result<Arc<Dataset>> {
        let mut scanner = self.dataset.scan();
        scanner.with_row_id();

        if let Some(expr) = &self.condition {
            scanner.filter_expr(expr.clone());
        }

        let stream = scanner.try_into_stream().await?.into();

        // We keep track of seen row ids so we can delete them from the existing
        // fragments.
        let removed_row_ids = Arc::new(RwLock::new(RoaringTreemap::new()));
        let stream = make_rowid_capture_stream(removed_row_ids.clone(), stream)?;

        let schema = stream.schema();

        let expected_schema = self.dataset.schema().into();
        if schema.as_ref() != &expected_schema {
            return Err(Error::Internal {
                message: format!("Expected schema {:?} but got {:?}", expected_schema, schema),
                location: location!(),
            });
        }

        let updates_ref = self.updates.clone();
        // TODO: consider CPU parallelism for this map in the stream.
        let stream = stream.map(move |batch| Self::apply_updates(batch?, updates_ref.clone()));
        let stream = RecordBatchStreamAdapter::new(schema, stream);

        let new_fragments = write_fragments_internal(
            self.dataset.object_store.clone(),
            &self.dataset.base,
            self.dataset.schema(),
            Box::pin(stream),
            Default::default(),
        )
        .await?;

        // Apply deletions
        let removed_row_ids = Arc::into_inner(removed_row_ids)
            .unwrap()
            .into_inner()
            .unwrap();
        let old_fragments = self.apply_deletions(&removed_row_ids).await?;

        // Commit updated and new fragments
        self.commit(old_fragments, new_fragments).await
    }

    fn apply_updates(
        batch: RecordBatch,
        updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    ) -> DFResult<RecordBatch> {
        let mut batch = batch.clone();
        for (column, expr) in updates.iter() {
            let new_values = expr.evaluate(&batch)?.into_array(batch.num_rows());
            batch = batch.replace_column_by_name(column.as_str(), new_values)?;
        }
        Ok(batch)
    }

    /// Use previous found rows ids to delete rows from existing fragments.
    ///
    /// Returns the set of modified fragments, if any.
    async fn apply_deletions(&self, removed_row_ids: &RoaringTreemap) -> Result<Vec<Fragment>> {
        let bitmaps = Arc::new(removed_row_ids.bitmaps().collect::<BTreeMap<_, _>>());

        futures::stream::iter(self.dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment.id() as u32)) {
                        fragment.apply_deletions(*bitmap).await.map(Some)
                    } else {
                        Ok(None)
                    }
                }
            })
            .buffer_unordered(num_cpus::get() * 4)
            .try_filter_map(|f| futures::future::ready(Ok(f)))
            .try_collect::<Vec<Fragment>>()
            .await
    }

    async fn commit(
        &self,
        updated_fragments: Vec<Fragment>,
        new_fragments: Vec<Fragment>,
    ) -> Result<Arc<Dataset>> {
        let columns_updated = self.updates.keys().cloned().collect::<Vec<_>>();
        let operation = Operation::Update {
            columns_updated,
            updated_fragments,
            new_fragments,
        };
        let transaction = Transaction::new(self.dataset.manifest.version, operation, None);

        let manifest = commit_transaction(
            self.dataset.as_ref(),
            self.dataset.object_store(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        let mut dataset = self.dataset.as_ref().clone();
        dataset.manifest = Arc::new(manifest);

        Ok(Arc::new(dataset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_test_dataset() {
        todo!()
    }

    #[tokio::test]
    async fn test_update_validation() {
        todo!()

        // 1. Validates condition columns exist
        // 2. Validates update key columns exist
        // 3. Validate update expressions reference valid columns
        // 4. Validates there is at least one update expression
    }

    #[tokio::test]
    async fn test_update_all() {
        todo!()
    }

    #[tokio::test]
    async fn test_update_conditional() {
        todo!()
    }
}
