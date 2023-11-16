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
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_schema::{ArrowError, DataType, Schema as ArrowSchema};
use arrow_select::concat::concat;
use datafusion::common::DFSchema;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::ExprSchemable;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{ColumnarValue, PhysicalExpr};
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::error::{box_error, InvalidInputSnafu};
use lance_core::format::Fragment;
use lance_index::util::datafusion::safe_coerce_scalar;
use roaring::RoaringTreemap;
use snafu::{location, Location, ResultExt};

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
///     .build?
///     .execute()
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
        let expr = planner
            .parse_filter(filter)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;
        self.condition = Some(
            planner
                .optimize_expr(expr)
                .map_err(box_error)
                .context(InvalidInputSnafu)?,
        );
        Ok(self)
    }

    pub fn set(mut self, column: impl AsRef<str>, value: &str) -> Result<Self> {
        let field = self
            .dataset
            .schema()
            .field(column.as_ref())
            .ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "Column '{}' does not exist in dataset schema: {:?}",
                        column.as_ref(),
                        self.dataset.schema()
                    ),
                    location!(),
                )
            })?;

        // TODO: support nested column references. This is mostly blocked on the
        // ability to insert them into the RecordBatch properly.
        if column.as_ref().contains('.') {
            return Err(Error::NotSupported {
                source: format!(
                    "Nested column references are not yet supported. Referenced: {}",
                    column.as_ref(),
                )
                .into(),
                location: location!(),
            });
        }

        let schema: Arc<ArrowSchema> = Arc::new(self.dataset.schema().into());
        let planner = Planner::new(schema.clone());
        let mut expr = planner
            .parse_expr(value)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;

        // Cast expression to the column's data type if necessary.
        let dest_type = field.data_type();
        let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
        let src_type = expr
            .get_type(&df_schema)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;
        if dest_type != src_type {
            // TODO: remove this once DataFusion supports casting List to FSL
            expr = match expr {
                Expr::Literal(value @ ScalarValue::List(_, _))
                    if matches!(dest_type, DataType::FixedSizeList(_, _)) =>
                {
                    Expr::Literal(safe_coerce_scalar(&value, &dest_type).ok_or_else(|| {
                        ArrowError::CastError(format!(
                            "Failed to cast {} to {} during planning",
                            value.data_type(),
                            dest_type
                        ))
                    })?)
                }
                _ => expr
                    .cast_to(&dest_type, &df_schema)
                    .map_err(box_error)
                    .context(InvalidInputSnafu)?,
            };
        }

        // Optimize the expression. For example, this might apply the cast on
        // literals.
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;

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

        if updates.is_empty() {
            return Err(Error::invalid_input("No updates provided", location!()));
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
        let (old_fragments, removed_fragment_ids) = self.apply_deletions(&removed_row_ids).await?;

        // Commit updated and new fragments
        self.commit(removed_fragment_ids, old_fragments, new_fragments)
            .await
    }

    fn apply_updates(
        batch: RecordBatch,
        updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    ) -> DFResult<RecordBatch> {
        let mut batch = batch.clone();
        for (column, expr) in updates.iter() {
            let new_values = expr.evaluate(&batch)?;
            // TODO: once datafusion implements ColumnarValue::into_array() for
            // FixedSizeList, we can remove this branch.
            let new_values = if matches!(new_values.data_type(), DataType::FixedSizeList(_, _)) {
                match new_values {
                    ColumnarValue::Array(array) => array,
                    ColumnarValue::Scalar(scalar) => {
                        if let ScalarValue::Fixedsizelist(values, field, size) = scalar {
                            if let Some(values) = values {
                                // Convert each scalar to an array, then concat to get a single list.
                                let values: Vec<ArrayRef> =
                                    values.iter().map(|v| v.to_array()).collect();
                                let value_refs: Vec<&dyn Array> =
                                    values.iter().map(|v| v.as_ref()).collect();
                                let values = concat(&value_refs).unwrap();

                                // Repeat the list num_rows times, and concat again.
                                let value_refs = std::iter::repeat(values.as_ref())
                                    .take(batch.num_rows())
                                    .collect::<Vec<_>>();
                                let values = concat(&value_refs).unwrap();

                                Arc::new(FixedSizeListArray::new(field, size, values, None))
                            } else {
                                return Err(DataFusionError::NotImplemented(
                                    "FixedSizeList null literals".into(),
                                ));
                            }
                        } else {
                            unreachable!()
                        }
                    }
                }
            } else {
                new_values.into_array(batch.num_rows())
            };
            batch = batch.replace_column_by_name(column.as_str(), new_values)?;
        }
        Ok(batch)
    }

    /// Use previous found rows ids to delete rows from existing fragments.
    ///
    /// Returns the set of modified fragments and removed fragments, if any.
    async fn apply_deletions(
        &self,
        removed_row_ids: &RoaringTreemap,
    ) -> Result<(Vec<Fragment>, Vec<u64>)> {
        let bitmaps = Arc::new(removed_row_ids.bitmaps().collect::<BTreeMap<_, _>>());

        enum FragmentChange {
            Unchanged,
            Modified(Fragment),
            Removed(u64),
        }

        let mut updated_fragments = Vec::new();
        let mut removed_fragments = Vec::new();

        let mut stream = futures::stream::iter(self.dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    let fragment_id = fragment.id();
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                        match fragment.apply_deletions(*bitmap).await {
                            Ok(Some(new_fragment)) => {
                                Ok(FragmentChange::Modified(new_fragment.metadata))
                            }
                            Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(FragmentChange::Unchanged)
                    }
                }
            })
            .buffer_unordered(num_cpus::get() * 4);

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }

    async fn commit(
        &self,
        removed_fragment_ids: Vec<u64>,
        updated_fragments: Vec<Fragment>,
        new_fragments: Vec<Fragment>,
    ) -> Result<Arc<Dataset>> {
        let operation = Operation::Update {
            removed_fragment_ids,
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
    use crate::dataset::WriteParams;

    use super::*;

    use arrow_array::{Int64Array, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use arrow_select::concat::concat_batches;
    use futures::TryStreamExt;
    use tempfile::{tempdir, TempDir};

    /// Returns a dataset with 3 fragments, each with 10 rows.
    ///
    /// Also returns the TempDir, which should be kept alive as long as the
    /// dataset is being accessed. Once that is dropped, the temp directory is
    /// deleted.
    async fn make_test_dataset() -> (Arc<Dataset>, TempDir) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    std::iter::repeat("foo").take(30),
                )),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 10,
            ..Default::default()
        };

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());
        let ds = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        (Arc::new(ds), test_dir)
    }

    #[tokio::test]
    async fn test_update_validation() {
        let (dataset, _test_dir) = make_test_dataset().await;

        let builder = UpdateBuilder::new(dataset.clone());

        assert!(
            matches!(
                builder.clone().update_where("foo = 10"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if condition references non-existent column"
        );

        assert!(
            matches!(
                builder.clone().set("foo", "1"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if update key references non-existent column"
        );

        assert!(
            matches!(
                builder.clone().set("id", "id2 + 1"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if update expression references non-existent column"
        );

        assert!(
            matches!(builder.clone().build(), Err(Error::InvalidInput { .. })),
            "Should return error if no update expressions are provided"
        );
    }

    #[tokio::test]
    async fn test_update_all() {
        let (dataset, _test_dir) = make_test_dataset().await;

        let dataset = UpdateBuilder::new(dataset)
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        let expected = RecordBatch::try_new(
            Arc::new(dataset.schema().into()),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    (0..30).map(|i| format!("bar{}", i)),
                )),
            ],
        )
        .unwrap();

        assert_eq!(actual_batch, expected);

        assert_eq!(dataset.get_fragments().len(), 1);
    }

    #[tokio::test]
    async fn test_update_conditional() {
        let (dataset, _test_dir) = make_test_dataset().await;

        let original_fragments = dataset.get_fragments();

        let dataset = UpdateBuilder::new(dataset)
            .update_where("id >= 15")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        let expected = RecordBatch::try_new(
            Arc::new(dataset.schema().into()),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    (0..15)
                        .map(|_| "foo".to_string())
                        .chain((15..30).map(|i| format!("bar{}", i))),
                )),
            ],
        )
        .unwrap();

        assert_eq!(actual_batch, expected);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 3);

        // One fragment not touched (id = 0..10)
        assert_eq!(fragments[0].metadata, original_fragments[0].metadata,);
        // One fragment partially modified (id = 10..15)
        assert_eq!(
            fragments[1].metadata.files,
            original_fragments[1].metadata.files,
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .and_then(|f| f.num_deleted_rows),
            Some(5)
        );
        // One fragment fully modified
        assert_eq!(fragments[2].metadata.physical_rows, Some(15));
    }
}
