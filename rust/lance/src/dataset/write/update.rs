// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

use super::super::utils::make_rowid_capture_stream;
use super::{write_fragments_internal, WriteParams};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, Schema as ArrowSchema};
use datafusion::common::DFSchema;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::ExprSchemable;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::error::{box_error, InvalidInputSnafu};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::expr::safe_coerce_scalar;
use lance_table::format::Fragment;
use roaring::RoaringTreemap;
use snafu::{location, Location, ResultExt};

use crate::dataset::transaction::{Operation, Transaction};
use crate::io::commit::commit_transaction;
use crate::{io::exec::Planner, Dataset};
use crate::{Error, Result};

/// Build an update operation.
///
/// This operation is similar to SQL's UPDATE statement. It allows you to change
/// the values of all or a subset of columns with SQL expresions.
///
/// Use the [UpdateBuilder] to construct an update job. For example:
///
/// ```ignore
/// let dataset = UpdateBuilder::new(dataset.clone())
///     .update_where("region_id = 10")
///     .set("region_name", "New York")
///     .build()?
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
            expr = match expr {
                // TODO: remove this branch once DataFusion supports casting List to FSL
                // This should happen in Arrow 51.0.0
                Expr::Literal(value @ ScalarValue::List(_))
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
        // literals. (Expr.cast_to() only wraps the expression in a Cast node,
        // it doesn't actually apply the cast to the literals.)
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu)?;

        self.updates.insert(column.as_ref().to_string(), expr);
        Ok(self)
    }

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
pub struct UpdateResult {
    pub new_dataset: Arc<Dataset>,
    pub rows_updated: u64,
}

#[derive(Debug, Clone)]
pub struct UpdateJob {
    dataset: Arc<Dataset>,
    condition: Option<Expr>,
    updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
}

impl UpdateJob {
    pub async fn execute(self) -> Result<UpdateResult> {
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
        let stream = stream
            .map(move |batch| {
                let updates = updates_ref.clone();
                tokio::task::spawn_blocking(move || Self::apply_updates(batch?, updates))
            })
            .buffered(get_num_compute_intensive_cpus())
            .map(|res| match res {
                Ok(Ok(batch)) => Ok(batch),
                Ok(Err(err)) => Err(err),
                Err(e) => Err(DataFusionError::Execution(e.to_string())),
            });
        let stream = RecordBatchStreamAdapter::new(schema, stream);

        let version = self
            .dataset
            .manifest()
            .data_storage_format
            .lance_file_version()?;
        let new_fragments = write_fragments_internal(
            Some(&self.dataset),
            self.dataset.object_store.clone(),
            &self.dataset.base,
            self.dataset.schema(),
            Box::pin(stream),
            WriteParams::with_storage_version(version),
        )
        .await?;
        // Apply deletions
        let removed_row_ids = Arc::into_inner(removed_row_ids)
            .unwrap()
            .into_inner()
            .unwrap();
        let (old_fragments, removed_fragment_ids) = self.apply_deletions(&removed_row_ids).await?;

        let num_updated_rows = new_fragments
            .iter()
            .map(|f| f.physical_rows.unwrap() as u64)
            .sum::<u64>();
        // Commit updated and new fragments
        let new_dataset = self
            .commit(removed_fragment_ids, old_fragments, new_fragments)
            .await?;
        Ok(UpdateResult {
            new_dataset,
            rows_updated: num_updated_rows,
        })
    }

    fn apply_updates(
        batch: RecordBatch,
        updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    ) -> DFResult<RecordBatch> {
        let mut batch = batch.clone();
        for (column, expr) in updates.iter() {
            let new_values = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
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
                        match fragment.extend_deletions(*bitmap).await {
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
            .buffer_unordered(self.dataset.object_store.io_parallelism());

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
            self.dataset.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.dataset.manifest_naming_scheme,
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
    use arrow_schema::{Field, Schema as ArrowSchema};
    use arrow_select::concat::concat_batches;
    use futures::TryStreamExt;
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;
    use tempfile::{tempdir, TempDir};

    /// Returns a dataset with 3 fragments, each with 10 rows.
    ///
    /// Also returns the TempDir, which should be kept alive as long as the
    /// dataset is being accessed. Once that is dropped, the temp directory is
    /// deleted.
    async fn make_test_dataset(version: LanceFileVersion) -> (Arc<Dataset>, TempDir) {
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
            data_storage_version: Some(version),
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
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::Legacy).await;

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

    #[rstest]
    #[tokio::test]
    async fn test_update_all(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    ) {
        let (dataset, _test_dir) = make_test_dataset(version).await;

        let update_result = UpdateBuilder::new(dataset)
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
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

    #[rstest]
    #[tokio::test]
    async fn test_update_conditional(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    ) {
        let (dataset, _test_dir) = make_test_dataset(version).await;

        let original_fragments = dataset.get_fragments();

        let update_result = UpdateBuilder::new(dataset)
            .update_where("id >= 15")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
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
