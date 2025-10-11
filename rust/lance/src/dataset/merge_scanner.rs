// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merge Scanner for LSM-tree style merging of multiple datasets
//!
//! This module provides a scanner that can merge results from multiple Lance datasets
//! in a log-structured merge (LSM) tree fashion. When duplicate primary keys are encountered,
//! the value from the dataset with higher precedence (earlier in the list) is kept.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::physical_plan::projection::ProjectionExec as DFProjectionExec;
use datafusion::physical_plan::{limit::GlobalLimitExec, ExecutionPlan};
use futures::future::BoxFuture;
use lance_datafusion::exec::{execute_plan, LanceExecutionOptions, StrictBatchSizeExec};
use snafu::location;
use tracing::instrument;

use super::scanner::{LanceFilter, MaterializationStyle, Scanner, BATCH_SIZE_FALLBACK};
use super::Dataset;
use crate::dataset::scanner::{ColumnOrdering, DatasetRecordBatchStream};
use crate::io::exec::LsmMergeExec;
use crate::{Error, Result};

/// Merge Scanner for multiple datasets
///
/// This scanner merges results from multiple Lance datasets with the same schema
/// in LSM-tree style. The datasets are provided in precedence order - when duplicate
/// primary keys (determined by _rowid) are found, the value from the dataset with
/// higher precedence is kept.
///
/// # Example
///
/// ```ignore
/// use lance::dataset::MergeScanner;
///
/// let dataset1 = Dataset::open(uri1).await?;
/// let dataset2 = Dataset::open(uri2).await?;
/// let dataset3 = Dataset::open(uri3).await?;
///
/// // Create a merge scanner with dataset1 having highest precedence
/// let stream = MergeScanner::new(vec![
///     Arc::new(dataset1),
///     Arc::new(dataset2),
///     Arc::new(dataset3),
/// ])
/// .project(&["col1", "col2"])?
/// .filter("col1 > 10")?
/// .limit(Some(100), None)?
/// .try_into_stream()
/// .await?;
/// ```
#[derive(Clone)]
pub struct MergeScanner {
    /// Datasets to merge, ordered by precedence (highest first)
    datasets: Vec<Arc<Dataset>>,

    /// Projection columns
    projection: Option<Vec<String>>,

    /// Filter expression
    filter: Option<LanceFilter>,

    /// Batch size
    batch_size: Option<usize>,

    /// Limit and offset
    limit: Option<i64>,
    offset: Option<i64>,

    /// Ordering
    ordering: Option<Vec<ColumnOrdering>>,

    /// Materialization style
    materialization_style: MaterializationStyle,

    /// Whether to use scalar indices
    use_scalar_index: bool,

    /// Whether to use statistics
    use_stats: bool,

    /// Whether to scan in deterministic order
    ordered: bool,

    /// Whether to enforce strict batch size
    strict_batch_size: bool,
}

impl MergeScanner {
    /// Create a new MergeScanner from multiple datasets
    ///
    /// # Arguments
    ///
    /// * `datasets` - List of datasets to merge, ordered by precedence (highest first)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dataset list is empty
    /// - Datasets have incompatible schemas
    pub fn new(datasets: Vec<Arc<Dataset>>) -> Result<Self> {
        if datasets.is_empty() {
            return Err(Error::InvalidInput {
                source: "MergeScanner requires at least one dataset".into(),
                location: location!(),
            });
        }

        // Validate that all datasets have the same schema
        let base_schema = datasets[0].schema();
        for (idx, dataset) in datasets.iter().enumerate().skip(1) {
            let schema = dataset.schema();
            if base_schema != schema {
                return Err(Error::InvalidInput {
                    source: format!(
                        "Dataset at index {} has incompatible schema. Expected: {:?}, Got: {:?}",
                        idx, base_schema, schema
                    )
                    .into(),
                    location: location!(),
                });
            }
        }

        Ok(Self {
            datasets,
            projection: None,
            filter: None,
            batch_size: None,
            limit: None,
            offset: None,
            ordering: None,
            materialization_style: MaterializationStyle::Heuristic,
            use_scalar_index: true,
            use_stats: true,
            ordered: true,
            strict_batch_size: false,
        })
    }

    /// Set projection columns
    pub fn project<T: AsRef<str>>(&mut self, columns: &[T]) -> Result<&mut Self> {
        self.projection = Some(columns.iter().map(|c| c.as_ref().to_string()).collect());
        Ok(self)
    }

    /// Set filter expression
    pub fn filter(&mut self, filter: &str) -> Result<&mut Self> {
        self.filter = Some(LanceFilter::Sql(filter.to_string()));
        Ok(self)
    }

    /// Set batch size
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set limit and offset
    pub fn limit(&mut self, limit: Option<i64>, offset: Option<i64>) -> Result<&mut Self> {
        if limit.unwrap_or_default() < 0 {
            return Err(Error::invalid_input(
                "Limit must be non-negative".to_string(),
                location!(),
            ));
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(Error::invalid_input(
                    "Offset must be non-negative".to_string(),
                    location!(),
                ));
            }
        }
        self.limit = limit;
        self.offset = offset;
        Ok(self)
    }

    /// Set ordering
    pub fn order_by(&mut self, ordering: Option<Vec<ColumnOrdering>>) -> Result<&mut Self> {
        self.ordering = ordering;
        Ok(self)
    }

    /// Set materialization style
    pub fn materialization_style(&mut self, style: MaterializationStyle) -> &mut Self {
        self.materialization_style = style;
        self
    }

    /// Set whether to use scalar indices
    pub fn use_scalar_index(&mut self, use_scalar_index: bool) -> &mut Self {
        self.use_scalar_index = use_scalar_index;
        self
    }

    /// Set whether to use statistics
    pub fn use_stats(&mut self, use_stats: bool) -> &mut Self {
        self.use_stats = use_stats;
        self
    }

    /// Set whether to scan in deterministic order
    pub fn scan_in_order(&mut self, ordered: bool) -> &mut Self {
        self.ordered = ordered;
        self
    }

    /// Set whether to enforce strict batch size
    pub fn strict_batch_size(&mut self, strict_batch_size: bool) -> &mut Self {
        self.strict_batch_size = strict_batch_size;
        self
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size.unwrap_or_else(|| {
            std::cmp::max(
                self.datasets[0].object_store().block_size() / 4,
                BATCH_SIZE_FALLBACK,
            )
        })
    }

    /// Get the schema of the output
    pub async fn schema(&self) -> Result<SchemaRef> {
        let plan = self.create_plan().await?;
        Ok(plan.schema())
    }

    /// Create the execution plan
    #[instrument(level = "debug", skip_all)]
    pub async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let mut plans = Vec::with_capacity(self.datasets.len());

        // Create a scanner for each dataset
        for dataset in &self.datasets {
            let mut scanner = Scanner::new(dataset.clone());

            // Apply common configuration
            if let Some(ref proj) = self.projection {
                scanner.project(proj.as_slice())?;
            }

            if let Some(ref filter) = self.filter {
                match filter {
                    LanceFilter::Sql(sql) => {
                        scanner.filter(sql)?;
                    }
                    LanceFilter::Datafusion(expr) => {
                        scanner.filter_expr(expr.clone());
                    }
                    #[cfg(feature = "substrait")]
                    LanceFilter::Substrait(bytes) => {
                        scanner.filter_substrait(bytes)?;
                    }
                    #[cfg(not(feature = "substrait"))]
                    LanceFilter::Substrait(_) => {
                        return Err(Error::invalid_input(
                            "Substrait not supported in this build".to_string(),
                            location!(),
                        ));
                    }
                }
            }

            if let Some(batch_size) = self.batch_size {
                scanner.batch_size(batch_size);
            }

            if let Some(ref ordering) = self.ordering {
                scanner.order_by(Some(ordering.clone()))?;
            }

            scanner
                .materialization_style(self.materialization_style.clone())
                .use_scalar_index(self.use_scalar_index)
                .use_stats(self.use_stats)
                .scan_in_order(self.ordered);

            // Always include _rowid for merging
            scanner.with_row_id();

            // Create plan for this dataset
            let plan = scanner.create_plan().await?;
            plans.push(plan);
        }

        // Create the LSM merge execution plan
        let mut plan = Arc::new(LsmMergeExec::try_new(plans)?) as Arc<dyn ExecutionPlan>;

        // Apply limit/offset if specified
        if self.limit.unwrap_or(0) > 0 || self.offset.is_some() {
            plan = Arc::new(GlobalLimitExec::new(
                plan,
                *self.offset.as_ref().unwrap_or(&0) as usize,
                self.limit.map(|l| l as usize),
            ));
        }

        // Remove _rowid from output unless it was explicitly requested
        let should_remove_rowid = self
            .projection
            .as_ref()
            .map(|cols| !cols.iter().any(|c| c == "_rowid"))
            .unwrap_or(true); // If no projection, remove _rowid

        if should_remove_rowid {
            // Project to exclude _rowid
            let schema = plan.schema();
            let projection = schema
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| field.name() != "_rowid")
                .map(|(idx, field)| {
                    (
                        Arc::new(datafusion::physical_plan::expressions::Column::new(
                            field.name(),
                            idx,
                        )) as Arc<dyn datafusion::physical_expr::PhysicalExpr>,
                        field.name().clone(),
                    )
                })
                .collect::<Vec<_>>();

            plan = Arc::new(DFProjectionExec::try_new(projection, plan)?);
        }

        // Apply strict batch size if requested
        if self.strict_batch_size {
            plan = Arc::new(StrictBatchSizeExec::new(plan, self.get_batch_size()));
        }

        Ok(plan)
    }

    /// Execute the merge scan and return a stream of record batches
    #[instrument(skip_all)]
    pub fn try_into_stream(&self) -> BoxFuture<'_, Result<DatasetRecordBatchStream>> {
        Box::pin(async move {
            let plan = self.create_plan().await?;

            Ok(DatasetRecordBatchStream::new(execute_plan(
                plan,
                LanceExecutionOptions {
                    batch_size: self.batch_size,
                    ..Default::default()
                },
            )?))
        })
    }

    /// Collect all results into a single RecordBatch
    pub async fn try_into_batch(&self) -> Result<arrow_array::RecordBatch> {
        use arrow_select::concat::concat_batches;
        use futures::TryStreamExt;

        let stream = self.try_into_stream().await?;
        let schema = self.schema().await?;
        let batches = stream.try_collect::<Vec<_>>().await?;
        Ok(concat_batches(&schema, &batches)?)
    }
}
