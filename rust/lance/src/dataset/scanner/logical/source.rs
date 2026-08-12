// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The scan leaf for the logical read path.
//!
//! [`LanceScanSource`] is the [`TableProvider`] that a `TableScan` in a Lance logical plan is
//! backed by. It exists instead of reusing [`LanceTableProvider`] because that provider's
//! `scan()` recurses into `Dataset::scan().create_plan()` — the imperative builder this path is
//! meant to replace. This one lowers straight to a [`FilteredReadExec`].
//!
//! [`LanceTableProvider`]: crate::datafusion::LanceTableProvider

use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::DataFusionError;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use lance_arrow::SchemaExt as ArrowSchemaExt;
use lance_core::datatypes::{OnMissing, Projection};
use lance_core::{
    ROW_ADDR_FIELD, ROW_CREATED_AT_VERSION_FIELD, ROW_ID_FIELD, ROW_LAST_UPDATED_AT_VERSION_FIELD,
};
use lance_file::reader::FileReaderOptions;
use lance_index::scalar::expression::PlannerIndexExt;
use lance_select::result::IndexExprResultWireFormat;
use lance_table::format::Fragment;

use crate::Result;
use crate::dataset::Dataset;
use crate::index::DatasetIndexInternalExt;
use crate::io::exec::filtered_read::{
    FilteredReadExec, FilteredReadOptions, FilteredReadThreadingMode,
};
use crate::io::exec::scalar_index::ScalarIndexExec;
use crate::io::exec::{FilterPlan as ExprFilterPlan, Planner};

/// The subset of [`Scanner`](crate::dataset::Scanner) state that reaches the scan leaf.
///
/// Captured up front so the provider does not hold a `Scanner` reference: DataFusion owns the
/// provider for the life of the plan, well past the borrow the builder runs under.
#[derive(Debug, Clone)]
pub struct ScanSourceOptions {
    pub batch_size: Option<usize>,
    pub batch_readahead: usize,
    pub fragment_readahead: Option<usize>,
    pub io_buffer_size: Option<u64>,
    pub file_reader_options: Option<FileReaderOptions>,
    pub fragments: Option<Arc<Vec<Fragment>>>,
    pub index_expr_result_format: IndexExprResultWireFormat,
    pub use_scalar_index: bool,
    /// Answer from indices only: fragments a scalar index does not cover are skipped rather than
    /// scanned. Only meaningful alongside an index query.
    pub fast_search: bool,
}

pub struct LanceScanSource {
    dataset: Arc<Dataset>,
    options: ScanSourceOptions,
    /// Dataset schema plus the system columns the leaf can produce. DataFusion's projection
    /// indices index into this, so it must stay stable for the life of the provider.
    full_schema: SchemaRef,
    row_id_idx: usize,
    row_addr_idx: usize,
    created_version_idx: usize,
    updated_version_idx: usize,
}

impl std::fmt::Debug for LanceScanSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceScanSource")
            .field("uri", &self.dataset.base)
            .field("options", &self.options)
            .finish()
    }
}

impl LanceScanSource {
    pub fn new(dataset: Arc<Dataset>, options: ScanSourceOptions) -> Result<Self> {
        let base = ArrowSchema::from(dataset.schema());
        let full_schema = base
            .try_with_column(ROW_ID_FIELD.clone())?
            .try_with_column(ROW_ADDR_FIELD.clone())?
            .try_with_column(ROW_CREATED_AT_VERSION_FIELD.clone())?
            .try_with_column(ROW_LAST_UPDATED_AT_VERSION_FIELD.clone())?;
        let updated_version_idx = full_schema.fields.len() - 1;
        let created_version_idx = updated_version_idx - 1;
        let row_addr_idx = created_version_idx - 1;
        let row_id_idx = row_addr_idx - 1;
        Ok(Self {
            dataset,
            options,
            full_schema: Arc::new(full_schema),
            row_id_idx,
            row_addr_idx,
            created_version_idx,
            updated_version_idx,
        })
    }

    /// The same source restricted to a subset of fragments.
    ///
    /// Used by [`SplitPartiallyIndexedSearch`](super::rules::SplitPartiallyIndexedSearch) to give
    /// the indexed and unindexed branches disjoint halves of the dataset. The schema is
    /// unchanged, so a `TableScan`'s `projected_schema` stays valid across the swap.
    pub fn restricted_to(&self, fragments: Vec<Fragment>) -> Self {
        Self {
            dataset: self.dataset.clone(),
            options: ScanSourceOptions {
                fragments: Some(Arc::new(fragments)),
                ..self.options.clone()
            },
            full_schema: self.full_schema.clone(),
            row_id_idx: self.row_id_idx,
            row_addr_idx: self.row_addr_idx,
            created_version_idx: self.created_version_idx,
            updated_version_idx: self.updated_version_idx,
        }
    }

    /// Translate DataFusion's positional projection into a Lance [`Projection`].
    ///
    /// `None` means "every column", matching DataFusion's convention.
    fn to_lance_projection(&self, projection: Option<&Vec<usize>>) -> Result<Projection> {
        let Some(projection) = projection else {
            return Ok(self.dataset.full_projection());
        };
        let mut columns = Vec::with_capacity(projection.len());
        let mut result = self.dataset.empty_projection();
        for idx in projection {
            if *idx == self.row_id_idx {
                result = result.with_row_id();
            } else if *idx == self.row_addr_idx {
                result = result.with_row_addr();
            } else if *idx == self.created_version_idx {
                result.with_row_created_at_version = true;
            } else if *idx == self.updated_version_idx {
                result.with_row_last_updated_at_version = true;
            } else {
                columns.push(self.full_schema.field(*idx).name());
            }
        }
        result.union_columns(columns, OnMissing::Error)
    }

    /// Split the pushed-down predicates into a scalar-index query plus a refine expression.
    ///
    /// This mirrors [`Scanner::create_filter_plan`](crate::dataset::Scanner), including its
    /// guard that scalar indices are unusable when any fragment is missing a row count.
    async fn build_filter_plan(&self, filters: &[Expr]) -> Result<ExprFilterPlan> {
        let Some(expr) = conjunction(filters) else {
            return Ok(ExprFilterPlan::default());
        };

        let planner = Planner::new(self.full_schema.clone());
        let index_info = self.dataset.scalar_index_info().await?;
        let plan =
            planner.create_filter_plan(expr.clone(), &index_info, self.options.use_scalar_index)?;

        if plan.index_query.is_some() && self.fragments().iter().any(|f| f.physical_rows.is_none())
        {
            // Scalar index results are expressed in row addresses, which need fragment row
            // counts to interpret. Without them, fall back to a pure refine filter.
            return planner.create_filter_plan(expr, &index_info, false);
        }
        Ok(plan)
    }

    fn fragments(&self) -> &Arc<Vec<Fragment>> {
        self.options
            .fragments
            .as_ref()
            .unwrap_or_else(|| self.dataset.fragments())
    }

    async fn scan_impl(
        &self,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut projection = self.to_lance_projection(projection)?;
        if projection.is_empty() {
            // Reading nothing is not a thing the reader can do; the row address is the cheapest
            // stand-in. Matches `Scanner::filtered_read_source`.
            projection.with_row_addr = true;
        }
        let filter_plan = self.build_filter_plan(filters).await?;

        let mut read_options = FilteredReadOptions::basic_full_read(&self.dataset)
            .with_filter_plan(filter_plan.clone())
            .with_projection(projection)
            .with_threading_mode(FilteredReadThreadingMode::OnePartitionMultipleThreads(
                self.options.batch_readahead,
            ));

        if let Some(fragments) = self.options.fragments.clone() {
            read_options = read_options.with_fragments(fragments);
        }
        if let Some(batch_size) = self.options.batch_size {
            read_options = read_options.with_batch_size(batch_size as u32);
        }
        if let Some(file_reader_options) = self.options.file_reader_options.clone() {
            read_options = read_options.with_file_reader_options(file_reader_options);
        }
        if let Some(fragment_readahead) = self.options.fragment_readahead {
            read_options = read_options.with_fragment_readahead(fragment_readahead);
        }
        if let Some(io_buffer_size) = self.options.io_buffer_size {
            read_options = read_options.with_io_buffer_size(io_buffer_size);
        }
        if self.options.fast_search && filter_plan.has_index_query() {
            read_options = read_options.with_only_indexed_fragments();
        }
        if let Some(limit) = limit
            && !filter_plan.has_any_filter()
        {
            read_options = read_options.with_scan_range_before_filter(0..limit as u64)?;
        }

        let index_input = filter_plan.index_query.map(|index_query| {
            Arc::new(ScalarIndexExec::new(
                self.dataset.clone(),
                index_query,
                self.options.index_expr_result_format,
            )) as Arc<dyn ExecutionPlan>
        });

        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            read_options,
            index_input,
        )?))
    }
}

#[async_trait]
impl TableProvider for LanceScanSource {
    fn schema(&self) -> SchemaRef {
        self.full_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        self.scan_impl(projection, filters, limit)
            .await
            .map_err(DataFusionError::from)
    }

    /// `FilteredReadExec` applies the predicate itself, so DataFusion never needs to re-check it.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Exact)
            .collect())
    }
}

fn conjunction(filters: &[Expr]) -> Option<Expr> {
    filters
        .iter()
        .cloned()
        .reduce(datafusion::logical_expr::Expr::and)
}
