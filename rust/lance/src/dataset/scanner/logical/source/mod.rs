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

pub mod v1;

use std::ops::Range;
use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::DataFusionError;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ExecutionPlan, expressions};
use lance_arrow::{DataTypeExt, SchemaExt as ArrowSchemaExt};
use lance_core::datatypes::{BlobHandling, Field as LanceField, OnMissing, Projection};
use lance_core::{
    ROW_ADDR_FIELD, ROW_CREATED_AT_VERSION_FIELD, ROW_ID_FIELD, ROW_LAST_UPDATED_AT_VERSION_FIELD,
};
use lance_file::reader::FileReaderOptions;
use lance_index::scalar::expression::PlannerIndexExt;
use lance_select::mask::{RowAddrMask, RowAddrTreeMap};
use lance_select::result::{IndexExprResult, IndexExprResultWireFormat};
use lance_table::format::Fragment;

use crate::dataset::scanner::MaterializationStyle;
use crate::dataset::{Dataset, Scanner};
use crate::index::{DatasetIndexInternalExt, ScalarIndexInfo};
use crate::io::exec::filtered_read::{
    FilteredReadExec, FilteredReadOptions, FilteredReadThreadingMode,
};
use crate::io::exec::scalar_index::ScalarIndexExec;
use crate::io::exec::{FilterPlan as ExprFilterPlan, Planner};
use crate::{Error, Result};

/// How a branch's scan is narrowed to the rows that branch is responsible for.
///
/// The two variants are the same statement at two granularities, which is the point: index
/// coverage has holes at fragment level (the index never saw those rows) and at row level (the
/// index saw the row before a data overlay changed it), and both holes are filled by a
/// brute-force branch reading exactly the rows in the hole.
#[derive(Debug, Clone)]
pub enum ScanRestriction {
    /// Read only these fragments.
    Fragments(Arc<Vec<Fragment>>),
    /// Read only these rows, identified by row id.
    Rows(Arc<RowAddrTreeMap>),
}

/// The subset of [`Scanner`](crate::dataset::Scanner) state that reaches the scan leaf.
///
/// Captured up front so the provider does not hold a `Scanner` reference: DataFusion owns the
/// provider for the life of the plan, well past the borrow the builder runs under.
#[derive(Clone)]
pub struct ScanSourceOptions {
    pub batch_size: Option<usize>,
    pub batch_readahead: usize,
    pub fragment_readahead: Option<usize>,
    pub io_buffer_size: Option<u64>,
    pub file_reader_options: Option<FileReaderOptions>,
    pub fragments: Option<Arc<Vec<Fragment>>>,
    pub index_expr_result_format: IndexExprResultWireFormat,
    pub use_scalar_index: bool,
    /// Which columns are cheap enough to read alongside the filter, rather than take after it.
    pub materialization_style: MaterializationStyle,
    /// Whether blob columns come back as their full binary value or as a description of where it
    /// lives. It changes both what the leaf reads and how wide the column is, so the
    /// materialization decision reads it too.
    pub blob_handling: BlobHandling,
    /// Answer from indices only: fragments a scalar index does not cover are skipped rather than
    /// scanned. Only meaningful alongside an index query.
    pub fast_search: bool,
    /// The exact vector index segments a search may use, from `Scanner::with_index_segments`.
    ///
    /// The leaf itself never reads this — it is here because the leaf is where stage 2 recovers
    /// the scan-wide settings that are not otherwise expressed in the plan.
    pub index_segments: Option<Arc<Vec<uuid::Uuid>>>,
    /// Emit deleted rows too, carrying a null row id.
    ///
    /// A branch of a coverage split keeps this, and should: the branches read disjoint fragment
    /// sets, so between them they still emit every deleted row exactly once.
    pub include_deleted_rows: bool,
    /// Read only these rows, from [`ScanRestriction::Rows`].
    ///
    /// The row set arrives through the same leaf slot a scalar-index result would, so a scan
    /// restricted this way cannot also consult a scalar index — which is exactly right for the
    /// case that produces one: these rows were singled out *because* their index entries are no
    /// longer trustworthy.
    pub rows: Option<Arc<RowAddrTreeMap>>,
    /// How this scan's predicate splits into a scalar index query and a refine filter.
    ///
    /// `None` means nothing has resolved it yet and the leaf will do so itself. Filling this in is
    /// what makes the index decision part of the plan rather than a private detail of
    /// [`TableProvider::scan`] — see [`ResolveScalarIndexQuery`](ResolveScalarIndexQuery).
    pub filter_plan: Option<ExprFilterPlan>,
    /// Rows the scalar index result must not emit, because a data overlay invalidated its entries
    /// for them. The same rows are re-read on a sibling branch, restricted by
    /// [`ScanRestriction::Rows`].
    pub overlay_block: Option<Arc<RowAddrTreeMap>>,
    /// The scanner this plan came from, captured only for legacy (v1) datasets.
    ///
    /// The legacy scan builder reads its options straight off `&Scanner` and is frozen, so the v1
    /// leaf hands it one rather than changing its signature. `None` on current storage, where
    /// nothing needs it. See [`v1`].
    ///
    /// Omitted from `Debug`: `Scanner` does not implement it, and the leaf's debug output is about
    /// what the leaf will read, not how the query got here.
    pub legacy_scanner: Option<Arc<Scanner>>,
}

impl std::fmt::Debug for ScanSourceOptions {
    /// `Scanner` has no `Debug`, and what the leaf will read is what matters here, so the captured
    /// legacy scanner is reported only as present or absent.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanSourceOptions")
            .field("batch_size", &self.batch_size)
            .field("batch_readahead", &self.batch_readahead)
            .field("fragment_readahead", &self.fragment_readahead)
            .field("io_buffer_size", &self.io_buffer_size)
            .field("file_reader_options", &self.file_reader_options)
            .field("fragments", &self.fragments)
            .field("index_expr_result_format", &self.index_expr_result_format)
            .field("use_scalar_index", &self.use_scalar_index)
            .field("materialization_style", &self.materialization_style)
            .field("blob_handling", &self.blob_handling)
            .field("fast_search", &self.fast_search)
            .field("index_segments", &self.index_segments)
            .field("include_deleted_rows", &self.include_deleted_rows)
            .field("rows", &self.rows)
            .field("filter_plan", &self.filter_plan)
            .field("overlay_block", &self.overlay_block)
            .field("legacy", &self.legacy_scanner.is_some())
            .finish()
    }
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
        // Blob handling changes a blob column's type — full binary value or a description of where
        // it lives — so it belongs to the schema the plan is built against, not just to the read.
        let base = ArrowSchema::from(
            &dataset
                .full_projection()
                .with_blob_handling(options.blob_handling.clone())
                .to_bare_schema(),
        );
        // System columns go in the order `Projection::to_schema` appends them, because that is the
        // order the read emits. A provider that advertises a different one hands DataFusion
        // projection indices that name the wrong column.
        let full_schema = base
            .try_with_column(ROW_ID_FIELD.clone())?
            .try_with_column(ROW_ADDR_FIELD.clone())?
            .try_with_column(ROW_LAST_UPDATED_AT_VERSION_FIELD.clone())?
            .try_with_column(ROW_CREATED_AT_VERSION_FIELD.clone())?;
        let created_version_idx = full_schema.fields.len() - 1;
        let updated_version_idx = created_version_idx - 1;
        let row_addr_idx = updated_version_idx - 1;
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

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// The scan-wide settings this leaf was built with.
    ///
    /// Stage 2 reads them from here rather than from the `Scanner`, which is what lets stages 2
    /// through 4 run against any plan whose leaf is a Lance scan.
    pub fn options(&self) -> &ScanSourceOptions {
        &self.options
    }

    /// The same source narrowed to part of the dataset.
    ///
    /// Used by [`SplitOnIndexCoverage`](SplitOnIndexCoverage) to give the indexed
    /// and brute-force branches disjoint row sets. The schema is unchanged, so a `TableScan`'s
    /// `projected_schema` stays valid across the swap.
    pub fn restricted_to(&self, restriction: &ScanRestriction) -> Self {
        let options = match restriction {
            ScanRestriction::Fragments(fragments) => ScanSourceOptions {
                fragments: Some(fragments.clone()),
                ..self.options.clone()
            },
            // The resolved filter plan goes with it: it was resolved *using* the index whose
            // entries for these rows are the reason this branch exists.
            ScanRestriction::Rows(rows) => ScanSourceOptions {
                rows: Some(rows.clone()),
                use_scalar_index: false,
                filter_plan: None,
                overlay_block: None,
                ..self.options.clone()
            },
        };
        self.with_options(options)
    }

    /// How this scan's predicate splits into a scalar index query and a refine filter, once a rule
    /// has resolved it.
    ///
    /// This is the accessor that makes the index decision inspectable from the plan. Without it the
    /// decision exists only inside [`TableProvider::scan`], where no rule can reach it.
    pub fn filter_plan(&self) -> Option<&ExprFilterPlan> {
        self.options.filter_plan.as_ref()
    }

    pub fn with_filter_plan(&self, filter_plan: ExprFilterPlan) -> Self {
        self.with_options(ScanSourceOptions {
            filter_plan: Some(filter_plan),
            ..self.options.clone()
        })
    }

    /// The rows withheld from this source's index result, if a split has already happened here.
    pub fn overlay_block(&self) -> Option<&Arc<RowAddrTreeMap>> {
        self.options.overlay_block.as_ref()
    }

    /// The same source, with `rows` withheld from whatever its index query returns.
    pub fn blocking(&self, rows: Arc<RowAddrTreeMap>) -> Self {
        self.with_options(ScanSourceOptions {
            overlay_block: Some(rows),
            ..self.options.clone()
        })
    }

    fn with_options(&self, options: ScanSourceOptions) -> Self {
        Self {
            dataset: self.dataset.clone(),
            options,
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
        let blob_handling = self.options.blob_handling.clone();
        let Some(projection) = projection else {
            return Ok(self
                .dataset
                .full_projection()
                .with_blob_handling(blob_handling));
        };
        let mut columns = Vec::with_capacity(projection.len());
        let mut result = self
            .dataset
            .empty_projection()
            .with_blob_handling(blob_handling);
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

    /// Whether `field` is cheap enough to read alongside the filter rather than take after it.
    ///
    /// Mirrors `Scanner::is_early_field`, including the reasoning behind the heuristic's byte-width
    /// thresholds, which is written out there.
    fn is_early_field(&self, field: &LanceField) -> bool {
        match &self.options.materialization_style {
            MaterializationStyle::AllEarly => true,
            MaterializationStyle::AllLate => false,
            MaterializationStyle::AllEarlyExcept(cols) => !cols.contains(&(field.id as u32)),
            MaterializationStyle::Heuristic => {
                if field.is_blob() && self.options.blob_handling.returns_description(field) {
                    // A description is an offset and a size, so it is cheap however wide the blob
                    // it points at is.
                    return true;
                }
                let byte_width = field.data_type().byte_width_opt();
                match self.dataset.object_store.as_ref().is_cloud() {
                    true => byte_width.is_some_and(|width| width < 1000),
                    false => byte_width.is_some_and(|width| width < 10),
                }
            }
        }
    }

    /// The columns to read alongside the filter, when taking the rest afterwards is cheaper than
    /// reading everything in one pass.
    ///
    /// `None` means read everything in one pass, which is the answer whenever the split would buy
    /// nothing: there is no refine filter for a take to skip rows against, or every requested
    /// column is cheap enough to read anyway. Mirrors `Scanner::calc_eager_projection` and the part
    /// of `Scanner::filtered_read_source` that decides whether to call it.
    fn eager_projection(
        &self,
        filter_plan: &ExprFilterPlan,
        requested: &Projection,
    ) -> Result<Option<Projection>> {
        if !filter_plan.has_refine() {
            return Ok(None);
        }
        // The one legacy branch that ignores the projection it is handed cannot be narrowed; see
        // `v1::uses_statistics_pushdown`. The imperative path hits the same wall from the other
        // side — it computes an eager projection for v1 too, and the take it plans afterwards
        // finds every column already there.
        if let Some(scanner) = self.options.legacy_scanner.as_ref()
            && v1::uses_statistics_pushdown(
                scanner,
                filter_plan,
                requested,
                self.options.fragments.as_ref(),
            )?
        {
            return Ok(None);
        }
        // A column the filter reads is loaded by the read whether or not it is projected, so
        // deferring it would cost a take and save nothing. This is why the emptiness check below
        // has to come after the union rather than before it: a wide column the filter also reads
        // would otherwise look deferrable and leave the take with nothing to fetch.
        let filter_schema = self
            .dataset
            .empty_projection()
            .union_columns(filter_plan.all_columns(), OnMissing::Error)?
            .into_schema();
        let eager = requested
            .clone()
            .subtract_predicate(|field| !self.is_early_field(field))
            .union_schema(&filter_schema);
        // Compare what the two reads would emit, not which field ids they carry. A blob column
        // read as a description emits one column no matter how many storage fields sit under it,
        // so an id-level difference can name columns that never reach the output and leave the
        // take with nothing to fetch. This is the same comparison the take itself makes.
        let eager_schema = ArrowSchema::from(&eager.to_schema());
        if !requested
            .clone()
            .subtract_arrow_schema(&eager_schema, OnMissing::Ignore)?
            .has_data_fields()
        {
            return Ok(None);
        }
        Ok(Some(eager.with_row_id()))
    }

    /// Fetch the columns the eager read left behind, for the rows that survived the filter.
    fn take_late(
        &self,
        input: Arc<dyn ExecutionPlan>,
        requested: &Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let taken = match v1::is_legacy(&self.dataset) {
            true => v1::take(
                &self.dataset,
                input,
                requested.clone(),
                self.options.batch_size.map(|size| size as u32),
            )?,
            false => {
                let mut options = FilteredReadOptions::new(requested.clone().with_row_id());
                if self.options.include_deleted_rows {
                    // Forwarded so the row-stream read rejects it: a deleted row carries a null
                    // row id, which the take would silently drop. `Scanner::take_current` does
                    // the same.
                    options = options.with_deleted_rows()?;
                }
                if let Some(batch_size) = self.options.batch_size {
                    options = options.with_batch_size(batch_size as u32);
                }
                if let Some(fragments) = &self.options.fragments {
                    options = options.with_fragments(fragments.clone());
                }
                Arc::new(FilteredReadExec::try_new(
                    self.dataset.clone(),
                    options,
                    Some(input),
                )?) as Arc<dyn ExecutionPlan>
            }
        };

        // The take appends its columns to the ones the eager read carried, so its output is in
        // neither the requested order nor the requested set. Restating both here is what keeps the
        // leaf's promise to return exactly the projection it was asked for.
        let schema = taken.schema();
        let columns = requested
            .to_schema()
            .fields
            .iter()
            .map(|field| {
                Ok((
                    expressions::col(&field.name, schema.as_ref())?,
                    field.name.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new(columns, taken)?))
    }

    /// Split the pushed-down predicates into a scalar-index query plus a refine expression.
    ///
    /// This mirrors [`Scanner::create_filter_plan`](crate::dataset::Scanner), including its
    /// guard that scalar indices are unusable when any fragment is missing a row count. It is
    /// synchronous because [`ScanPlanningContext`](super::context::ScanPlanningContext) already
    /// holds the `ScalarIndexInfo`, which is what lets a rule call it.
    pub fn resolve_filter_plan(
        &self,
        filters: &[Expr],
        index_info: &ScalarIndexInfo,
    ) -> Result<ExprFilterPlan> {
        let Some(expr) = conjunction(filters) else {
            return Ok(ExprFilterPlan::default());
        };

        let planner = Planner::new(self.full_schema.clone());
        let plan =
            planner.create_filter_plan(expr.clone(), index_info, self.options.use_scalar_index)?;

        if plan.index_query.is_some() && self.fragments().iter().any(|f| f.physical_rows.is_none())
        {
            // Scalar index results are expressed in row addresses, which need fragment row
            // counts to interpret. Without them, fall back to a pure refine filter.
            return planner.create_filter_plan(expr, index_info, false);
        }
        Ok(plan)
    }

    /// [`Self::resolve_filter_plan`] for the case where no rule got there first.
    async fn build_filter_plan(&self, filters: &[Expr]) -> Result<ExprFilterPlan> {
        let index_info = self.dataset.scalar_index_info().await?;
        self.resolve_filter_plan(filters, &index_info)
    }

    /// Present a row allow list in the shape the leaf's index-result slot expects.
    ///
    /// `Scanner::row_ids_as_take_input` wraps the same batch in a `OneShotExec`. A memory source
    /// says the same thing without the one-shot restriction, since the allow list is a materialized
    /// mask. That does not by itself make the scan re-executable — `FilteredReadExec` drains one
    /// shared task stream per instance, so every Lance scan plan is single-execution — but it keeps
    /// the reason for that in one place instead of two.
    fn rows_as_index_input(&self, rows: &RowAddrTreeMap) -> Result<Arc<dyn ExecutionPlan>> {
        let result = IndexExprResult::exact(RowAddrMask::from_allowed(rows.clone()));
        let batch = result.serialize(
            self.dataset.fragment_bitmap.as_ref(),
            self.options.index_expr_result_format,
        )?;
        let schema = batch.schema();
        Ok(MemorySourceConfig::try_new_exec(
            &[vec![batch]],
            schema,
            None,
        )?)
    }

    fn fragments(&self) -> &Arc<Vec<Fragment>> {
        self.options
            .fragments
            .as_ref()
            .unwrap_or_else(|| self.dataset.fragments())
    }

    /// How much of the fragment sequence a pushed-down limit lets the read stop after.
    ///
    /// `limit` is what DataFusion pushed into the scan, so it already carries any offset above it.
    /// Mirrors `Scanner::get_scan_range`, including both of its refusals: a range is a count of
    /// physical positions, so it cannot describe a limit that applies after a filter, and on a
    /// stable-row-id dataset those positions can be tombstones whose live replacements sit in later
    /// fragments — spending the range on them would skip rows the limit should have returned.
    fn scan_range(&self, limit: Option<usize>, filter_plan: &ExprFilterPlan) -> Option<Range<u64>> {
        let limit = limit?;
        if filter_plan.has_any_filter() || self.dataset.manifest.uses_stable_row_ids() {
            return None;
        }
        // Clamping keeps the range honest about the read it describes; an over-long one reads the
        // same rows but claims otherwise in every plan that prints it.
        let rows = self
            .fragments()
            .iter()
            .map(|fragment| fragment.num_rows())
            .sum::<Option<usize>>();
        Some(0..rows.map_or(limit, |rows| limit.min(rows)) as u64)
    }

    async fn scan_impl(
        &self,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let requested = self.to_lance_projection(projection)?;
        // A read of no fragments is a read of no rows. Saying so here rather than building one
        // keeps a branch a coverage split could not avoid producing — and a scan of a dataset that
        // has no data yet — from costing an operator. Mirrors the `EmptyExec` short-circuits in
        // `Scanner::plan_fts`.
        if self.fragments().is_empty() {
            let schema = ArrowSchema::from(&requested.to_schema());
            return Ok(Arc::new(EmptyExec::new(Arc::new(schema))));
        }
        let filter_plan = match self.options.filter_plan.clone() {
            Some(filter_plan) => filter_plan,
            None => self.build_filter_plan(filters).await?,
        };

        // Reading the wide columns for every row only to filter most of them away is the case late
        // materialization exists for; when it applies, the read below is narrower than what the
        // caller asked for and a take makes up the difference.
        let late = self.eager_projection(&filter_plan, &requested)?;
        let mut projection = late.clone().unwrap_or_else(|| requested.clone());

        // DataFusion asks for no columns when only the row count matters — `COUNT(*)`. Reading
        // nothing is not a thing the reader can do, so an identity column stands in and is
        // projected away again below. Matches `Scanner::filtered_read_source`, except on v1: the
        // legacy scan reads `with_row_addr` off the `Scanner` rather than the projection it is
        // handed, so the row id is the column that actually gets through there.
        let counting_rows = projection.is_empty();
        if counting_rows {
            match v1::is_legacy(&self.dataset) {
                true => projection.with_row_id = true,
                false => projection.with_row_addr = true,
            }
        }

        let scan_range = self.scan_range(limit, &filter_plan);

        if v1::is_legacy(&self.dataset) {
            let Some(scanner) = self.options.legacy_scanner.as_ref() else {
                return Err(Error::internal(
                    "a legacy (v1) scan reached the leaf without the scanner it needs".to_string(),
                ));
            };
            // Both of these come from a data overlay, and no writer can put one on a v1 dataset:
            // `lance_file::versions::create_writer` refuses `V1`. Reaching here means a split rule
            // produced a branch for a dataset that cannot have the coverage gap it is filling.
            if self.options.rows.is_some() || self.options.overlay_block.is_some() {
                return Err(Error::internal(
                    "a legacy (v1) scan was restricted by a data overlay, which v1 cannot have"
                        .to_string(),
                ));
            }
            let plan = v1::scan(
                scanner,
                &filter_plan,
                projection,
                self.options.include_deleted_rows,
                self.options.fragments.clone(),
                scan_range,
            )
            .await?;
            return match (counting_rows, &late) {
                (true, _) => drop_all_columns(plan),
                (false, Some(_)) => self.take_late(plan, &requested),
                (false, None) => Ok(plan),
            };
        }

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
        if self.options.include_deleted_rows {
            read_options = read_options.with_deleted_rows()?;
        }
        if let Some(scan_range) = scan_range {
            read_options = read_options.with_scan_range_before_filter(scan_range)?;
        }

        if let Some(rows) = &self.options.overlay_block {
            read_options =
                read_options.with_overlay_block(RowAddrMask::from_block(rows.as_ref().clone()));
        }

        // A row restriction and a scalar-index query compete for the same slot, and
        // `restricted_to` has already turned the index off in that case.
        let index_input = match &self.options.rows {
            Some(rows) => Some(self.rows_as_index_input(rows)?),
            None => filter_plan.index_query.map(|index_query| {
                Arc::new(ScalarIndexExec::new(
                    self.dataset.clone(),
                    index_query,
                    self.options.index_expr_result_format,
                )) as Arc<dyn ExecutionPlan>
            }),
        };

        let plan = Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            read_options,
            index_input,
        )?) as Arc<dyn ExecutionPlan>;
        match (counting_rows, &late) {
            (true, _) => drop_all_columns(plan),
            (false, Some(_)) => self.take_late(plan, &requested),
            (false, None) => Ok(plan),
        }
    }
}

/// A plan with the same rows and no columns.
fn drop_all_columns(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let no_columns: Vec<(Arc<dyn datafusion::physical_expr::PhysicalExpr>, String)> = vec![];
    Ok(Arc::new(ProjectionExec::try_new(no_columns, plan)?))
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

    /// The leaf applies the predicate itself, so DataFusion never needs to re-check it.
    ///
    /// True on both storage versions, though for different reasons: `FilteredReadExec` always
    /// applies it, while on v1 [`v1::scan`] tops the legacy plan with a filter in the one case
    /// where the legacy builder would not have applied it.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
    }
}

fn conjunction(filters: &[Expr]) -> Option<Expr> {
    filters
        .iter()
        .cloned()
        .reduce(datafusion::logical_expr::Expr::and)
}
