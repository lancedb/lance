// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemTableScanner builder for creating query execution plans.

use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field, SchemaRef};
use datafusion::common::{ScalarValue, ToDFSchema};
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::{Expr, SessionContext};
use futures::TryStreamExt;
use lance_core::{Error, Result, ROW_ID};
use lance_datafusion::planner::Planner;
use lance_linalg::distance::DistanceType;
use snafu::location;

use super::exec::{BTreeIndexExec, FtsIndexExec, MemTableScanExec, VectorIndexExec};
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Vector search query parameters.
#[derive(Debug, Clone)]
pub struct VectorQuery {
    /// Column name containing vectors.
    pub column: String,
    /// Query vector.
    pub query_vector: Arc<dyn Array>,
    /// Number of results to return.
    pub k: usize,
    /// The minimum number of probes to search. More partitions may be searched
    /// if needed to satisfy k results or recall requirements. Defaults to 1.
    pub nprobes: usize,
    /// The maximum number of probes to search. If None, all partitions may be
    /// searched if needed to satisfy k results.
    pub maximum_nprobes: Option<usize>,
    /// Distance metric type. If None, uses the index's metric.
    pub distance_type: Option<DistanceType>,
    /// Number of candidates to reserve for HNSW search.
    pub ef: Option<usize>,
    /// Refine factor for re-ranking results using original vectors.
    pub refine_factor: Option<u32>,
    /// The lower bound (inclusive) of the distance to be searched.
    pub distance_lower_bound: Option<f32>,
    /// The upper bound (exclusive) of the distance to be searched.
    pub distance_upper_bound: Option<f32>,
}

/// Full-text search query type.
#[derive(Debug, Clone)]
pub enum FtsQueryType {
    /// Simple term match.
    Match {
        /// The search query string.
        query: String,
    },
    /// Phrase query with slop.
    Phrase {
        /// The phrase to search for.
        query: String,
        /// Maximum allowed distance between consecutive tokens.
        slop: u32,
    },
    /// Boolean query with MUST/SHOULD/MUST_NOT.
    Boolean {
        /// Terms that must match.
        must: Vec<String>,
        /// Terms that should match (adds to score).
        should: Vec<String>,
        /// Terms that must not match.
        must_not: Vec<String>,
    },
    /// Fuzzy match query with typo tolerance.
    Fuzzy {
        /// The search query string.
        query: String,
        /// Maximum edit distance (Levenshtein distance).
        /// None means auto-fuzziness based on token length.
        fuzziness: Option<u32>,
        /// Maximum number of terms to expand to.
        max_expansions: usize,
    },
}

/// Full-text search query parameters.
#[derive(Debug, Clone)]
pub struct FtsQuery {
    /// Column name to search.
    pub column: String,
    /// Query type.
    pub query_type: FtsQueryType,
    /// WAND factor for early termination (0.0 to 1.0).
    /// 1.0 = full recall (default), <1.0 = faster but may miss low-scoring results.
    pub wand_factor: f32,
}

/// Default maximum number of fuzzy expansions.
pub const DEFAULT_MAX_EXPANSIONS: usize = 50;

/// Default WAND factor for full recall (no early termination).
pub const DEFAULT_WAND_FACTOR: f32 = 1.0;

impl FtsQuery {
    /// Create a simple term match query.
    pub fn match_query(column: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            query_type: FtsQueryType::Match {
                query: query.into(),
            },
            wand_factor: DEFAULT_WAND_FACTOR,
        }
    }

    /// Create a phrase query.
    pub fn phrase(column: impl Into<String>, query: impl Into<String>, slop: u32) -> Self {
        Self {
            column: column.into(),
            query_type: FtsQueryType::Phrase {
                query: query.into(),
                slop,
            },
            wand_factor: DEFAULT_WAND_FACTOR,
        }
    }

    /// Create a Boolean query.
    pub fn boolean(
        column: impl Into<String>,
        must: Vec<String>,
        should: Vec<String>,
        must_not: Vec<String>,
    ) -> Self {
        Self {
            column: column.into(),
            query_type: FtsQueryType::Boolean {
                must,
                should,
                must_not,
            },
            wand_factor: DEFAULT_WAND_FACTOR,
        }
    }

    /// Create a fuzzy match query with auto-fuzziness.
    ///
    /// Auto-fuzziness is calculated based on token length:
    /// - 0-2 chars: 0 (exact match)
    /// - 3-5 chars: 1 edit allowed
    /// - 6+ chars: 2 edits allowed
    pub fn fuzzy(column: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            query_type: FtsQueryType::Fuzzy {
                query: query.into(),
                fuzziness: None,
                max_expansions: DEFAULT_MAX_EXPANSIONS,
            },
            wand_factor: DEFAULT_WAND_FACTOR,
        }
    }

    /// Create a fuzzy match query with specified edit distance.
    pub fn fuzzy_with_distance(
        column: impl Into<String>,
        query: impl Into<String>,
        fuzziness: u32,
    ) -> Self {
        Self {
            column: column.into(),
            query_type: FtsQueryType::Fuzzy {
                query: query.into(),
                fuzziness: Some(fuzziness),
                max_expansions: DEFAULT_MAX_EXPANSIONS,
            },
            wand_factor: DEFAULT_WAND_FACTOR,
        }
    }

    /// Create a fuzzy match query with full options.
    pub fn fuzzy_with_options(
        column: impl Into<String>,
        query: impl Into<String>,
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> Self {
        Self {
            column: column.into(),
            query_type: FtsQueryType::Fuzzy {
                query: query.into(),
                fuzziness,
                max_expansions,
            },
            wand_factor: DEFAULT_WAND_FACTOR,
        }
    }

    /// Set the WAND factor for early termination.
    ///
    /// - 1.0 = full recall (default)
    /// - 0.5 = prune documents scoring below 50% of the k-th best score
    /// - 0.0 = only return the absolute best match
    pub fn with_wand_factor(mut self, wand_factor: f32) -> Self {
        self.wand_factor = wand_factor.clamp(0.0, 1.0);
        self
    }
}

/// Scalar predicate for BTree index queries.
#[derive(Debug, Clone)]
pub enum ScalarPredicate {
    /// Exact match: column = value.
    Eq { column: String, value: ScalarValue },
    /// Range query: column in [lower, upper).
    Range {
        column: String,
        lower: Option<ScalarValue>,
        upper: Option<ScalarValue>,
    },
    /// IN query: column in (values...).
    In {
        column: String,
        values: Vec<ScalarValue>,
    },
}

impl ScalarPredicate {
    /// Get the column name for this predicate.
    pub fn column(&self) -> &str {
        match self {
            Self::Eq { column, .. } => column,
            Self::Range { column, .. } => column,
            Self::In { column, .. } => column,
        }
    }
}

/// Scanner builder for querying MemTable data.
///
/// Provides a builder pattern similar to Lance's Scanner interface
/// for constructing DataFusion execution plans over in-memory data.
///
/// # Index Visibility Model
///
/// The scanner captures `max_indexed_batch_position` from the `IndexStore` at
/// construction time. This frozen visibility ensures queries only see data
/// that has been indexed, providing consistent results.
///
/// # Example
///
/// ```ignore
/// let scanner = MemTableScanner::new(batch_store, indexes, schema)
///     .project(&["id", "name"])?
///     .filter("id > 10")?
///     .limit(100, None)?;
///
/// let stream = scanner.try_into_stream().await?;
/// ```
pub struct MemTableScanner {
    batch_store: Arc<BatchStore>,
    indexes: Arc<IndexStore>,
    schema: SchemaRef,
    /// Frozen visibility captured at scanner construction time.
    /// This is the `max_indexed_batch_position` from the IndexStore.
    max_visible_batch_position: usize,
    projection: Option<Vec<String>>,
    filter: Option<Expr>,
    limit: Option<usize>,
    offset: Option<usize>,
    nearest: Option<VectorQuery>,
    full_text_query: Option<FtsQuery>,
    use_index: bool,
    batch_size: Option<usize>,
    /// Whether to include _rowid column in output.
    /// In MemTable, _rowid is the row_position (global row offset).
    with_row_id: bool,
}

impl MemTableScanner {
    /// Create a new scanner.
    ///
    /// Captures `max_indexed_batch_position` from the `IndexStore` at construction
    /// time to ensure consistent query visibility.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - Lock-free batch store containing the data
    /// * `indexes` - Index registry (required for visibility tracking)
    /// * `schema` - Schema of the data
    pub fn new(batch_store: Arc<BatchStore>, indexes: Arc<IndexStore>, schema: SchemaRef) -> Self {
        // Capture max_indexed_batch_position at construction time
        let max_visible_batch_position = indexes.max_indexed_batch_position();

        Self {
            batch_store,
            indexes,
            schema,
            max_visible_batch_position,
            projection: None,
            filter: None,
            limit: None,
            offset: None,
            nearest: None,
            full_text_query: None,
            use_index: true,
            batch_size: None,
            with_row_id: false,
        }
    }

    /// Project only the specified columns.
    ///
    /// Special columns:
    /// - `_rowid`: Returns the row position (global row offset in MemTable)
    pub fn project(&mut self, columns: &[&str]) -> &mut Self {
        // Check if _rowid is requested in projection
        let mut filtered_columns = Vec::new();
        for col in columns {
            if *col == ROW_ID {
                self.with_row_id = true;
            } else {
                filtered_columns.push(col.to_string());
            }
        }
        // Only set projection if there are non-special columns
        if !filtered_columns.is_empty() || self.with_row_id {
            self.projection = Some(filtered_columns);
        }
        self
    }

    /// Include the _rowid column in output.
    ///
    /// In MemTable, _rowid is the row_position (global row offset).
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    /// Set a filter expression using SQL-like syntax.
    pub fn filter(&mut self, filter_expr: &str) -> Result<&mut Self> {
        let ctx = SessionContext::new();
        let df_schema = self.schema.clone().to_dfschema().map_err(|e| {
            Error::invalid_input(format!("Failed to create DFSchema: {}", e), location!())
        })?;
        let expr = ctx.parse_sql_expr(filter_expr, &df_schema).map_err(|e| {
            Error::invalid_input(
                format!("Failed to parse filter expression: {}", e),
                location!(),
            )
        })?;
        self.filter = Some(expr);
        Ok(self)
    }

    /// Set a filter expression directly.
    pub fn filter_expr(&mut self, expr: Expr) -> &mut Self {
        self.filter = Some(expr);
        self
    }

    /// Limit the number of results.
    pub fn limit(&mut self, limit: usize, offset: Option<usize>) -> &mut Self {
        self.limit = Some(limit);
        self.offset = offset;
        self
    }

    /// Set up a vector similarity search.
    ///
    /// # Arguments
    ///
    /// * `column` - The name of the vector column to search.
    /// * `query` - The query vector.
    /// * `k` - Number of nearest neighbors to return.
    pub fn nearest(&mut self, column: &str, query: Arc<dyn Array>, k: usize) -> &mut Self {
        self.nearest = Some(VectorQuery {
            column: column.to_string(),
            query_vector: query,
            k,
            nprobes: 1,
            maximum_nprobes: None,
            distance_type: None,
            ef: None,
            refine_factor: None,
            distance_lower_bound: None,
            distance_upper_bound: None,
        });
        self
    }

    /// Set the number of probes for IVF search.
    ///
    /// This is a convenience method that sets both minimum and maximum nprobes
    /// to the same value, guaranteeing exactly `n` partitions will be searched.
    pub fn nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.nprobes = n;
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the minimum number of probes for IVF search.
    ///
    /// This is the minimum number of partitions to search. More partitions may be
    /// searched if needed to satisfy k results or recall requirements. Defaults to 1.
    pub fn minimum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.nprobes = n;
        } else {
            log::warn!("minimum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the maximum number of probes for IVF search.
    ///
    /// If not set, all partitions may be searched if needed to satisfy k results.
    pub fn maximum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("maximum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the distance metric type for vector search.
    ///
    /// If not set, uses the index's default metric type.
    pub fn distance_metric(&mut self, metric: DistanceType) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.distance_type = Some(metric);
        } else {
            log::warn!("distance_metric is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the ef parameter for HNSW search.
    ///
    /// The number of candidates to reserve while searching. This controls the
    /// accuracy/speed tradeoff for HNSW-based indices.
    pub fn ef(&mut self, ef: usize) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.ef = Some(ef);
        } else {
            log::warn!("ef is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the refine factor for re-ranking results.
    ///
    /// When set, the search will first retrieve `k * refine_factor` candidates
    /// using the approximate index, then re-rank them using the original vectors.
    pub fn refine(&mut self, factor: u32) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.refine_factor = Some(factor);
        } else {
            log::warn!("refine is not set because nearest has not been called yet");
        }
        self
    }

    /// Set the distance range for filtering results.
    ///
    /// * `lower` - The lower bound (inclusive) of the distance.
    /// * `upper` - The upper bound (exclusive) of the distance.
    pub fn distance_range(&mut self, lower: Option<f32>, upper: Option<f32>) -> &mut Self {
        if let Some(ref mut q) = self.nearest {
            q.distance_lower_bound = lower;
            q.distance_upper_bound = upper;
        } else {
            log::warn!("distance_range is not set because nearest has not been called yet");
        }
        self
    }

    /// Set up a full-text search with simple term matching.
    pub fn full_text_search(&mut self, column: &str, query: &str) -> &mut Self {
        self.full_text_query = Some(FtsQuery::match_query(column, query));
        self
    }

    /// Set up a full-text phrase search.
    ///
    /// # Arguments
    ///
    /// * `column` - The column to search.
    /// * `phrase` - The phrase to search for.
    /// * `slop` - Maximum allowed distance between consecutive tokens.
    ///   0 means exact phrase match (tokens must be adjacent).
    pub fn full_text_phrase(&mut self, column: &str, phrase: &str, slop: u32) -> &mut Self {
        self.full_text_query = Some(FtsQuery::phrase(column, phrase, slop));
        self
    }

    /// Set up a full-text Boolean search.
    ///
    /// # Arguments
    ///
    /// * `column` - The column to search.
    /// * `must` - Terms that must match (intersection).
    /// * `should` - Terms that should match (adds to score).
    /// * `must_not` - Terms that must not match (exclusion).
    pub fn full_text_boolean(
        &mut self,
        column: &str,
        must: Vec<String>,
        should: Vec<String>,
        must_not: Vec<String>,
    ) -> &mut Self {
        self.full_text_query = Some(FtsQuery::boolean(column, must, should, must_not));
        self
    }

    /// Set up a full-text fuzzy search with auto-fuzziness.
    ///
    /// Auto-fuzziness is calculated based on token length:
    /// - 0-2 chars: 0 (exact match)
    /// - 3-5 chars: 1 edit allowed
    /// - 6+ chars: 2 edits allowed
    ///
    /// # Arguments
    ///
    /// * `column` - The column to search.
    /// * `query` - The search query (may contain typos).
    pub fn full_text_fuzzy(&mut self, column: &str, query: &str) -> &mut Self {
        self.full_text_query = Some(FtsQuery::fuzzy(column, query));
        self
    }

    /// Set up a full-text fuzzy search with specified edit distance.
    ///
    /// # Arguments
    ///
    /// * `column` - The column to search.
    /// * `query` - The search query (may contain typos).
    /// * `fuzziness` - Maximum edit distance (Levenshtein distance).
    pub fn full_text_fuzzy_with_distance(
        &mut self,
        column: &str,
        query: &str,
        fuzziness: u32,
    ) -> &mut Self {
        self.full_text_query = Some(FtsQuery::fuzzy_with_distance(column, query, fuzziness));
        self
    }

    /// Set up a full-text fuzzy search with full options.
    ///
    /// # Arguments
    ///
    /// * `column` - The column to search.
    /// * `query` - The search query (may contain typos).
    /// * `fuzziness` - Maximum edit distance. None means auto-fuzziness.
    /// * `max_expansions` - Maximum number of terms to expand to.
    pub fn full_text_fuzzy_with_options(
        &mut self,
        column: &str,
        query: &str,
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> &mut Self {
        self.full_text_query = Some(FtsQuery::fuzzy_with_options(
            column,
            query,
            fuzziness,
            max_expansions,
        ));
        self
    }

    /// Set the WAND factor for FTS queries to control performance/recall tradeoff.
    ///
    /// This only applies when a full-text query is set.
    ///
    /// - 1.0 = full recall (default)
    /// - 0.5 = prune documents scoring below 50% of the k-th best score
    /// - 0.0 = only return the absolute best match
    ///
    /// # Arguments
    ///
    /// * `wand_factor` - Value between 0.0 and 1.0
    pub fn fts_wand_factor(&mut self, wand_factor: f32) -> &mut Self {
        if let Some(ref mut q) = self.full_text_query {
            q.wand_factor = wand_factor.clamp(0.0, 1.0);
        } else {
            log::warn!(
                "fts_wand_factor is not set because full_text_query has not been called yet"
            );
        }
        self
    }

    /// Enable or disable index usage.
    pub fn use_index(&mut self, use_index: bool) -> &mut Self {
        self.use_index = use_index;
        self
    }

    /// Set the batch size for output.
    pub fn batch_size(&mut self, size: usize) -> &mut Self {
        self.batch_size = Some(size);
        self
    }

    /// Execute the scan and return a stream of record batches.
    pub async fn try_into_stream(&self) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        plan.execute(0, task_ctx)
            .map_err(|e| Error::io(format!("Failed to execute plan: {}", e), location!()))
    }

    /// Execute the scan and collect all results into a single RecordBatch.
    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| Error::io(format!("Failed to collect batches: {}", e), location!()))?;

        if batches.is_empty() {
            return Ok(RecordBatch::new_empty(self.output_schema()));
        }

        arrow_select::concat::concat_batches(&self.output_schema(), &batches)
            .map_err(|e| Error::io(format!("Failed to concatenate batches: {}", e), location!()))
    }

    /// Count the number of rows that match the query.
    pub async fn count_rows(&self) -> Result<u64> {
        let stream = self.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| Error::io(format!("Failed to count rows: {}", e), location!()))?;

        Ok(batches.iter().map(|b| b.num_rows() as u64).sum())
    }

    /// Get the output schema after projection.
    ///
    /// If `with_row_id` is true, adds `_rowid` column at the end.
    pub fn output_schema(&self) -> SchemaRef {
        let mut fields: Vec<Field> = if let Some(ref projection) = self.projection {
            projection
                .iter()
                .filter_map(|name| self.schema.field_with_name(name).ok().cloned())
                .collect()
        } else {
            self.schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect()
        };

        // Add _rowid column if requested
        if self.with_row_id {
            fields.push(Field::new(ROW_ID, DataType::UInt64, true));
        }

        Arc::new(arrow_schema::Schema::new(fields))
    }

    /// Get the base output schema after projection, WITHOUT special columns like _rowid.
    /// This is used by index execs that add their own special columns.
    fn base_output_schema(&self) -> SchemaRef {
        let fields: Vec<Field> = if let Some(ref projection) = self.projection {
            projection
                .iter()
                .filter_map(|name| self.schema.field_with_name(name).ok().cloned())
                .collect()
        } else {
            self.schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect()
        };
        Arc::new(arrow_schema::Schema::new(fields))
    }

    /// Create the execution plan based on the query configuration.
    pub async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        // Determine which type of plan to create
        if let Some(ref vector_query) = self.nearest {
            return self.plan_vector_search(vector_query).await;
        }

        if let Some(ref fts_query) = self.full_text_query {
            return self.plan_fts_search(fts_query).await;
        }

        // Check if we can use a BTree index for the filter
        if self.use_index {
            if let Some(predicate) = self.extract_btree_predicate() {
                if self.has_btree_index(predicate.column()) {
                    return self.plan_btree_query(&predicate).await;
                }
            }
        }

        // Fall back to full scan
        self.plan_full_scan().await
    }

    /// Plan a full table scan.
    async fn plan_full_scan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let projection_indices = self.compute_projection_indices()?;

        // Build filter predicate if present
        let (filter_predicate, filter_expr) = if let Some(ref filter) = self.filter {
            let planner = Planner::new(self.schema.clone());
            let predicate = planner.create_physical_expr(filter)?;
            (Some(predicate), Some(filter.clone()))
        } else {
            (None, None)
        };

        let scan = MemTableScanExec::with_filter(
            self.batch_store.clone(),
            self.max_visible_batch_position,
            projection_indices,
            self.output_schema(),
            self.schema.clone(),
            self.with_row_id,
            filter_predicate,
            filter_expr,
        );

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(scan);

        // Apply limit if present
        if let Some(limit) = self.limit {
            plan = Arc::new(GlobalLimitExec::new(
                plan,
                self.offset.unwrap_or(0),
                Some(limit),
            ));
        }

        Ok(plan)
    }

    /// Plan a BTree index query.
    ///
    /// Uses the effective visibility (min of max_visible and max_indexed) to ensure
    /// queries only see indexed data. Falls back to full scan if no index exists.
    async fn plan_btree_query(
        &self,
        predicate: &ScalarPredicate,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !self.has_btree_index(predicate.column()) {
            return self.plan_full_scan().await;
        }

        let max_visible = self.max_visible_batch_position;
        let projection_indices = self.compute_projection_indices()?;

        let index_exec = BTreeIndexExec::new(
            self.batch_store.clone(),
            self.indexes.clone(),
            predicate.clone(),
            max_visible,
            projection_indices,
            self.output_schema(),
            self.with_row_id,
        )?;
        self.apply_post_index_ops(Arc::new(index_exec)).await
    }

    /// Plan a vector similarity search.
    ///
    /// Uses the effective visibility (min of max_visible and max_indexed) to ensure
    /// queries only see indexed data. Falls back to full scan if no index exists.
    async fn plan_vector_search(&self, query: &VectorQuery) -> Result<Arc<dyn ExecutionPlan>> {
        if !self.has_vector_index(&query.column) {
            return self.plan_full_scan().await;
        }

        let max_visible = self.max_visible_batch_position;
        let projection_indices = self.compute_projection_indices()?;

        let index_exec = VectorIndexExec::new(
            self.batch_store.clone(),
            self.indexes.clone(),
            query.clone(),
            max_visible,
            projection_indices,
            self.base_output_schema(),
            self.with_row_id,
        )?;
        self.apply_post_index_ops(Arc::new(index_exec)).await
    }

    /// Plan a full-text search.
    ///
    /// Uses the effective visibility (min of max_visible and max_indexed) to ensure
    /// queries only see indexed data. Falls back to full scan if no index exists.
    async fn plan_fts_search(&self, query: &FtsQuery) -> Result<Arc<dyn ExecutionPlan>> {
        if !self.has_fts_index(&query.column) {
            return self.plan_full_scan().await;
        }

        let max_visible = self.max_visible_batch_position;
        let projection_indices = self.compute_projection_indices()?;

        let index_exec = FtsIndexExec::new(
            self.batch_store.clone(),
            self.indexes.clone(),
            query.clone(),
            max_visible,
            projection_indices,
            self.base_output_schema(),
            self.with_row_id,
        )?;
        self.apply_post_index_ops(Arc::new(index_exec)).await
    }

    /// Apply limit and other post-processing operations.
    async fn apply_post_index_ops(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut result = plan;

        if let Some(limit) = self.limit {
            result = Arc::new(GlobalLimitExec::new(
                result,
                self.offset.unwrap_or(0),
                Some(limit),
            ));
        }

        Ok(result)
    }

    /// Compute column indices for projection.
    fn compute_projection_indices(&self) -> Result<Option<Vec<usize>>> {
        if let Some(ref columns) = self.projection {
            let indices: Result<Vec<usize>> = columns
                .iter()
                .map(|name| {
                    self.schema
                        .column_with_name(name)
                        .map(|(idx, _)| idx)
                        .ok_or_else(|| {
                            Error::invalid_input(
                                format!("Column '{}' not found in schema", name),
                                location!(),
                            )
                        })
                })
                .collect();
            Ok(Some(indices?))
        } else {
            Ok(None)
        }
    }

    /// Extract a BTree-compatible predicate from the filter.
    fn extract_btree_predicate(&self) -> Option<ScalarPredicate> {
        let filter = self.filter.as_ref()?;

        // Simple pattern matching for common predicates
        match filter {
            Expr::BinaryExpr(binary) => {
                if let (Expr::Column(col), Expr::Literal(lit, _)) =
                    (binary.left.as_ref(), binary.right.as_ref())
                {
                    match binary.op {
                        datafusion::logical_expr::Operator::Eq => {
                            return Some(ScalarPredicate::Eq {
                                column: col.name.clone(),
                                value: lit.clone(),
                            });
                        }
                        datafusion::logical_expr::Operator::Lt
                        | datafusion::logical_expr::Operator::LtEq => {
                            return Some(ScalarPredicate::Range {
                                column: col.name.clone(),
                                lower: None,
                                upper: Some(lit.clone()),
                            });
                        }
                        datafusion::logical_expr::Operator::Gt
                        | datafusion::logical_expr::Operator::GtEq => {
                            return Some(ScalarPredicate::Range {
                                column: col.name.clone(),
                                lower: Some(lit.clone()),
                                upper: None,
                            });
                        }
                        _ => {}
                    }
                }
            }
            Expr::InList(in_list) => {
                if let Expr::Column(col) = in_list.expr.as_ref() {
                    let values: Vec<ScalarValue> = in_list
                        .list
                        .iter()
                        .filter_map(|e| {
                            if let Expr::Literal(lit, _) = e {
                                Some(lit.clone())
                            } else {
                                None
                            }
                        })
                        .collect();

                    if values.len() == in_list.list.len() {
                        return Some(ScalarPredicate::In {
                            column: col.name.clone(),
                            values,
                        });
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Check if a BTree index exists for a column.
    fn has_btree_index(&self, column: &str) -> bool {
        self.indexes.get_btree_by_column(column).is_some()
    }

    /// Check if a vector index exists for a column.
    fn has_vector_index(&self, column: &str) -> bool {
        self.indexes.get_ivf_pq_by_column(column).is_some()
    }

    /// Check if an FTS index exists for a column.
    fn has_fts_index(&self, column: &str) -> bool {
        self.indexes.get_fts_by_column(column).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &Schema, start_id: i32, count: usize) -> RecordBatch {
        let ids: Vec<i32> = (start_id..start_id + count as i32).collect();
        let names: Vec<String> = ids.iter().map(|id| format!("name_{}", id)).collect();

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    /// Create an IndexStore and insert batches with batch position tracking.
    fn create_index_store_with_batches(
        batch_store: &Arc<BatchStore>,
        schema: &Schema,
        batches: &[(i32, usize)], // (start_id, count)
    ) -> Arc<IndexStore> {
        let mut index_store = IndexStore::new();
        // Add a btree index on "id" column
        index_store.add_btree("id_idx".to_string(), 0, "id".to_string());

        let mut row_offset = 0u64;
        for (batch_pos, (start_id, count)) in batches.iter().enumerate() {
            let batch = create_test_batch(schema, *start_id, *count);
            batch_store.append(batch.clone()).unwrap();

            // Insert into indexes with batch position tracking
            index_store
                .insert_with_batch_position(&batch, row_offset, Some(batch_pos))
                .unwrap();

            row_offset += *count as u64;
        }

        Arc::new(index_store)
    }

    #[tokio::test]
    async fn test_scanner_basic_scan() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        // Insert test data with index tracking
        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let scanner = MemTableScanner::new(batch_store, indexes, schema.clone());

        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), 10);
    }

    #[tokio::test]
    async fn test_scanner_visibility_filtering() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        // Create index store and insert 2 batches (positions 0, 1)
        let mut index_store = IndexStore::new();
        index_store.add_btree("id_idx".to_string(), 0, "id".to_string());

        let batch1 = create_test_batch(&schema, 0, 10);
        batch_store.append(batch1.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch1, 0, Some(0))
            .unwrap();

        let batch2 = create_test_batch(&schema, 10, 10);
        batch_store.append(batch2.clone()).unwrap();
        index_store
            .insert_with_batch_position(&batch2, 10, Some(1))
            .unwrap();

        // Add a third batch to batch_store but DON'T index it
        let batch3 = create_test_batch(&schema, 20, 10);
        batch_store.append(batch3).unwrap();

        // Scanner should only see indexed data (batches 0 and 1)
        let indexes = Arc::new(index_store);
        let scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        let result = scanner.try_into_batch().await.unwrap();
        // max_indexed_batch_position is 1, so we see batches 0 and 1 (20 rows)
        assert_eq!(result.num_rows(), 20);
    }

    #[tokio::test]
    async fn test_scanner_projection() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        scanner.project(&["id"]);

        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_columns(), 1);
        assert_eq!(result.schema().field(0).name(), "id");
    }

    #[tokio::test]
    async fn test_scanner_limit() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 100)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        scanner.limit(10, None);

        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), 10);
    }

    #[tokio::test]
    async fn test_scanner_count_rows() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 50)]);

        let scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        let count = scanner.count_rows().await.unwrap();
        assert_eq!(count, 50);
    }

    #[tokio::test]
    async fn test_scanner_with_row_id() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        scanner.with_row_id();

        // Verify output schema includes _rowid
        let output_schema = scanner.output_schema();
        assert_eq!(output_schema.fields().len(), 3);
        assert_eq!(output_schema.field(0).name(), "id");
        assert_eq!(output_schema.field(1).name(), "name");
        assert_eq!(output_schema.field(2).name(), "_rowid");
        assert_eq!(output_schema.field(2).data_type(), &DataType::UInt64);

        // Verify data includes correct row IDs
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.schema().field(2).name(), "_rowid");

        let row_ids = result
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        assert_eq!(row_ids.len(), 10);
        // Row IDs should be 0-9 for a single batch
        for i in 0..10 {
            assert_eq!(row_ids.value(i), i as u64);
        }
    }

    #[tokio::test]
    async fn test_scanner_project_with_row_id() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        // Project only "id" and "_rowid"
        scanner.project(&["id", "_rowid"]);

        // Verify output schema
        let output_schema = scanner.output_schema();
        assert_eq!(output_schema.fields().len(), 2);
        assert_eq!(output_schema.field(0).name(), "id");
        assert_eq!(output_schema.field(1).name(), "_rowid");

        // Verify data
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "_rowid");
    }

    #[tokio::test]
    async fn test_scanner_row_id_across_batches() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        // Insert two batches with 5 rows each
        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 5), (5, 5)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        scanner.with_row_id();

        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), 10);

        let row_ids = result
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();

        // Row IDs should be 0-9 across both batches
        for i in 0..10 {
            assert_eq!(row_ids.value(i), i as u64);
        }
    }

    #[test]
    fn test_output_schema_with_row_id() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let indexes = Arc::new(IndexStore::new());

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema);

        // Without with_row_id, schema should not include _rowid
        let output_schema = scanner.output_schema();
        assert_eq!(output_schema.fields().len(), 2);
        assert!(output_schema.field_with_name("_rowid").is_err());

        // With with_row_id, schema should include _rowid
        scanner.with_row_id();
        let output_schema = scanner.output_schema();
        assert_eq!(output_schema.fields().len(), 3);
        assert!(output_schema.field_with_name("_rowid").is_ok());
    }

    #[test]
    fn test_project_extracts_row_id() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let indexes = Arc::new(IndexStore::new());

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema);

        // Project with _rowid should set with_row_id flag
        scanner.project(&["id", "_rowid"]);

        // with_row_id should be true now
        assert!(scanner.with_row_id);

        // _rowid should not be in projection list (it's handled separately)
        assert_eq!(scanner.projection, Some(vec!["id".to_string()]));

        // Output schema should include _rowid at the end
        let output_schema = scanner.output_schema();
        assert_eq!(output_schema.fields().len(), 2);
        assert_eq!(output_schema.field(0).name(), "id");
        assert_eq!(output_schema.field(1).name(), "_rowid");
    }

    #[tokio::test]
    async fn test_scan_plan_with_row_id() {
        use crate::utils::test::assert_plan_node_equals;

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        scanner.with_row_id();

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan structure using assert_plan_node_equals
        assert_plan_node_equals(
            plan,
            "MemTableScanExec: projection=[id, name, _rowid], with_row_id=true",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_scan_plan_projection_with_row_id() {
        use crate::utils::test::assert_plan_node_equals;

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let mut scanner = MemTableScanner::new(batch_store, indexes, schema.clone());
        scanner.project(&["id", "_rowid"]);

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan structure with projection
        assert_plan_node_equals(
            plan,
            "MemTableScanExec: projection=[id, _rowid], with_row_id=true",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_scan_plan_without_row_id() {
        use crate::utils::test::assert_plan_node_equals;

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let indexes = create_index_store_with_batches(&batch_store, &schema, &[(0, 10)]);

        let scanner = MemTableScanner::new(batch_store, indexes, schema.clone());

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan structure without _rowid
        assert_plan_node_equals(
            plan,
            "MemTableScanExec: projection=[id, name], with_row_id=false",
        )
        .await
        .unwrap();
    }
}
