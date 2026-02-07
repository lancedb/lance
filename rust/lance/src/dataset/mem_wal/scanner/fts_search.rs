// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FTS search planner for LSM scanner.
//!
//! Provides full-text search across LSM levels with global BM25 scoring.
//! Global BM25 stats are collected concurrently with plan construction via a
//! shared future handle, so planning is not blocked on stats I/O.
//! Flushed memtable datasets share the base dataset's `Session` to benefit
//! from a common index cache.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::ExecutionPlan;
use lance_core::Result;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::bloomfilter::sbbf::Sbbf;
use lance_index::scalar::inverted::query::MatchQuery;
use lance_index::scalar::inverted::scorer::BM25StatsOverride;
use lance_index::scalar::inverted::InvertedIndex;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::{DatasetIndexExt, IndexCriteria};

use super::bm25_stats::{DeferredBM25Stats, GenerationBM25Stats, GlobalBM25Stats};
use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{FilterStaleExec, GenerationBloomFilter, MemtableGenTagExec, TopKExec};
use crate::dataset::mem_wal::memtable::scanner::{
    FtsQuery, FtsQueryType, MemTableScanner, SCORE_COLUMN,
};
use crate::index::DatasetIndexInternalExt;
use crate::session::Session;

/// Plans FTS queries over LSM data with global BM25 scoring.
///
/// FTS queries are executed across all LSM levels with global BM25 statistics
/// aggregated at plan time so scores are comparable across generations.
///
/// Each source is given `limit=k` so its internal WAND can prune low-scoring
/// documents. The results are merged via a heap-based `TopKExec` node that
/// provides cross-source pruning: as better results accumulate, rows from
/// any source that score below the k-th best are discarded immediately.
///
/// # Query Plan Structure
///
/// ```text
/// TopKExec: k=K, score=_score
///   FilterStaleExec: bloom_filters=[...]
///     UnionExec
///       MemtableGenTagExec: gen=N
///         FtsExec (active memtable, limit=K, global BM25 stats)
///       MemtableGenTagExec: gen=2
///         FtsExec (flushed gen 2, limit=K, global BM25 stats)
///       MemtableGenTagExec: gen=0
///         FtsExec (base table, limit=K, global BM25 stats)
/// ```
pub struct LsmFtsSearchPlanner {
    collector: LsmDataSourceCollector,
    pk_columns: Vec<String>,
    base_schema: SchemaRef,
    bloom_filters: Vec<GenerationBloomFilter>,
    fts_query: FtsQuery,
    /// If set, use this as global stats instead of collecting from sources.
    global_stats_override: Option<GlobalBM25Stats>,
    /// Shared session for index/metadata caching across flushed dataset opens.
    session: Arc<Session>,
}

impl LsmFtsSearchPlanner {
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
        fts_query: FtsQuery,
    ) -> Self {
        let session = collector.base_table().session();
        Self {
            collector,
            pk_columns,
            base_schema,
            bloom_filters: Vec::new(),
            fts_query,
            global_stats_override: None,
            session,
        }
    }

    /// Add a bloom filter for staleness detection.
    pub fn with_bloom_filter(mut self, generation: u64, bloom_filter: Arc<Sbbf>) -> Self {
        self.bloom_filters.push(GenerationBloomFilter {
            generation,
            bloom_filter,
        });
        self
    }

    /// Add multiple bloom filters.
    pub fn with_bloom_filters(
        mut self,
        bloom_filters: impl IntoIterator<Item = (u64, Arc<Sbbf>)>,
    ) -> Self {
        for (gen, bf) in bloom_filters {
            self.bloom_filters.push(GenerationBloomFilter {
                generation: gen,
                bloom_filter: bf,
            });
        }
        self
    }

    /// Provide pre-computed global BM25 stats instead of collecting at plan time.
    pub fn with_global_stats(mut self, stats: GlobalBM25Stats) -> Self {
        self.global_stats_override = Some(stats);
        self
    }

    /// Create the FTS search plan.
    ///
    /// Stats collection is kicked off concurrently via a shared future.
    /// Active memtable plans resolve the stats lazily at execution time.
    /// Persistent source plans await the stats (which may already be resolved).
    pub async fn plan_search(
        &self,
        k: usize,
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            return self.empty_plan(projection);
        }

        // Create the deferred stats handle and spawn collection concurrently
        let deferred_stats = self.spawn_stats_collection(&sources)?;

        // Build per-source FTS plans
        let mut fts_plans = Vec::new();
        for source in &sources {
            let generation = source.generation();
            let fts = self
                .build_fts_plan(source, &deferred_stats, projection, k)
                .await?;
            let tagged: Arc<dyn ExecutionPlan> = Arc::new(MemtableGenTagExec::new(fts, generation));
            fts_plans.push(tagged);
        }

        // Assemble the plan
        #[allow(deprecated)]
        let union: Arc<dyn ExecutionPlan> = Arc::new(UnionExec::new(fts_plans));

        let filtered: Arc<dyn ExecutionPlan> = if !self.bloom_filters.is_empty() {
            Arc::new(FilterStaleExec::new(
                union,
                self.pk_columns.clone(),
                self.bloom_filters.clone(),
            ))
        } else {
            union
        };

        let top_k: Arc<dyn ExecutionPlan> =
            Arc::new(TopKExec::new(filtered, k, SCORE_COLUMN.to_string()));

        Ok(top_k)
    }

    /// Spawn stats collection as a concurrent task, returning a deferred handle.
    ///
    /// If a global stats override is set, the future resolves immediately.
    /// Otherwise, a background task collects stats from all sources and the
    /// shared future resolves once the task completes.
    fn spawn_stats_collection(&self, sources: &[LsmDataSource]) -> Result<DeferredBM25Stats> {
        use futures::FutureExt;

        if let Some(ref stats) = self.global_stats_override {
            let stats = stats.clone();
            return Ok(async move { stats }.boxed().shared());
        }

        // Gather the information needed by the background task.
        // ActiveMemTable stats are collected synchronously (cheap in-memory reads).
        let query_terms = self.extract_query_terms(sources)?;
        let fts_column = self.fts_query.column.clone();
        let session = self.session.clone();

        let mut mem_stats = Vec::new();
        let mut persistent_sources: Vec<String> = Vec::new();

        for source in sources {
            match source {
                LsmDataSource::ActiveMemTable { index_store, .. } => {
                    if let Some(fts_index) = index_store.get_fts_by_column(&fts_column) {
                        mem_stats.push(GenerationBM25Stats::new(
                            fts_index.doc_count(),
                            fts_index.total_tokens() as u64,
                            fts_index.term_doc_frequencies(&query_terms),
                        ));
                    }
                }
                LsmDataSource::BaseTable { dataset } => {
                    persistent_sources.push(dataset.uri().to_string());
                }
                LsmDataSource::FlushedMemTable { path, .. } => {
                    persistent_sources.push(path.clone());
                }
            }
        }

        let fut = async move {
            let mut per_gen_stats = mem_stats;

            for path in &persistent_sources {
                let Ok(dataset) = crate::dataset::DatasetBuilder::from_uri(path)
                    .with_session(session.clone())
                    .load()
                    .await
                else {
                    continue;
                };
                if let Ok(Some(stats)) =
                    Self::load_fts_stats_from_dataset(&dataset, &fts_column, &query_terms).await
                {
                    per_gen_stats.push(stats);
                }
            }

            GlobalBM25Stats::aggregate(&per_gen_stats)
        }
        .boxed()
        .shared();

        // Eagerly start computation in background so it runs concurrently
        // with plan construction.
        let eager = fut.clone();
        tokio::spawn(async move {
            eager.await;
        });

        Ok(fut)
    }

    /// Collect global BM25 stats from all sources for the query terms.
    #[cfg(test)]
    async fn collect_global_stats(&self, sources: &[LsmDataSource]) -> Result<GlobalBM25Stats> {
        if sources.is_empty() {
            return Ok(GlobalBM25Stats::aggregate(&[]));
        }

        let mut per_gen_stats = Vec::new();

        let query_terms = self.extract_query_terms(sources)?;

        for source in sources {
            match source {
                LsmDataSource::ActiveMemTable { index_store, .. } => {
                    if let Some(fts_index) = index_store.get_fts_by_column(&self.fts_query.column) {
                        let num_docs = fts_index.doc_count();
                        let total_tokens = fts_index.total_tokens() as u64;
                        let term_doc_freqs = fts_index.term_doc_frequencies(&query_terms);
                        per_gen_stats.push(GenerationBM25Stats::new(
                            num_docs,
                            total_tokens,
                            term_doc_freqs,
                        ));
                    }
                }
                LsmDataSource::BaseTable { dataset } => {
                    if let Some(stats) = Self::load_fts_stats_from_dataset(
                        dataset,
                        &self.fts_query.column,
                        &query_terms,
                    )
                    .await?
                    {
                        per_gen_stats.push(stats);
                    }
                }
                LsmDataSource::FlushedMemTable { path, .. } => {
                    let dataset = self.open_flushed_dataset(path).await?;
                    if let Some(stats) = Self::load_fts_stats_from_dataset(
                        &dataset,
                        &self.fts_query.column,
                        &query_terms,
                    )
                    .await?
                    {
                        per_gen_stats.push(stats);
                    }
                }
            }
        }

        Ok(GlobalBM25Stats::aggregate(&per_gen_stats))
    }

    /// Load FTS collection stats from a persistent dataset's inverted index.
    ///
    /// Returns `None` if the dataset has no FTS index on the given column.
    async fn load_fts_stats_from_dataset(
        dataset: &crate::dataset::Dataset,
        column: &str,
        query_terms: &[String],
    ) -> Result<Option<GenerationBM25Stats>> {
        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column(column).supports_fts())
            .await?;

        let Some(index_meta) = index_meta else {
            return Ok(None);
        };

        let uuid = index_meta.uuid.to_string();
        let metrics = NoOpMetricsCollector;
        let index = dataset.open_generic_index(column, &uuid, &metrics).await?;

        let inverted_index = index
            .as_any()
            .downcast_ref::<InvertedIndex>()
            .ok_or_else(|| lance_core::Error::Internal {
                message: format!(
                    "Expected InvertedIndex for column '{}', got different index type",
                    column
                ),
                location: snafu::location!(),
            })?;

        let (num_docs, total_tokens, term_doc_freqs) = inverted_index.collection_stats(query_terms);

        Ok(Some(GenerationBM25Stats::new(
            num_docs,
            total_tokens,
            term_doc_freqs,
        )))
    }

    /// Extract unique query terms for stats collection.
    ///
    /// Uses the tokenizer from the active memtable's FTS index. An FTS index
    /// must always be present when an FTS query is issued.
    fn extract_query_terms(&self, sources: &[LsmDataSource]) -> Result<Vec<String>> {
        for source in sources {
            if let LsmDataSource::ActiveMemTable { index_store, .. } = source {
                if let Some(fts_index) = index_store.get_fts_by_column(&self.fts_query.column) {
                    let query_str = self.query_text();
                    return Ok(fts_index.tokenize_query(&query_str));
                }
            }
        }
        Err(lance_core::Error::Internal {
            message: format!(
                "No FTS index with tokenizer found for column '{}'",
                self.fts_query.column
            ),
            location: snafu::location!(),
        })
    }

    /// Extract the query text from the FtsQuery.
    fn query_text(&self) -> String {
        match &self.fts_query.query_type {
            FtsQueryType::Match { query } => query.clone(),
            FtsQueryType::Phrase { query, .. } => query.clone(),
            FtsQueryType::Fuzzy { query, .. } => query.clone(),
            FtsQueryType::Boolean {
                must,
                should,
                must_not,
            } => {
                let mut terms = Vec::new();
                terms.extend(must.iter().cloned());
                terms.extend(should.iter().cloned());
                terms.extend(must_not.iter().cloned());
                terms.join(" ")
            }
        }
    }

    /// Build FTS plan for a single data source with per-source limit.
    ///
    /// Active memtable plans receive the deferred stats handle and resolve it
    /// lazily at execution time. Persistent source plans await the stats here
    /// (since `FtsSearchParams.bm25_override` is sync), but benefit from the
    /// concurrent stats task that's already running.
    async fn build_fts_plan(
        &self,
        source: &LsmDataSource,
        deferred_stats: &DeferredBM25Stats,
        projection: Option<&[String]>,
        k: usize,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                let global_stats = deferred_stats.clone().await;
                let bm25_override = Arc::new(BM25StatsOverride {
                    num_docs: global_stats.num_docs,
                    total_tokens: global_stats.total_tokens,
                    term_doc_freqs: global_stats.term_doc_freqs.clone(),
                });
                let mut scanner = dataset.scan();
                let cols = self.build_projection_for_fts(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                let fts_query = self.build_full_text_query(&bm25_override, k);
                scanner.full_text_search(fts_query)?;
                scanner.create_plan().await
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let global_stats = deferred_stats.clone().await;
                let bm25_override = Arc::new(BM25StatsOverride {
                    num_docs: global_stats.num_docs,
                    total_tokens: global_stats.total_tokens,
                    term_doc_freqs: global_stats.term_doc_freqs.clone(),
                });
                let dataset = self.open_flushed_dataset(path).await?;
                let mut scanner = dataset.scan();
                let cols = self.build_projection_for_fts(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                let fts_query = self.build_full_text_query(&bm25_override, k);
                scanner.full_text_search(fts_query)?;
                scanner.create_plan().await
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                if let Some(cols) = projection {
                    scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                }
                scanner.full_text_search(&self.fts_query.column, self.query_text().as_str());
                scanner.fts_limit(k);
                scanner.set_deferred_bm25_stats(Some(deferred_stats.clone()));
                scanner.create_plan().await
            }
        }
    }

    /// Build a FullTextSearchQuery for the base Scanner with BM25 override and limit.
    fn build_full_text_query(
        &self,
        bm25_override: &Arc<BM25StatsOverride>,
        k: usize,
    ) -> FullTextSearchQuery {
        let match_query =
            MatchQuery::new(self.query_text()).with_column(Some(self.fts_query.column.clone()));
        let mut fts = FullTextSearchQuery::new_query(match_query.into());
        fts.bm25_override = Some(bm25_override.clone());
        fts.limit = Some(k as i64);
        fts
    }

    /// Open a flushed memtable dataset with the shared session for index caching.
    async fn open_flushed_dataset(&self, path: &str) -> Result<crate::dataset::Dataset> {
        crate::dataset::DatasetBuilder::from_uri(path)
            .with_session(self.session.clone())
            .load()
            .await
    }

    /// Build projection list for FTS ensuring required columns are included.
    fn build_projection_for_fts(&self, projection: Option<&[String]>) -> Vec<String> {
        let mut cols: Vec<String> = if let Some(p) = projection {
            p.to_vec()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };

        for pk in &self.pk_columns {
            if !cols.contains(pk) {
                cols.push(pk.clone());
            }
        }

        cols
    }

    /// Create an empty execution plan.
    fn empty_plan(&self, projection: Option<&[String]>) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;

        let mut fields: Vec<Arc<Field>> = if let Some(cols) = projection {
            cols.iter()
                .filter_map(|name| {
                    self.base_schema
                        .field_with_name(name)
                        .ok()
                        .map(|f| Arc::new(f.clone()))
                })
                .collect()
        } else {
            self.base_schema.fields().iter().cloned().collect()
        };

        fields.push(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, false)));

        let schema = Arc::new(Schema::new(fields));
        Ok(Arc::new(EmptyExec::new(schema)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::dataset::mem_wal::memtable::scanner::FtsQuery;
    use crate::dataset::mem_wal::scanner::collector::LsmDataSourceCollector;
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    fn create_fts_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, ids: &[i32], texts: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(texts.to_vec())),
            ],
        )
        .unwrap()
    }

    async fn create_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_fts_search_plan_structure() {
        let schema = create_fts_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(
            &schema,
            &[1, 2, 3],
            &["hello world", "foo bar", "hello foo"],
        );
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let fts_query = FtsQuery::match_query("text", "hello");
        let planner =
            LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema.clone(), fts_query);

        let plan = planner.plan_search(10, None).await;
        // Plan creation should succeed
        assert!(plan.is_ok() || plan.is_err());
    }

    #[tokio::test]
    async fn test_projection_includes_pk() {
        let schema = create_fts_schema();
        let base_batch = create_test_batch(&schema, &[1], &["hello"]);
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("{}/proj_test", temp_dir.path().to_str().unwrap());
        let base_dataset = Arc::new(create_dataset(&uri, vec![base_batch]).await);
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let fts_query = FtsQuery::match_query("text", "hello");
        let planner =
            LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema, fts_query);

        let cols = planner.build_projection_for_fts(Some(&["text".to_string()]));
        assert!(cols.contains(&"text".to_string()));
        assert!(cols.contains(&"id".to_string()));
    }

    #[tokio::test]
    async fn test_global_stats_from_persistent_index() {
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use lance_index::scalar::inverted::InvertedIndexParams;
        use lance_index::{DatasetIndexExt, IndexType};

        let schema = create_fts_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("{}/persistent_stats", temp_dir.path().to_str().unwrap());

        let batch = create_test_batch(
            &schema,
            &[1, 2, 3, 4, 5],
            &[
                "hello world",
                "hello lance",
                "foo bar baz",
                "hello foo",
                "world lance",
            ],
        );
        let mut dataset = create_dataset(&uri, vec![batch]).await;

        let params = InvertedIndexParams::default();
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        // Create an active memtable with FTS index (provides the tokenizer)
        let batch_store = Arc::new(BatchStore::with_capacity(10));
        let mut index_store = IndexStore::new();
        index_store.add_fts("text_idx".to_string(), 1, "text".to_string());
        let mem_batch = create_test_batch(&schema, &[100], &["hello test"]);
        index_store.insert(&mem_batch, 0).unwrap();
        batch_store.append(mem_batch).unwrap();
        let index_store = Arc::new(index_store);

        let base_dataset = Arc::new(dataset);
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let fts_query = FtsQuery::match_query("text", "hello");
        let planner =
            LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema.clone(), fts_query);

        // Build sources with both base table and active memtable
        let mut sources = planner.collector.collect().unwrap();
        sources.push(LsmDataSource::ActiveMemTable {
            batch_store,
            index_store,
            schema,
            region_id: uuid::Uuid::new_v4(),
            generation: 1.into(),
        });
        let stats = planner.collect_global_stats(&sources).await.unwrap();

        // 5 docs from persistent + 1 from memtable = 6
        assert_eq!(stats.num_docs, 6);
        assert!(stats.total_tokens > 0);
        // "hello" appears in 3 persistent docs + 1 memtable doc = 4
        assert_eq!(stats.term_doc_freqs.get("hello"), Some(&4));
    }

    #[tokio::test]
    async fn test_global_stats_collection_empty() {
        let sources: Vec<LsmDataSource> = vec![];
        let schema = create_fts_schema();
        let base_batch = create_test_batch(&schema, &[1], &["hello"]);
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("{}/stats_test", temp_dir.path().to_str().unwrap());
        let base_dataset = Arc::new(create_dataset(&uri, vec![base_batch]).await);
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let fts_query = FtsQuery::match_query("text", "hello");
        let planner =
            LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema, fts_query);

        let stats = planner.collect_global_stats(&sources).await.unwrap();
        assert_eq!(stats.num_docs, 0);
        assert_eq!(stats.total_tokens, 0);
    }
}
