// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deferred-load wrapper around [`DocSet`].
//!
//! The inverted-index `DocSet` holds the per-doc `row_id` and `num_tokens`
//! arrays for a partition. Eager loading on partition open pulls roughly
//! 12 bytes × num_docs per partition; across thousands of partitions on
//! cold object storage that's tens of GiB of IO before a query has even
//! checked whether a partition contains the term it's looking for.
//!
//! [`LazyDocSet`] defers the load. Cheap sync getters (`len`,
//! `total_tokens_cached`) work without IO; async getters fetch on
//! demand and cache. Wand scoring still needs per-doc num_tokens, but
//! only partitions that actually contribute hits pay
//! `ensure_num_tokens_loaded`/`ensure_loaded`.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{UInt32Type, UInt64Type};
use arrow_array::{Array, UInt32Array, UInt64Array};
use lance_core::ROW_ID;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use tokio::sync::OnceCell;

use crate::scalar::RowIdRemapper;
use crate::scalar::inverted::index::{DocSet, NUM_TOKEN_COL};
use crate::scalar::{IndexReader, IndexStore};
use lance_select::mask::RowAddrMask;

/// Lazy view over an inverted-index partition's `DocSet`.
///
/// Two variants:
/// - `Loaded`: a pre-materialized DocSet (legacy paths, tests).
///   Sync accessors return cached values; async accessors return
///   the same DocSet.
/// - `Deferred`: backed by an [`IndexReader`]; columns are read and
///   cached on first request.
pub enum LazyDocSet {
    Loaded(LoadedDocSet),
    Deferred(Box<DeferredDocSet>),
}

/// Pre-materialized DocSet view -- no reader, no IO.
pub struct LoadedDocSet {
    docs: Arc<DocSet>,
    num_rows: usize,
    total_tokens: u64,
}

/// Atomically published num-tokens state for deferred scoring.
///
/// Keeping the Arrow column and the zero-copy `DocSet` view that carries its
/// total in one `OnceCell` prevents cancellation from exposing a partially
/// initialized scoring cache.
struct NumTokensSnapshot {
    column: Arc<UInt32Array>,
    docs: Arc<DocSet>,
}

/// Store-backed DocSet view that loads on demand and caches.
///
/// Holds the [`IndexStore`] and docs-file path rather than an open
/// [`IndexReader`], so a cached partition does not pin a docs-file
/// handle for its whole lifetime. The reader is re-opened on demand
/// inside each column accessor and dropped when that read completes;
/// because the resulting buffers are cached in the `OnceCell`s below,
/// a contributing partition re-opens only on a cold miss, and a
/// partition that never scores never opens the docs file at all after
/// construction.
pub struct DeferredDocSet {
    store: Arc<dyn IndexStore>,
    docs_path: String,
    is_legacy: bool,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    /// 256-document-block partitions score with quantized document lengths; the
    /// flag is applied to every `DocSet` this deferred set materializes.
    quantized_scoring: bool,
    /// Doc count cached at construction so `len()` stays sync + IO-free.
    num_rows: usize,
    /// `NUM_TOKEN_COL` and its zero-copy scoring view carrying the cached sum,
    /// published together on first read.
    num_tokens: OnceCell<NumTokensSnapshot>,
    /// `ROW_ID` arrow buffer cached on first read.
    row_ids_col: OnceCell<Arc<UInt64Array>>,
    /// Full DocSet, materialized on first `ensure_loaded`.
    full: OnceCell<Arc<DocSet>>,
}

impl std::fmt::Debug for LazyDocSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded(l) => f
                .debug_struct("LazyDocSet::Loaded")
                .field("num_rows", &l.num_rows)
                .field("total_tokens", &l.total_tokens)
                .finish(),
            Self::Deferred(d) => f
                .debug_struct("LazyDocSet::Deferred")
                .field("num_rows", &d.num_rows)
                .field(
                    "total_tokens_loaded",
                    &(d.num_tokens.initialized() || d.full.initialized()),
                )
                .field("num_tokens_loaded", &d.num_tokens.initialized())
                .field("full_loaded", &d.full.initialized())
                .finish(),
        }
    }
}

impl DeepSizeOf for LazyDocSet {
    fn deep_size_of_children(&self, ctx: &mut lance_core::deepsize::Context) -> usize {
        match self {
            Self::Loaded(l) => l.docs.deep_size_of_children(ctx),
            Self::Deferred(d) => {
                d.full
                    .get()
                    .map(|d| d.deep_size_of_children(ctx))
                    .unwrap_or(0)
                    + d.num_tokens
                        .get()
                        .map(|snapshot| {
                            let arr: &dyn Array = snapshot.column.as_ref();
                            snapshot.docs.deep_size_of_children(ctx)
                                + arr.deep_size_of_children(ctx)
                        })
                        .unwrap_or(0)
                    + d.row_ids_col
                        .get()
                        .map(|arr| {
                            let arr: &dyn Array = arr.as_ref();
                            arr.deep_size_of_children(ctx)
                        })
                        .unwrap_or(0)
            }
        }
    }
}

impl LazyDocSet {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<dyn IndexStore>,
        docs_path: String,
        num_rows: usize,
        is_legacy: bool,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        quantized_scoring: bool,
    ) -> Self {
        Self::Deferred(Box::new(DeferredDocSet {
            store,
            docs_path,
            is_legacy,
            frag_reuse_index,
            quantized_scoring,
            num_rows,
            num_tokens: OnceCell::new(),
            row_ids_col: OnceCell::new(),
            full: OnceCell::new(),
        }))
    }

    /// Wrap an already-materialized [`DocSet`]. Used by legacy paths
    /// and tests that need to seed a partition without a reader.
    pub fn from_loaded(docs: DocSet) -> Self {
        let num_rows = docs.len();
        let total_tokens = docs.total_tokens_num();
        Self::Loaded(LoadedDocSet {
            docs: Arc::new(docs),
            num_rows,
            total_tokens,
        })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Loaded(l) => l.num_rows,
            Self::Deferred(d) => d.num_rows,
        }
    }

    /// Sync read of cached `total_tokens`. Returns `None` for a
    /// `Deferred` LazyDocSet that hasn't yet had any of
    /// `total_tokens_num` / `ensure_num_tokens_loaded` / `ensure_loaded`
    /// run. Used by sync scoring code that has already paid for one
    /// of those async calls.
    pub fn total_tokens_cached(&self) -> Option<u64> {
        match self {
            Self::Loaded(l) => Some(l.total_tokens),
            Self::Deferred(d) => d
                .full
                .get()
                .map(|docs| docs.total_tokens_num())
                .or_else(|| {
                    d.num_tokens
                        .get()
                        .map(|snapshot| snapshot.docs.total_tokens_num())
                }),
        }
    }

    /// True if this DocSet carries a FragReuseIndex. Callers MUST
    /// avoid the deferred-row_id path when this is set: targeted
    /// row_id reads return raw stored ids, bypassing the per-id
    /// `remap_row_id` filter that `DocSet::from_columns` applies.
    pub fn has_frag_reuse_remap(&self) -> bool {
        match self {
            Self::Loaded(_) => false,
            Self::Deferred(d) => d.frag_reuse_index.is_some(),
        }
    }

    /// Sum of `num_tokens` across all docs.
    pub async fn total_tokens_num(&self) -> Result<u64> {
        match self {
            Self::Loaded(l) => Ok(l.total_tokens),
            Self::Deferred(d) => d.total_tokens_num().await,
        }
    }

    /// Materialize the full DocSet, including row_ids.
    pub async fn ensure_loaded(&self) -> Result<Arc<DocSet>> {
        match self {
            Self::Loaded(l) => Ok(l.docs.clone()),
            Self::Deferred(d) => d.ensure_loaded().await,
        }
    }

    /// Materialize a DocSet that carries num_tokens but no row_ids.
    /// Used by the deferred-row_id scoring path; the per-partition
    /// caller resolves surviving doc_ids -> row_ids post-wand via
    /// [`Self::resolve_row_ids`]. The tokens-only result is cached separately;
    /// a later `ensure_loaded` must still produce a full DocSet.
    pub async fn ensure_num_tokens_loaded(&self) -> Result<Arc<DocSet>> {
        match self {
            Self::Loaded(l) => Ok(l.docs.clone()),
            Self::Deferred(d) => d.ensure_num_tokens_loaded().await,
        }
    }

    /// Pick the right DocSet shape for a wand walk under `mask`:
    /// the num_tokens-only deferred form when the mask is trivial
    /// AND no FragReuseIndex needs to filter row_ids; otherwise the
    /// full DocSet. Encapsulates the policy so callers don't have to
    /// rederive the conditions for the targeted-read fast path.
    pub async fn docs_for_wand(&self, mask: &RowAddrMask) -> Result<Arc<DocSet>> {
        if mask.is_select_all() && !self.has_frag_reuse_remap() {
            self.ensure_num_tokens_loaded().await
        } else {
            self.ensure_loaded().await
        }
    }

    /// Resolve a batch of `doc_id`s to their `row_id`s. Used by the
    /// deferred-row_id scoring path to map post-wand top-K candidates
    /// without going through a full DocSet build.
    ///
    /// Not safe with a FragReuseIndex (see
    /// [`Self::has_frag_reuse_remap`]): the targeted reads return
    /// raw stored ids without applying the remap/skip.
    pub async fn resolve_row_ids(&self, doc_ids: &[u32]) -> Result<Vec<u64>> {
        match self {
            Self::Loaded(l) => Ok(doc_ids.iter().map(|&d| l.docs.row_id(d)).collect()),
            Self::Deferred(d) => d.resolve_row_ids(doc_ids).await,
        }
    }
}

impl DeferredDocSet {
    /// Open a fresh docs-file reader. Dropped by the caller once its read
    /// completes, so no handle is pinned across the partition's lifetime.
    async fn reader(&self) -> Result<Arc<dyn IndexReader>> {
        self.store.open_index_file(&self.docs_path).await
    }

    async fn total_tokens_num(&self) -> Result<u64> {
        if let Some(full) = self.full.get() {
            return Ok(full.total_tokens_num());
        }
        Ok(self.num_tokens_snapshot().await?.docs.total_tokens_num())
    }

    async fn num_tokens_snapshot(&self) -> Result<&NumTokensSnapshot> {
        self.num_tokens
            .get_or_try_init(|| async {
                let reader = self.reader().await?;
                let batch = reader
                    .read_range(0..self.num_rows, Some(&[NUM_TOKEN_COL]))
                    .await?;
                let column = Arc::new(batch[NUM_TOKEN_COL].as_primitive::<UInt32Type>().clone());
                let total_tokens = column.values().iter().map(|&n| n as u64).sum();
                let mut docs = DocSet::from_cached_num_tokens(column.as_ref(), total_tokens);
                docs.set_quantized_scoring(self.quantized_scoring);
                Result::Ok(NumTokensSnapshot {
                    column,
                    docs: Arc::new(docs),
                })
            })
            .await
    }

    async fn row_ids_column(&self) -> Result<Arc<UInt64Array>> {
        self.row_ids_col
            .get_or_try_init(|| async {
                let reader = self.reader().await?;
                let batch = reader.read_range(0..self.num_rows, Some(&[ROW_ID])).await?;
                Result::Ok(Arc::new(batch[ROW_ID].as_primitive::<UInt64Type>().clone()))
            })
            .await
            .cloned()
    }

    async fn ensure_loaded(&self) -> Result<Arc<DocSet>> {
        let docs = self
            .full
            .get_or_try_init(|| async {
                // If the stats path already pulled NUM_TOKEN_COL,
                // read only ROW_ID and rebuild from the two columns.
                let mut docs = if let Some(num_tokens) = self.num_tokens.get() {
                    let row_ids = self.row_ids_column().await?;
                    DocSet::from_columns(
                        row_ids.as_ref(),
                        num_tokens.column.as_ref(),
                        self.is_legacy,
                        self.frag_reuse_index.clone(),
                    )?
                } else {
                    DocSet::load(
                        self.reader().await?,
                        self.is_legacy,
                        self.frag_reuse_index.clone(),
                    )
                    .await?
                };
                docs.set_quantized_scoring(self.quantized_scoring);
                Result::Ok(Arc::new(docs))
            })
            .await?
            .clone();
        Ok(docs)
    }

    async fn ensure_num_tokens_loaded(&self) -> Result<Arc<DocSet>> {
        if let Some(full) = self.full.get() {
            return Ok(full.clone());
        }
        Ok(self.num_tokens_snapshot().await?.docs.clone())
    }

    async fn resolve_row_ids(&self, doc_ids: &[u32]) -> Result<Vec<u64>> {
        if let Some(full) = self.full.get()
            && full.has_row_ids()
        {
            return Ok(doc_ids.iter().map(|&d| full.row_id(d)).collect());
        }
        if let Some(arr) = self.row_ids_col.get() {
            return Ok(doc_ids.iter().map(|&d| arr.value(d as usize)).collect());
        }
        let ranges: Vec<std::ops::Range<usize>> = doc_ids
            .iter()
            .map(|&d| d as usize..d as usize + 1)
            .collect();
        let reader = self.reader().await?;
        let batch = reader.read_ranges(&ranges, Some(&[ROW_ID])).await?;
        let arr = batch[ROW_ID].as_primitive::<UInt64Type>();
        Ok((0..arr.len()).map(|i| arr.value(i)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::lance_format::LanceIndexStore;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;

    #[tokio::test]
    async fn test_full_docset_is_a_complete_cached_snapshot() {
        let temp_dir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            temp_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let docs = LazyDocSet::new(store, "unused".to_owned(), 3, false, None, false);
        assert_eq!(docs.total_tokens_cached(), None);

        let row_ids = UInt64Array::from(vec![10, 20, 30]);
        let num_tokens = UInt32Array::from(vec![3, 5, 8]);
        let full = Arc::new(DocSet::from_columns(&row_ids, &num_tokens, false, None).unwrap());
        let LazyDocSet::Deferred(deferred) = &docs else {
            panic!("expected a deferred DocSet");
        };
        deferred.full.set(full.clone()).unwrap();

        let wand_docs = docs.ensure_num_tokens_loaded().await.unwrap();
        assert!(Arc::ptr_eq(&wand_docs, &full));
        assert_eq!(wand_docs.total_tokens_num(), 16);
        assert_eq!(docs.total_tokens_cached(), Some(16));
    }
}
