// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deferred-load wrapper around [`DocSet`].
//!
//! The inverted-index `DocSet` holds the per-doc `row_id` and `num_tokens`
//! arrays for a partition. Today every partition opens with the full set
//! materialized — roughly 12 bytes × num_docs (~10 MiB per partition on
//! large indexes). Across thousands of partitions and cold object storage
//! that's tens of GiB of IO pulled before a query knows whether any
//! particular partition even contains the term it's looking for.
//!
//! [`LazyDocSet`] defers the load. Construction only stashes the reader and
//! reads `num_rows` (no IO). The cheap, scoring-irrelevant queries the
//! stats path needs — `len`, `total_tokens` — go through async accessors
//! that compute on demand and cache. Wand scoring still requires the full
//! `DocSet`, so the caller pays `ensure_loaded` only for partitions that
//! actually contribute hits.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{UInt32Type, UInt64Type};
use arrow_array::{UInt32Array, UInt64Array};
use lance_core::Result;
use tokio::sync::OnceCell;

use lance_core::ROW_ID;

use crate::frag_reuse::FragReuseIndex;
use crate::scalar::IndexReader;
use crate::scalar::inverted::index::{DocSet, NUM_TOKEN_COL};

/// Lazy view over an inverted-index partition's `DocSet`.
///
/// All sync getters work without IO; async getters fetch on demand and
/// cache. Methods that need the full per-doc arrays (`row_id`,
/// `num_tokens`, iteration) require an explicit [`Self::ensure_loaded`]
/// first.
pub struct LazyDocSet {
    reader: Arc<dyn IndexReader>,
    is_legacy: bool,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
    /// `reader.num_rows()` cached at construction. Equivalent to the eager
    /// `DocSet::len`.
    num_rows: usize,
    /// `sum(num_tokens)` cached on first request, either from the
    /// fully-loaded DocSet or from a single-column read.
    total_tokens: OnceCell<u64>,
    /// `NUM_TOKEN_COL` arrow buffer cached the first time it's read (by
    /// either `total_tokens_num` or `ensure_loaded`). Reusing it on the
    /// scoring path avoids re-reading the same column for hit partitions
    /// after the stats path already pulled it.
    num_tokens_col: OnceCell<Arc<UInt32Array>>,
    /// Full DocSet, materialized lazily when scoring needs per-doc
    /// `row_id`/`num_tokens`.
    full: OnceCell<Arc<DocSet>>,
}

impl std::fmt::Debug for LazyDocSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyDocSet")
            .field("num_rows", &self.num_rows)
            .field("total_tokens_loaded", &self.total_tokens.initialized())
            .field("full_loaded", &self.full.initialized())
            .finish()
    }
}

impl deepsize::DeepSizeOf for LazyDocSet {
    fn deep_size_of_children(&self, ctx: &mut deepsize::Context) -> usize {
        // Approximate: only account for the fully-loaded DocSet and the
        // cached num_tokens column when present.
        self.full
            .get()
            .map(|d| d.deep_size_of_children(ctx))
            .unwrap_or(0)
            + self
                .num_tokens_col
                .get()
                .map(|arr| arr.len() * std::mem::size_of::<u32>())
                .unwrap_or(0)
    }
}

impl LazyDocSet {
    pub fn new(
        reader: Arc<dyn IndexReader>,
        is_legacy: bool,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        let num_rows = reader.num_rows();
        Self {
            reader,
            is_legacy,
            frag_reuse_index,
            num_rows,
            total_tokens: OnceCell::new(),
            num_tokens_col: OnceCell::new(),
            full: OnceCell::new(),
        }
    }

    /// Wrap an already-materialized [`DocSet`]. Useful for legacy paths and
    /// tests that need to seed a partition without an underlying reader.
    pub fn from_loaded(docs: DocSet) -> Self {
        // Build a synthetic LazyDocSet whose `full` cell is pre-populated.
        // The reader/frag-reuse fields will never be touched.
        let num_rows = docs.len();
        let total_tokens = docs.total_tokens_num();
        let me = Self {
            reader: panic_reader(),
            is_legacy: false,
            frag_reuse_index: None,
            num_rows,
            total_tokens: OnceCell::new(),
            num_tokens_col: OnceCell::new(),
            full: OnceCell::new(),
        };
        let _ = me.total_tokens.set(total_tokens);
        let _ = me.full.set(Arc::new(docs));
        me
    }

    /// Number of docs in the partition (cheap, no IO).
    pub fn len(&self) -> usize {
        self.num_rows
    }

    /// Returns the full [`DocSet`] if it has already been loaded; otherwise
    /// `None`. Sync; callers that need a guaranteed value should call
    /// [`Self::ensure_loaded`] first.
    pub fn loaded(&self) -> Option<&Arc<DocSet>> {
        self.full.get()
    }

    /// Sum of `num_tokens` across all docs. Lazy-computed on first call:
    /// if the full DocSet is loaded, reads from it; otherwise issues a
    /// single-column read of `NUM_TOKEN_COL` from the docs file (about
    /// half the bytes of a full DocSet load). Caches both the sum and
    /// the per-row arrow buffer so a subsequent `ensure_loaded` can
    /// reuse the buffer instead of re-reading.
    pub async fn total_tokens_num(&self) -> Result<u64> {
        if let Some(v) = self.total_tokens.get() {
            return Ok(*v);
        }
        if let Some(full) = self.full.get() {
            let v = full.total_tokens_num();
            let _ = self.total_tokens.set(v);
            return Ok(v);
        }
        let col = self.read_num_tokens_column().await?;
        let sum: u64 = col.values().iter().map(|&n| n as u64).sum();
        let _ = self.total_tokens.set(sum);
        Ok(sum)
    }

    /// Internal helper: read (or return cached) `NUM_TOKEN_COL`.
    async fn read_num_tokens_column(&self) -> Result<Arc<UInt32Array>> {
        if let Some(arr) = self.num_tokens_col.get() {
            return Ok(arr.clone());
        }
        let batch = self
            .reader
            .read_range(0..self.num_rows, Some(&[NUM_TOKEN_COL]))
            .await?;
        let arr = Arc::new(batch[NUM_TOKEN_COL].as_primitive::<UInt32Type>().clone());
        // `set` errors if another caller raced; their value is equivalent.
        let _ = self.num_tokens_col.set(arr.clone());
        Ok(self.num_tokens_col.get().unwrap().clone())
    }

    /// Materialize the full [`DocSet`] and cache it. If a prior
    /// `total_tokens_num` already pulled `NUM_TOKEN_COL`, we read only
    /// `ROW_ID` here and rebuild the DocSet from the two columns —
    /// halving the bytes pulled for hit partitions whose stats path
    /// already paid the num_tokens read.
    pub async fn ensure_loaded(&self) -> Result<Arc<DocSet>> {
        if let Some(full) = self.full.get() {
            return Ok(full.clone());
        }
        let docs = if self.num_tokens_col.get().is_some() {
            let num_tokens = self.read_num_tokens_column().await?;
            let batch = self
                .reader
                .read_range(0..self.num_rows, Some(&[ROW_ID]))
                .await?;
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            DocSet::from_columns(
                row_ids,
                num_tokens.as_ref(),
                self.is_legacy,
                self.frag_reuse_index.clone(),
            )?
        } else {
            // Cold path: nothing cached yet, fall back to the full read.
            DocSet::load(
                self.reader.clone(),
                self.is_legacy,
                self.frag_reuse_index.clone(),
            )
            .await?
        };
        let docs = Arc::new(docs);
        let _ = self.full.set(docs.clone());
        let _ = self.total_tokens.set(docs.total_tokens_num());
        Ok(self.full.get().unwrap().clone())
    }
}

/// Sentinel reader used by [`LazyDocSet::from_loaded`]; the LazyDocSet's
/// IO paths never touch it because `total_tokens` and `full` are
/// pre-populated.
fn panic_reader() -> Arc<dyn IndexReader> {
    struct Panic;
    #[async_trait::async_trait]
    impl IndexReader for Panic {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn read_record_batch(
            &self,
            _n: u64,
            _batch_size: u64,
        ) -> Result<arrow_array::RecordBatch> {
            panic!("synthetic LazyDocSet reader should never be queried")
        }
        async fn read_range(
            &self,
            _range: std::ops::Range<usize>,
            _projection: Option<&[&str]>,
        ) -> Result<arrow_array::RecordBatch> {
            panic!("synthetic LazyDocSet reader should never be queried")
        }
        async fn num_batches(&self, _batch_size: u64) -> u32 {
            0
        }
        fn num_rows(&self) -> usize {
            0
        }
        fn schema(&self) -> &lance_core::datatypes::Schema {
            panic!("synthetic LazyDocSet reader should never be queried")
        }
    }
    Arc::new(Panic)
}
