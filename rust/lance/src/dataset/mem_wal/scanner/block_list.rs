// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-source block-list construction for LSM vector search.
//!
//! Given the LSM data sources for a query, this module builds the
//! [`GenPkIndex`] membership for each generation and assembles the per-source
//! block-list bitmap (a [`RowAddrMask`]) that suppresses rows superseded by a
//! newer generation. The bitmaps are produced here but *not* yet wired into KNN
//! execution — how the mask drives the search is decided separately.
//!
//! Each mask is keyed by the **`_rowid`** the source's KNN emits, so a candidate
//! can be matched against it (it is consumed only by its own source's search):
//! - active / frozen memtables: the memtable row position (`row_offset + row`),
//!   which is exactly the memtable `_rowid`;
//! - flushed generations / base table: the dataset `_rowid` that `fast_search`
//!   emits (equal to `_rowaddr` unless stable row ids are enabled).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt64Array};
use futures::TryStreamExt;
use lance_core::utils::mask::{RowAddrMask, RowAddrTreeMap};
use lance_core::{ROW_ID, Result};

use super::data_source::{LsmDataSource, LsmGeneration};
use super::exec::{FreshnessPolarity, compute_pk_hash, resolve_pk_indices};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use super::gen_pk_index::{GenPkIndex, compute_block_lists};
use crate::dataset::Dataset;
use crate::dataset::mem_wal::write::BatchStore;
use crate::session::Session;

/// Build a [`GenPkIndex`] for an in-memory memtable (active or frozen) from its
/// committed `BatchStore` rows.
///
/// The row address is the memtable row position (`row_offset + row_index`),
/// which is monotonic with insert order — the same address a flat filtered scan
/// over the memtable would skip, and the value an insert-order freshness
/// comparison ranks (larger = newer).
pub fn pk_index_from_batch_store(store: &BatchStore, pk_columns: &[String]) -> Result<GenPkIndex> {
    let len = store.len();
    let mut batches: Vec<RecordBatch> = Vec::with_capacity(len);
    let mut row_offsets: Vec<u64> = Vec::with_capacity(len);
    for i in 0..len {
        if let Some(stored) = store.get(i) {
            row_offsets.push(stored.row_offset);
            batches.push(stored.data.clone());
        }
    }
    GenPkIndex::from_batches(&batches, pk_columns, |batch_idx, row_idx| {
        row_offsets[batch_idx] + row_idx as u64
    })
}

/// Compute the per-source block-list bitmap for every LSM source in `sources`.
///
/// Each returned [`RowAddrMask`] blocks the rows of one generation that are
/// superseded by a newer generation (and, for flushed / in-memory sources, rows
/// superseded *within* their own generation). The base table is blocked only
/// cross-generation: it has no [`GenPkIndex`] — within-base duplicates are not
/// addressed (see the design's Open items). Bitmaps are keyed by generation;
/// how a mask drives the actual KNN search is decided separately.
///
/// Row-address conventions are per-source and never compared across sources:
/// memtable positions for active/frozen, `_rowid` for flushed/base.
pub async fn compute_source_block_lists(
    sources: &[LsmDataSource],
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<HashMap<LsmGeneration, Arc<RowAddrMask>>> {
    // Build a GenPkIndex for every non-base source (flushed + active/frozen).
    let mut indexed: Vec<(LsmGeneration, FreshnessPolarity, GenPkIndex)> = Vec::new();
    let mut base: Option<&Arc<Dataset>> = None;
    for source in sources {
        match source {
            LsmDataSource::BaseTable { dataset } => base = Some(dataset),
            LsmDataSource::ActiveMemTable {
                batch_store,
                generation,
                ..
            } => {
                // Active and frozen memtables are insert-ordered (newest = largest position).
                let index = pk_index_from_batch_store(batch_store, pk_columns)?;
                indexed.push((*generation, FreshnessPolarity::InsertOrder, index));
            }
            LsmDataSource::FlushedMemTable {
                path, generation, ..
            } => {
                // Flushed generations are reverse-written (newest = smallest `_rowid`).
                let dataset = open_flushed_dataset(path, session, flushed_cache).await?;
                let batches = scan_pk_rowid(&dataset, pk_columns).await?;
                let index = pk_index_from_scanned(&batches, pk_columns)?;
                indexed.push((*generation, FreshnessPolarity::ReverseWrite, index));
            }
        }
    }

    // Newest generation first so each older generation is blocked against the
    // union of every newer generation's membership (`NEWER(G)`).
    indexed.sort_by_key(|(generation, _, _)| std::cmp::Reverse(*generation));
    let gens_newest_first: Vec<(FreshnessPolarity, &GenPkIndex)> = indexed
        .iter()
        .map(|(_, polarity, index)| (*polarity, index))
        .collect();
    let (block_trees, membership) = compute_block_lists(&gens_newest_first);

    // Keep only generations that actually block a row, so a caller can treat a
    // map entry as "this source needs filtering" and skip the cost otherwise.
    let mut block_lists: HashMap<LsmGeneration, Arc<RowAddrMask>> = HashMap::new();
    for ((generation, _, _), tree) in indexed.iter().zip(block_trees) {
        if !tree_is_empty(&tree) {
            block_lists.insert(*generation, Arc::new(RowAddrMask::from_block(tree)));
        }
    }

    // Base (generation 0): block only rows whose PK has a newer version anywhere.
    if let Some(dataset) = base
        && !membership.is_empty()
    {
        let batches = scan_pk_rowid(dataset, pk_columns).await?;
        let tree = base_superseded_rowids(&batches, pk_columns, &membership)?;
        if !tree_is_empty(&tree) {
            block_lists.insert(
                LsmGeneration::BASE_TABLE,
                Arc::new(RowAddrMask::from_block(tree)),
            );
        }
    }

    Ok(block_lists)
}

/// Scan a dataset's PK columns plus `_rowid`, collecting the result batches.
async fn scan_pk_rowid(dataset: &Dataset, pk_columns: &[String]) -> Result<Vec<RecordBatch>> {
    let pk_refs: Vec<&str> = pk_columns.iter().map(String::as_str).collect();
    let mut scanner = dataset.scan();
    scanner.project(&pk_refs)?;
    scanner.with_row_id();
    let stream = scanner.try_into_stream().await?;
    stream.try_collect::<Vec<_>>().await
}

/// Build a [`GenPkIndex`] from disk-scanned `(pk columns, _rowid)` batches.
fn pk_index_from_scanned(batches: &[RecordBatch], pk_columns: &[String]) -> Result<GenPkIndex> {
    let rowids: Vec<&UInt64Array> = batches.iter().map(rowid_column).collect::<Result<_>>()?;
    GenPkIndex::from_batches(batches, pk_columns, |batch_idx, row_idx| {
        rowids[batch_idx].value(row_idx)
    })
}

/// Row addresses of base rows whose PK hash is in `membership` (i.e. has a newer
/// version in some later generation).
fn base_superseded_rowids(
    batches: &[RecordBatch],
    pk_columns: &[String],
    membership: &HashSet<u64>,
) -> Result<RowAddrTreeMap> {
    let mut blocked = RowAddrTreeMap::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let pk_indices = resolve_pk_indices(batch, pk_columns)
            .map_err(|e| lance_core::Error::invalid_input(e.to_string()))?;
        let rowids = rowid_column(batch)?;
        for row in 0..batch.num_rows() {
            if membership.contains(&compute_pk_hash(batch, &pk_indices, row)) {
                blocked.insert(rowids.value(row));
            }
        }
    }
    Ok(blocked)
}

/// Whether the block tree contains no row addresses. Our trees are built from
/// individual inserts (never whole-fragment blocks), so `row_addrs` is always
/// enumerable; a non-enumerable tree is conservatively treated as non-empty.
fn tree_is_empty(tree: &RowAddrTreeMap) -> bool {
    tree.row_addrs()
        .map(|mut addrs| addrs.next().is_none())
        .unwrap_or(false)
}

/// Extract the `_rowid` (UInt64) column added by `with_row_id`.
fn rowid_column(batch: &RecordBatch) -> Result<&UInt64Array> {
    batch
        .column_by_name(ROW_ID)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            lance_core::Error::internal(format!("scan result missing UInt64 `{ROW_ID}` column"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::scanner::exec::FreshnessPolarity;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::mask::RowAddrMask;
    use std::sync::Arc;

    fn id_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    #[test]
    fn batch_store_index_tracks_positions_and_within_gen_dups() {
        let store = BatchStore::with_capacity(8);
        // Two single-row batches, both pk=1 (a within-gen update): positions 0, 1.
        store.append(id_batch(&[1])).unwrap();
        store.append(id_batch(&[1])).unwrap();
        // A two-row batch starting at position 2: pk=2 @ 2, pk=3 @ 3.
        store.append(id_batch(&[2, 3])).unwrap();

        let index = pk_index_from_batch_store(&store, &["id".to_string()]).unwrap();
        assert_eq!(index.len(), 3); // distinct pks: 1, 2, 3

        // Insert order: the older pk=1 copy (position 0) is superseded by the
        // newer one (position 1); unique pks are never blocked.
        let block = index.within_gen_superseded(FreshnessPolarity::InsertOrder);
        let mask = RowAddrMask::from_block(block);
        assert!(!mask.selected(0)); // blocked (older pk=1)
        assert!(mask.selected(1)); // kept (newest pk=1)
        assert!(mask.selected(2)); // pk=2
        assert!(mask.selected(3)); // pk=3
    }

    #[tokio::test]
    async fn block_lists_suppress_stale_across_in_memory_gens() {
        use crate::dataset::mem_wal::scanner::data_source::{LsmDataSource, LsmGeneration};
        use crate::dataset::mem_wal::write::IndexStore;
        use uuid::Uuid;

        let shard = Uuid::new_v4();
        let mk = |batches: &[&[i32]], generation: u64| {
            let store = BatchStore::with_capacity(8);
            for ids in batches {
                store.append(id_batch(ids)).unwrap();
            }
            LsmDataSource::ActiveMemTable {
                batch_store: Arc::new(store),
                index_store: Arc::new(IndexStore::new()),
                schema: id_batch(&[1]).schema(),
                shard_id: shard,
                generation: LsmGeneration::memtable(generation),
            }
        };

        // Frozen gen 1: stale pk=1 @ position 0.
        // Active gen 2: pk=1 re-written @ 0, pk=2 @ 1.
        let sources = vec![mk(&[&[1]], 1), mk(&[&[1], &[2]], 2)];

        let masks = compute_source_block_lists(&sources, &["id".to_string()], None, None)
            .await
            .unwrap();

        let g1 = LsmGeneration::memtable(1);
        let g2 = LsmGeneration::memtable(2);
        // The newer active write supersedes the frozen copy: gen 1's pk=1 is blocked.
        assert!(!masks[&g1].selected(0));
        // The active (newest) generation blocks nothing, so it has no mask entry.
        assert!(!masks.contains_key(&g2));
    }

    #[tokio::test]
    async fn block_lists_suppress_stale_base_row() {
        use crate::dataset::mem_wal::scanner::data_source::{LsmDataSource, LsmGeneration};
        use crate::dataset::mem_wal::write::IndexStore;
        use crate::dataset::{Dataset, WriteParams};
        use arrow_array::RecordBatchIterator;
        use uuid::Uuid;

        // Base (gen 0): pk=1 @ _rowid 0 (stale), pk=3 @ _rowid 1 (live).
        let base_batch = id_batch(&[1, 3]);
        let schema = base_batch.schema();
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let reader = RecordBatchIterator::new(vec![Ok(base_batch)], schema.clone());
        let base = Arc::new(
            Dataset::write(reader, &uri, Some(WriteParams::default()))
                .await
                .unwrap(),
        );

        // Active gen 1: pk=1 re-written, pk=2 new.
        let store = BatchStore::with_capacity(8);
        store.append(id_batch(&[1])).unwrap();
        store.append(id_batch(&[2])).unwrap();

        let sources = vec![
            LsmDataSource::BaseTable { dataset: base },
            LsmDataSource::ActiveMemTable {
                batch_store: Arc::new(store),
                index_store: Arc::new(IndexStore::new()),
                schema,
                shard_id: Uuid::new_v4(),
                generation: LsmGeneration::memtable(1),
            },
        ];

        let masks = compute_source_block_lists(&sources, &["id".to_string()], None, None)
            .await
            .unwrap();

        // Base's stale pk=1 (_rowid 0) is blocked; the unrelated live pk=3
        // (_rowid 1) survives — base is blocked cross-generation only.
        let base_mask = &masks[&LsmGeneration::BASE_TABLE];
        assert!(!base_mask.selected(0));
        assert!(base_mask.selected(1));
    }
}
