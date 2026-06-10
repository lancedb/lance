// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-source block-list construction for LSM vector search.
//!
//! A generation's membership is an `Arc<HashSet<u64>>` of PK hashes
//! ([`compute_pk_hash`]), built once (immutable gens cached). Each source gets a
//! `Vec<Arc<HashSet<u64>>>` of the newer generations' sets (`NEWER(G)`; base: all
//! of them) — referenced, never merged. The KNN drops candidates whose PK is in
//! any (see [`super::exec::PkHashFilterExec`]).
//!
//! Cross-generation only: within-gen dups share a hash and fall to the global
//! dedup's `(generation, freshness)` tiebreaker.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::TryStreamExt;
use lance_core::Result;

use uuid::Uuid;

use super::data_source::{AsOfCut, LsmDataSource, LsmGeneration};
use super::exec::{compute_pk_hash, resolve_pk_indices};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use crate::dataset::Dataset;
use crate::dataset::mem_wal::write::BatchStore;
use crate::session::Session;

/// Per-source blocked PK-hash sets, keyed by `(shard_id, generation)`. Each
/// value is the membership sets of the generations newer than that source.
pub type SourceBlockLists = HashMap<(Option<Uuid>, LsmGeneration), Vec<Arc<HashSet<u64>>>>;

/// A shard's generations paired with their PK-hash membership, before sorting.
type ShardGenSets = HashMap<Uuid, Vec<(LsmGeneration, Arc<HashSet<u64>>)>>;

/// Per-source `NEWER(G)`, keyed by `(shard_id, generation)`. Generations are
/// per-shard, so a source is superseded only by strictly-newer generations of
/// the **same** shard — it never appears in its own blocked list. The base table
/// is shardless (`None`, oldest) and superseded by every non-base generation.
/// Only superseded sources get an entry; the newest of each shard never does.
pub async fn compute_source_block_lists(
    sources: &[LsmDataSource],
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<SourceBlockLists> {
    // Hash each non-base source's membership, grouped by shard (generations are
    // per-shard, so supersession is within-shard only).
    let mut by_shard: ShardGenSets = HashMap::new();
    let mut has_base = false;
    for source in sources {
        match source {
            LsmDataSource::BaseTable { .. } => has_base = true,
            LsmDataSource::ActiveMemTable {
                batch_store,
                shard_id,
                generation,
                ..
            } => {
                let hashes = Arc::new(pk_hashes_from_batch_store(batch_store, pk_columns)?);
                by_shard
                    .entry(*shard_id)
                    .or_default()
                    .push((*generation, hashes));
            }
            LsmDataSource::FlushedMemTable {
                path,
                shard_id,
                generation,
                ..
            } => {
                // Cached by immutable path so repeated searches skip the PK scan.
                let hashes = flushed_pk_hashes(path, pk_columns, session, flushed_cache).await?;
                by_shard
                    .entry(*shard_id)
                    .or_default()
                    .push((*generation, hashes));
            }
        }
    }

    let mut blocked: SourceBlockLists = HashMap::new();
    // Base (shardless, oldest) is superseded by every non-base generation.
    let mut base_blocked: Vec<Arc<HashSet<u64>>> = Vec::new();
    for (shard, mut gens) in by_shard {
        // Newest-first: a gen's blocked list is its own shard's newer gens.
        gens.sort_by_key(|(generation, _)| std::cmp::Reverse(*generation));
        let mut newer: Vec<Arc<HashSet<u64>>> = Vec::new();
        for (generation, hashes) in gens {
            if !newer.is_empty() {
                blocked.insert((Some(shard), generation), newer.clone());
            }
            if !hashes.is_empty() {
                base_blocked.push(hashes.clone());
                newer.push(hashes);
            }
        }
    }
    if has_base && !base_blocked.is_empty() {
        blocked.insert((None, LsmGeneration::BASE_TABLE), base_blocked);
    }
    Ok(blocked)
}

/// The fresh-tier block-list: one membership set per generation that shadows the
/// base table — active + frozen memtables (hashed now) and flushed generations
/// (from the cache). Same `Vec<Arc<HashSet<u64>>>` shape the vector-search filter
/// consumes; a base/external reader can drop any row whose PK is in one of them.
/// The base source, if present, is skipped (it is what gets shadowed).
///
/// When `as_of` carries a cut for a source's shard, membership is bounded to
/// that snapshot (see [`AsOfCut`]): in-memory sources at a generation above
/// `active_generation` are excluded (they appeared after the snapshot); the
/// active generation is bounded to its first `active_batch_count` batches;
/// everything below the active generation (frozen and flushed) is immutable and
/// included whole. A shard absent from `as_of` (or `as_of == None`) uses the
/// live tier. The bound only excludes rows, so it can under-count (a tolerable
/// stale read) but never over-count.
pub async fn fresh_tier_block_list(
    sources: &[LsmDataSource],
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
    as_of: Option<&HashMap<Uuid, AsOfCut>>,
) -> Result<Vec<Arc<HashSet<u64>>>> {
    let mut sets = Vec::new();
    for source in sources {
        let set = match source {
            LsmDataSource::BaseTable { .. } => continue,
            LsmDataSource::ActiveMemTable {
                batch_store,
                shard_id,
                generation,
                ..
            } => match as_of.and_then(|m| m.get(shard_id)) {
                Some(cut) => {
                    let g = generation.as_u64();
                    if g > cut.active_generation {
                        // A newer in-memory generation that rolled in after the
                        // snapshot — not observed by the scan arm.
                        continue;
                    }
                    // The active generation grew between the arm and now; bound
                    // it to the batches the arm saw. Lower (frozen) generations
                    // are immutable, so all their batches were observed.
                    let limit = if g == cut.active_generation {
                        cut.active_batch_count as usize
                    } else {
                        usize::MAX
                    };
                    Arc::new(pk_hashes_from_batch_store_bounded(
                        batch_store,
                        pk_columns,
                        limit,
                    )?)
                }
                None => Arc::new(pk_hashes_from_batch_store(batch_store, pk_columns)?),
            },
            LsmDataSource::FlushedMemTable {
                path,
                shard_id,
                generation,
                ..
            } => {
                // A flushed generation at or above the active generation was
                // produced by a flush after the snapshot (and may hold rows the
                // arm never saw), so it must not contribute. Lower generations
                // are immutable and were fully observed.
                let flushed_after_snapshot = as_of
                    .and_then(|m| m.get(shard_id))
                    .is_some_and(|cut| generation.as_u64() >= cut.active_generation);
                if flushed_after_snapshot {
                    continue;
                }
                flushed_pk_hashes(path, pk_columns, session, flushed_cache).await?
            }
        };
        if !set.is_empty() {
            sets.push(set);
        }
    }
    Ok(sets)
}

/// Hash the PK membership of an in-memory memtable (active or frozen) from its
/// committed `BatchStore` rows.
pub fn pk_hashes_from_batch_store(
    store: &BatchStore,
    pk_columns: &[String],
) -> Result<HashSet<u64>> {
    let mut batches: Vec<RecordBatch> = Vec::with_capacity(store.len());
    for i in 0..store.len() {
        if let Some(stored) = store.get(i) {
            batches.push(stored.data.clone());
        }
    }
    pk_hashes_from_batches(&batches, pk_columns)
}

/// As-of variant of [`pk_hashes_from_batch_store`]: includes only the first
/// `limit` batches (by append index) — the batches a prior scan observed before
/// the memtable grew. `usize::MAX` includes all (an immutable lower generation).
/// Bounding the membership to what the arm saw keeps the block-list from
/// dropping a base row whose replacement the arm did not deliver.
pub fn pk_hashes_from_batch_store_bounded(
    store: &BatchStore,
    pk_columns: &[String],
    limit: usize,
) -> Result<HashSet<u64>> {
    let end = store.len().min(limit);
    let mut batches: Vec<RecordBatch> = Vec::with_capacity(end);
    for i in 0..end {
        if let Some(stored) = store.get(i) {
            batches.push(stored.data.clone());
        }
    }
    pk_hashes_from_batches(&batches, pk_columns)
}

/// Hash every row's primary key across `batches` into a membership set.
fn pk_hashes_from_batches(batches: &[RecordBatch], pk_columns: &[String]) -> Result<HashSet<u64>> {
    let mut pk_hashes = HashSet::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let pk_indices = resolve_pk_indices(batch, pk_columns)
            .map_err(|e| lance_core::Error::invalid_input(e.to_string()))?;
        for row_idx in 0..batch.num_rows() {
            pk_hashes.insert(compute_pk_hash(batch, &pk_indices, row_idx));
        }
    }
    Ok(pk_hashes)
}

/// Build (or fetch the cached) PK-hash membership for one flushed generation.
/// Cached by immutable path (single-flight); the build scans the flushed
/// dataset's PK columns.
async fn flushed_pk_hashes(
    path: &str,
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<Arc<HashSet<u64>>> {
    match flushed_cache {
        Some(cache) => {
            let build_cache = cache.clone();
            let build_path = path.to_string();
            let build_session = session.cloned();
            let build_pk = pk_columns.to_vec();
            cache
                .get_or_build_pk_hashes(
                    path,
                    // `Box::pin` keeps this build future off the caller's future
                    // (avoids `clippy::large_futures`).
                    Box::pin(async move {
                        let dataset = open_flushed_dataset(
                            &build_path,
                            build_session.as_ref(),
                            Some(&build_cache),
                        )
                        .await?;
                        scan_pk_hashes(&dataset, &build_pk).await
                    }),
                )
                .await
        }
        None => {
            let dataset = open_flushed_dataset(path, session, None).await?;
            Ok(Arc::new(scan_pk_hashes(&dataset, pk_columns).await?))
        }
    }
}

/// Scan a dataset's PK columns and fold them into a membership set, one batch
/// resident at a time (no full PK-column buffer).
async fn scan_pk_hashes(dataset: &Dataset, pk_columns: &[String]) -> Result<HashSet<u64>> {
    let pk_refs: Vec<&str> = pk_columns.iter().map(String::as_str).collect();
    let mut scanner = dataset.scan();
    scanner.project(&pk_refs)?;
    let mut stream = scanner.try_into_stream().await?;
    let mut hashes = HashSet::new();
    while let Some(batch) = stream.try_next().await? {
        if batch.num_rows() == 0 {
            continue;
        }
        let pk_indices = resolve_pk_indices(&batch, pk_columns)
            .map_err(|e| lance_core::Error::invalid_input(e.to_string()))?;
        for row in 0..batch.num_rows() {
            hashes.insert(compute_pk_hash(&batch, &pk_indices, row));
        }
    }
    Ok(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn id_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    /// Hash a single Int32 `id` PK the way the planner does, so a test can probe
    /// a returned blocked set by value.
    fn hash_id(id: i32) -> u64 {
        let batch = id_batch(&[id]);
        let pk_indices = resolve_pk_indices(&batch, &["id".to_string()]).unwrap();
        compute_pk_hash(&batch, &pk_indices, 0)
    }

    /// Whether `id`'s PK hash is blocked by any of a source's newer-gen sets.
    fn blocks(sets: &[Arc<HashSet<u64>>], id: i32) -> bool {
        sets.iter().any(|s| s.contains(&hash_id(id)))
    }

    #[test]
    fn pk_hashes_collapse_within_gen_duplicates() {
        // Two rows share pk=1 (a within-gen duplicate); pk=2 is unique.
        let hashes = pk_hashes_from_batches(&[id_batch(&[1, 2, 1])], &["id".to_string()]).unwrap();
        assert_eq!(hashes.len(), 2); // distinct pks: 1, 2
    }

    #[test]
    fn empty_batches_yield_empty_membership() {
        let hashes = pk_hashes_from_batches(&[id_batch(&[])], &["id".to_string()]).unwrap();
        assert!(hashes.is_empty());
    }

    #[test]
    fn batch_store_membership_collapses_within_gen_dups() {
        let store = BatchStore::with_capacity(8);
        // Two single-row batches, both pk=1 (a within-gen update).
        store.append(id_batch(&[1])).unwrap();
        store.append(id_batch(&[1])).unwrap();
        // A two-row batch: pk=2, pk=3.
        store.append(id_batch(&[2, 3])).unwrap();

        let hashes = pk_hashes_from_batch_store(&store, &["id".to_string()]).unwrap();
        assert_eq!(hashes.len(), 3); // distinct pks: 1, 2, 3
    }

    #[tokio::test]
    async fn fresh_tier_block_list_one_set_per_in_memory_gen() {
        use crate::dataset::mem_wal::scanner::data_source::{LsmDataSource, LsmGeneration};
        use crate::dataset::mem_wal::write::IndexStore;
        use uuid::Uuid;

        let shard = Uuid::new_v4();
        let mk = |ids: &[i32], generation: u64| {
            let store = BatchStore::with_capacity(8);
            store.append(id_batch(ids)).unwrap();
            LsmDataSource::ActiveMemTable {
                batch_store: Arc::new(store),
                index_store: Arc::new(IndexStore::new()),
                schema: id_batch(&[1]).schema(),
                shard_id: shard,
                generation: LsmGeneration::memtable(generation),
            }
        };
        // Active gen 2: pk=1,2. Frozen gen 1: pk=3.
        let sources = vec![mk(&[1, 2], 2), mk(&[3], 1)];

        let sets = fresh_tier_block_list(&sources, &["id".to_string()], None, None, None)
            .await
            .unwrap();

        // One set per generation; together they cover pk=1,2,3 (not 4).
        assert_eq!(sets.len(), 2);
        for id in [1, 2, 3] {
            assert!(blocks(&sets, id));
        }
        assert!(!blocks(&sets, 4));
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

        // Frozen gen 1: stale pk=1.
        // Active gen 2: pk=1 re-written, pk=2 new.
        let sources = vec![mk(&[&[1]], 1), mk(&[&[1], &[2]], 2)];

        let blocked = Box::pin(compute_source_block_lists(
            &sources,
            &["id".to_string()],
            None,
            None,
        ))
        .await
        .unwrap();

        let g1 = LsmGeneration::memtable(1);
        let g2 = LsmGeneration::memtable(2);
        // The newer active write supersedes the frozen copy: gen 1 is blocked on
        // pk=1, so its KNN drops pk=1.
        assert!(blocks(&blocked[&(Some(shard), g1)], 1));
        // The active (newest) generation is superseded by nothing — no entry.
        assert!(!blocked.contains_key(&(Some(shard), g2)));
    }

    #[tokio::test]
    async fn block_lists_suppress_stale_base_row() {
        use crate::dataset::mem_wal::scanner::data_source::{LsmDataSource, LsmGeneration};
        use crate::dataset::mem_wal::write::IndexStore;
        use crate::dataset::{Dataset, WriteParams};
        use arrow_array::RecordBatchIterator;
        use uuid::Uuid;

        // Base (gen 0): pk=1 (stale), pk=3 (live).
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

        let blocked = Box::pin(compute_source_block_lists(
            &sources,
            &["id".to_string()],
            None,
            None,
        ))
        .await
        .unwrap();

        // Base is blocked by every newer gen: pk=1 (re-written in gen 1) is
        // blocked, pk=3 (base-only) is not. End-to-end drop: vector_search specs.
        let base_blocked = blocked
            .get(&(None, LsmGeneration::BASE_TABLE))
            .expect("base has a blocked set");
        assert!(blocks(base_blocked, 1));
        assert!(!blocks(base_blocked, 3));
    }

    #[tokio::test]
    async fn block_lists_are_keyed_per_shard() {
        // Regression: generations are per-shard, so a source must only be blocked
        // by newer generations of its OWN shard. A generation-only key would
        // cross-block same-generation sources from different shards.
        use crate::dataset::mem_wal::scanner::data_source::{LsmDataSource, LsmGeneration};
        use crate::dataset::mem_wal::write::IndexStore;
        use uuid::Uuid;

        let mk = |shard: Uuid, ids: &[i32], generation: u64| {
            let store = BatchStore::with_capacity(8);
            store.append(id_batch(ids)).unwrap();
            LsmDataSource::ActiveMemTable {
                batch_store: Arc::new(store),
                index_store: Arc::new(IndexStore::new()),
                schema: id_batch(&[1]).schema(),
                shard_id: shard,
                generation: LsmGeneration::memtable(generation),
            }
        };

        // Two shards, each: frozen gen 1 (stale) + active gen 2 (re-write).
        // Shard A keys pk=1; shard B keys pk=2 (disjoint partitions).
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let sources = vec![
            mk(a, &[1], 1),
            mk(a, &[1], 2),
            mk(b, &[2], 1),
            mk(b, &[2], 2),
        ];

        let blocked = Box::pin(compute_source_block_lists(
            &sources,
            &["id".to_string()],
            None,
            None,
        ))
        .await
        .unwrap();

        let g1 = LsmGeneration::memtable(1);
        let g2 = LsmGeneration::memtable(2);
        // Each shard's gen 1 is blocked by its OWN gen 2 only.
        assert!(blocks(&blocked[&(Some(a), g1)], 1));
        assert!(!blocks(&blocked[&(Some(a), g1)], 2));
        assert!(blocks(&blocked[&(Some(b), g1)], 2));
        assert!(!blocks(&blocked[&(Some(b), g1)], 1));
        // The newest generation of each shard is superseded by nothing.
        assert!(!blocked.contains_key(&(Some(a), g2)));
        assert!(!blocked.contains_key(&(Some(b), g2)));
    }

    /// An as-of cut bounds the active generation to the first
    /// `active_batch_count` batches — the ones the scan arm observed before the
    /// memtable grew. A later append (higher batch index) is invisible to the
    /// check, so a base row is never dropped without the arm having delivered
    /// its replacement.
    #[tokio::test]
    async fn as_of_cut_bounds_active_memtable_by_batch_count() {
        use crate::dataset::mem_wal::scanner::data_source::{
            AsOfCut, LsmDataSource, LsmGeneration,
        };
        use crate::dataset::mem_wal::write::IndexStore;
        use std::collections::HashMap;

        let shard = Uuid::new_v4();
        // Three single-row batches: indices 0, 1, 2.
        let store = BatchStore::with_capacity(8);
        store.append(id_batch(&[1])).unwrap(); // batch 0
        store.append(id_batch(&[2])).unwrap(); // batch 1
        store.append(id_batch(&[3])).unwrap(); // batch 2 (appended after the arm)
        let sources = vec![LsmDataSource::ActiveMemTable {
            batch_store: Arc::new(store),
            index_store: Arc::new(IndexStore::new()),
            schema: id_batch(&[1]).schema(),
            shard_id: shard,
            generation: LsmGeneration::memtable(1),
        }];

        // Cut at 2 batches of gen 1: pk=1,2 are members; pk=3 (batch 2) is not.
        let as_of: HashMap<Uuid, AsOfCut> = [(
            shard,
            AsOfCut {
                active_generation: 1,
                active_batch_count: 2,
            },
        )]
        .into_iter()
        .collect();
        let sets = fresh_tier_block_list(&sources, &["id".to_string()], None, None, Some(&as_of))
            .await
            .unwrap();
        assert!(blocks(&sets, 1));
        assert!(blocks(&sets, 2));
        assert!(!blocks(&sets, 3));

        // No cut → live tier: all three are members.
        let sets = fresh_tier_block_list(&sources, &["id".to_string()], None, None, None)
            .await
            .unwrap();
        for id in [1, 2, 3] {
            assert!(blocks(&sets, id));
        }
    }

    /// A generation above the active generation rolled in after the snapshot and
    /// is excluded whole; a generation below it is immutable (frozen) and
    /// included whole regardless of the active batch count.
    #[tokio::test]
    async fn as_of_cut_excludes_newer_gen_includes_lower_gen() {
        use crate::dataset::mem_wal::scanner::data_source::{
            AsOfCut, LsmDataSource, LsmGeneration,
        };
        use crate::dataset::mem_wal::write::IndexStore;
        use std::collections::HashMap;

        let shard = Uuid::new_v4();
        let mk = |ids: &[i32], generation: u64| {
            let store = BatchStore::with_capacity(8);
            for id in ids {
                store.append(id_batch(&[*id])).unwrap();
            }
            LsmDataSource::ActiveMemTable {
                batch_store: Arc::new(store),
                index_store: Arc::new(IndexStore::new()),
                schema: id_batch(&[1]).schema(),
                shard_id: shard,
                generation: LsmGeneration::memtable(generation),
            }
        };
        // gen 3 newer (after snapshot), gen 2 == active (bounded to 1 batch),
        // gen 1 lower/immutable (whole).
        let sources = vec![mk(&[100], 3), mk(&[20, 21], 2), mk(&[1, 2], 1)];

        let as_of: HashMap<Uuid, AsOfCut> = [(
            shard,
            AsOfCut {
                active_generation: 2,
                active_batch_count: 1,
            },
        )]
        .into_iter()
        .collect();
        let sets = fresh_tier_block_list(&sources, &["id".to_string()], None, None, Some(&as_of))
            .await
            .unwrap();
        assert!(blocks(&sets, 1)); // gen 1, whole
        assert!(blocks(&sets, 2)); // gen 1, whole
        assert!(blocks(&sets, 20)); // gen 2, batch 0
        assert!(!blocks(&sets, 21)); // gen 2, batch 1 — past the cut
        assert!(!blocks(&sets, 100)); // gen 3 — after the snapshot
    }

    /// A flushed generation at or above the active generation was produced by a
    /// flush after the snapshot and is excluded; one strictly below it is
    /// immutable and included.
    #[tokio::test]
    async fn as_of_cut_excludes_flushed_at_or_above_active() {
        use crate::dataset::mem_wal::scanner::data_source::{
            AsOfCut, LsmDataSource, LsmGeneration,
        };
        use crate::dataset::{Dataset, WriteParams};
        use arrow_array::RecordBatchIterator;
        use std::collections::HashMap;

        // A flushed generation 2 holding pk=5, written as a plain dataset.
        let flushed_batch = id_batch(&[5]);
        let schema = flushed_batch.schema();
        let tmp = tempfile::tempdir().unwrap();
        let path = format!("{}/gen2", tmp.path().to_str().unwrap());
        let reader = RecordBatchIterator::new(vec![Ok(flushed_batch)], schema.clone());
        Dataset::write(reader, &path, Some(WriteParams::default()))
            .await
            .unwrap();

        let shard = Uuid::new_v4();
        let sources = vec![LsmDataSource::FlushedMemTable {
            path,
            shard_id: shard,
            generation: LsmGeneration::memtable(2),
        }];

        // active_generation 2 (gen 2 flushed at/after the snapshot): excluded.
        let at: HashMap<Uuid, AsOfCut> = [(
            shard,
            AsOfCut {
                active_generation: 2,
                active_batch_count: u64::MAX,
            },
        )]
        .into_iter()
        .collect();
        let sets = fresh_tier_block_list(&sources, &["id".to_string()], None, None, Some(&at))
            .await
            .unwrap();
        assert!(!blocks(&sets, 5));

        // active_generation 3 (gen 2 strictly below, immutable): included.
        let above: HashMap<Uuid, AsOfCut> = [(
            shard,
            AsOfCut {
                active_generation: 3,
                active_batch_count: u64::MAX,
            },
        )]
        .into_iter()
        .collect();
        let sets = fresh_tier_block_list(&sources, &["id".to_string()], None, None, Some(&above))
            .await
            .unwrap();
        assert!(blocks(&sets, 5));
    }
}
