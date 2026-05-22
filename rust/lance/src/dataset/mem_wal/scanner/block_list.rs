// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-source block-list construction for LSM vector search.
//!
//! A generation's membership is its set of PK hashes ([`compute_pk_hash`]). This
//! builds each source's *blocked* set — `NEWER(G)`, the union of every newer
//! generation's membership (base table: the union of all of them). The KNN drops
//! candidates whose PK is in the set (see [`super::exec::PkHashFilterExec`]).
//!
//! Cross-generation only: within-gen duplicates share a hash and are collapsed
//! downstream by the global dedup's `(generation, freshness)` tiebreaker.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::TryStreamExt;
use lance_core::Result;

use super::data_source::{LsmDataSource, LsmGeneration};
use super::exec::{compute_pk_hash, resolve_pk_indices};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use crate::dataset::Dataset;
use crate::dataset::mem_wal::write::BatchStore;
use crate::session::Session;

/// Per-source blocked PK-hash set: each generation maps to `NEWER(G)`, the base
/// table to the union of all generations. Only superseded generations get an
/// entry (a present entry means "this source needs filtering"); the newest never
/// does.
pub async fn compute_source_block_lists(
    sources: &[LsmDataSource],
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<HashMap<LsmGeneration, Arc<HashSet<u64>>>> {
    // Hash each non-base source's membership. Base is the oldest source, so it
    // supersedes nothing; its blocked set is just the union of all of them.
    let mut indexed: Vec<(LsmGeneration, Arc<HashSet<u64>>)> = Vec::new();
    let mut has_base = false;
    for source in sources {
        match source {
            LsmDataSource::BaseTable { .. } => has_base = true,
            LsmDataSource::ActiveMemTable {
                batch_store,
                generation,
                ..
            } => {
                let hashes = Arc::new(pk_hashes_from_batch_store(batch_store, pk_columns)?);
                indexed.push((*generation, hashes));
            }
            LsmDataSource::FlushedMemTable {
                path, generation, ..
            } => {
                // Cached by immutable path so repeated searches skip the PK scan.
                let hashes = flushed_pk_hashes(path, pk_columns, session, flushed_cache).await?;
                indexed.push((*generation, hashes));
            }
        }
    }

    // Newest first so each older gen is blocked against `NEWER(G)`.
    indexed.sort_by_key(|(generation, _)| std::cmp::Reverse(*generation));
    let gens_newest_first: Vec<&HashSet<u64>> =
        indexed.iter().map(|(_, hashes)| hashes.as_ref()).collect();
    let (per_gen_blocked, full_union) = compute_blocked_sets(&gens_newest_first);

    let mut blocked: HashMap<LsmGeneration, Arc<HashSet<u64>>> = HashMap::new();
    for ((generation, _), set) in indexed.iter().zip(per_gen_blocked) {
        if !set.is_empty() {
            blocked.insert(*generation, Arc::new(set));
        }
    }
    // The base table (oldest) is superseded by every non-base generation.
    if has_base && !full_union.is_empty() {
        blocked.insert(LsmGeneration::BASE_TABLE, Arc::new(full_union));
    }
    Ok(blocked)
}

/// `NEWER(G)` for each generation (aligned with `gens_newest_first`; newest gets
/// an empty set), plus the union of all of them — the base table's blocked set.
fn compute_blocked_sets(gens_newest_first: &[&HashSet<u64>]) -> (Vec<HashSet<u64>>, HashSet<u64>) {
    let mut newer: HashSet<u64> = HashSet::new();
    let mut per_gen = Vec::with_capacity(gens_newest_first.len());
    for hashes in gens_newest_first {
        // NEWER(G): everything folded in so far (newest-first).
        per_gen.push(newer.clone());
        newer.extend(hashes.iter().copied());
    }
    (per_gen, newer)
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
                        let batches = scan_pk(&dataset, &build_pk).await?;
                        pk_hashes_from_batches(&batches, &build_pk)
                    }),
                )
                .await
        }
        None => {
            let dataset = open_flushed_dataset(path, session, None).await?;
            let batches = scan_pk(&dataset, pk_columns).await?;
            Ok(Arc::new(pk_hashes_from_batches(&batches, pk_columns)?))
        }
    }
}

/// Scan a dataset's PK columns, collecting the result batches.
async fn scan_pk(dataset: &Dataset, pk_columns: &[String]) -> Result<Vec<RecordBatch>> {
    let pk_refs: Vec<&str> = pk_columns.iter().map(String::as_str).collect();
    let mut scanner = dataset.scan();
    scanner.project(&pk_refs)?;
    let stream = scanner.try_into_stream().await?;
    stream.try_collect::<Vec<_>>().await
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

    #[test]
    fn compute_blocked_sets_accumulates_newer_generations() {
        // Newest gen: pk 1, 2. Older gen: pk 1, 3.
        let gen_new = HashSet::from([1u64, 2]);
        let gen_old = HashSet::from([1u64, 3]);

        let (per_gen, full_union) = compute_blocked_sets(&[&gen_new, &gen_old]);

        // The newest generation is superseded by nothing.
        assert!(per_gen[0].is_empty());
        // The older generation is superseded by the newer one's membership.
        assert_eq!(per_gen[1], HashSet::from([1, 2]));
        // The full union (base's blocked set) spans both generations.
        assert_eq!(full_union, HashSet::from([1, 2, 3]));
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
        // The newer active write supersedes the frozen copy: gen 1's blocked set
        // contains pk=1's hash, so its KNN drops pk=1.
        assert!(blocked[&g1].contains(&hash_id(1)));
        // The active (newest) generation is superseded by nothing — no entry.
        assert!(!blocked.contains_key(&g2));
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

        // Base's blocked set = union of newer gens: pk=1 (re-written in gen 1) is
        // blocked, pk=3 (base-only) is not. End-to-end drop: vector_search specs.
        let base_blocked = blocked
            .get(&LsmGeneration::BASE_TABLE)
            .expect("base has a blocked set");
        assert!(base_blocked.contains(&hash_id(1)));
        assert!(!base_blocked.contains(&hash_id(3)));
    }
}
