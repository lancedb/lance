// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-source block-list construction for LSM vector search.
//!
//! Given the LSM data sources for a query, this module builds each generation's
//! [`GenPkIndex`] (its set of PK hashes) and, from those, the per-source blocked
//! PK-hash set: `NEWER(G)`, the union of every newer generation's membership.
//! A source's vector search drops any candidate whose primary key hashes into
//! its set (see [`super::exec::PkHashFilterExec`]), suppressing rows superseded
//! by a newer generation. The base table (oldest) is blocked by the union of
//! every generation's membership.
//!
//! Within-generation duplicates are not handled here (see
//! [`super::gen_pk_index`]); the global dedup's `(generation, freshness)`
//! tiebreaker collapses those over the merged stream.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::TryStreamExt;
use lance_core::Result;

use super::data_source::{LsmDataSource, LsmGeneration};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use super::gen_pk_index::{GenPkIndex, compute_blocked_sets};
use crate::dataset::Dataset;
use crate::dataset::mem_wal::write::BatchStore;
use crate::session::Session;

/// Build a [`GenPkIndex`] (PK-hash membership) for an in-memory memtable (active
/// or frozen) from its committed `BatchStore` rows.
pub fn pk_index_from_batch_store(store: &BatchStore, pk_columns: &[String]) -> Result<GenPkIndex> {
    let mut batches: Vec<RecordBatch> = Vec::with_capacity(store.len());
    for i in 0..store.len() {
        if let Some(stored) = store.get(i) {
            batches.push(stored.data.clone());
        }
    }
    GenPkIndex::from_batches(&batches, pk_columns)
}

/// Compute the blocked PK-hash set for every LSM source in `sources`.
///
/// Each entry maps a generation to the set of PK hashes that supersede it:
/// `NEWER(G)` for a flushed / in-memory generation, and the union of every
/// generation's membership for the base table (which is older than all of them).
/// A source's KNN drops any candidate whose primary key hashes into its set.
///
/// Only generations that something supersedes get an entry, so the caller can
/// treat a present entry as "this source needs filtering" and skip the cost
/// otherwise. The newest generation never has an entry.
pub async fn compute_source_block_lists(
    sources: &[LsmDataSource],
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<HashMap<LsmGeneration, Arc<HashSet<u64>>>> {
    // Build a PK-hash membership index for every non-base source. The base table
    // carries no index: it is the oldest source, so it never supersedes anything
    // and its own blocked set is just the union of all newer generations.
    let mut indexed: Vec<(LsmGeneration, Arc<GenPkIndex>)> = Vec::new();
    let mut has_base = false;
    for source in sources {
        match source {
            LsmDataSource::BaseTable { .. } => has_base = true,
            LsmDataSource::ActiveMemTable {
                batch_store,
                generation,
                ..
            } => {
                let index = Arc::new(pk_index_from_batch_store(batch_store, pk_columns)?);
                indexed.push((*generation, index));
            }
            LsmDataSource::FlushedMemTable {
                path, generation, ..
            } => {
                // Cached by immutable path so repeated searches skip the PK scan.
                let index = flushed_pk_index(path, pk_columns, session, flushed_cache).await?;
                indexed.push((*generation, index));
            }
        }
    }

    // Newest generation first so each older generation is blocked against the
    // union of every newer generation's membership (`NEWER(G)`).
    indexed.sort_by_key(|(generation, _)| std::cmp::Reverse(*generation));
    let gens_newest_first: Vec<&GenPkIndex> =
        indexed.iter().map(|(_, index)| index.as_ref()).collect();
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

/// Build (or fetch the cached) [`GenPkIndex`] for one flushed generation.
///
/// With a cache, the index is built once per immutable path (single-flight) and
/// reused across queries; without one, it is built cold each call. The build
/// opens the flushed dataset and scans its PK columns.
async fn flushed_pk_index(
    path: &str,
    pk_columns: &[String],
    session: Option<&Arc<Session>>,
    flushed_cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<Arc<GenPkIndex>> {
    match flushed_cache {
        Some(cache) => {
            let build_cache = cache.clone();
            let build_path = path.to_string();
            let build_session = session.cloned();
            let build_pk = pk_columns.to_vec();
            cache
                .get_or_build_pk_index(
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
                        GenPkIndex::from_batches(&batches, &build_pk)
                    }),
                )
                .await
        }
        None => {
            let dataset = open_flushed_dataset(path, session, None).await?;
            let batches = scan_pk(&dataset, pk_columns).await?;
            Ok(Arc::new(GenPkIndex::from_batches(&batches, pk_columns)?))
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
        use crate::dataset::mem_wal::scanner::exec::{compute_pk_hash, resolve_pk_indices};
        let batch = id_batch(&[id]);
        let pk_indices = resolve_pk_indices(&batch, &["id".to_string()]).unwrap();
        compute_pk_hash(&batch, &pk_indices, 0)
    }

    #[test]
    fn batch_store_index_collapses_within_gen_dups() {
        let store = BatchStore::with_capacity(8);
        // Two single-row batches, both pk=1 (a within-gen update).
        store.append(id_batch(&[1])).unwrap();
        store.append(id_batch(&[1])).unwrap();
        // A two-row batch: pk=2, pk=3.
        store.append(id_batch(&[2, 3])).unwrap();

        let index = pk_index_from_batch_store(&store, &["id".to_string()]).unwrap();
        assert_eq!(index.len(), 3); // distinct pks: 1, 2, 3
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

        // Base's blocked set is the union of all newer generations. pk=1 was
        // re-written in gen 1, so its hash is blocked (the stale base copy gets
        // dropped); pk=3 exists only in base, so its hash is absent (kept). The
        // end-to-end drop is covered by the vector_search base specs.
        let base_blocked = blocked
            .get(&LsmGeneration::BASE_TABLE)
            .expect("base has a blocked set");
        assert!(base_blocked.contains(&hash_id(1)));
        assert!(!base_blocked.contains(&hash_id(3)));
    }
}
