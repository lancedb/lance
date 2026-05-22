// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-generation primary-key membership for the LSM vector-search block-list.
//!
//! A [`GenPkIndex`] is the set of primary-key hashes present in one LSM
//! generation. The vector-search planner uses it to suppress *cross-generation*
//! stale rows: a row in generation `G` is superseded if its PK also appears in a
//! newer generation, so each source drops candidates whose PK hashes into
//! `NEWER(G)` — the union of every strictly-newer generation's membership (see
//! [`compute_blocked_sets`]).
//!
//! Keys are [`compute_pk_hash`] — the same hash the dedup nodes use — so
//! membership probes stay consistent across the system.
//!
//! Within-generation duplicates (the same PK written twice before a generation
//! was sealed) are *not* handled here: they share a hash, so they can't be
//! disambiguated by membership. They are instead collapsed by the global dedup's
//! `(generation, freshness)` tiebreaker over the merged stream.

use std::collections::HashSet;

use arrow_array::RecordBatch;
use lance_core::Result;

use super::exec::{compute_pk_hash, resolve_pk_indices};

/// The set of PK hashes present in one generation. Immutable once built.
#[derive(Debug, Default, Clone)]
pub struct GenPkIndex {
    pk_hashes: HashSet<u64>,
}

impl GenPkIndex {
    /// Build from a generation's batches by hashing each row's primary key.
    pub fn from_batches(batches: &[RecordBatch], pk_columns: &[String]) -> Result<Self> {
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
        Ok(Self { pk_hashes })
    }

    /// Build directly from PK hashes. Useful for tests and for callers that
    /// already have hashes in hand.
    pub fn from_hashed(hashes: impl IntoIterator<Item = u64>) -> Self {
        Self {
            pk_hashes: hashes.into_iter().collect(),
        }
    }

    /// Number of distinct PKs in the generation.
    pub fn len(&self) -> usize {
        self.pk_hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pk_hashes.is_empty()
    }

    /// Union this generation's membership into `set`.
    pub fn extend_into(&self, set: &mut HashSet<u64>) {
        set.extend(self.pk_hashes.iter().copied());
    }
}

/// For each generation in `gens_newest_first`, the set of PK hashes that
/// supersede it: `NEWER(G)`, the union of every strictly-newer generation's
/// membership. A source drops any candidate whose PK hashes into its set. The
/// returned vector is aligned with `gens_newest_first`; the newest generation
/// gets an empty set (nothing supersedes it).
///
/// The second return value is the union of *every* input generation's
/// membership — the blocked set for the base table (generation 0), which is
/// older than all of them.
pub fn compute_blocked_sets(
    gens_newest_first: &[&GenPkIndex],
) -> (Vec<HashSet<u64>>, HashSet<u64>) {
    let mut newer: HashSet<u64> = HashSet::new();
    let mut per_gen = Vec::with_capacity(gens_newest_first.len());
    for index in gens_newest_first {
        // NEWER(G) is the union of generations strictly newer than this one,
        // i.e. everything folded in so far (we process newest-first).
        per_gen.push(newer.clone());
        index.extend_into(&mut newer);
    }
    (per_gen, newer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn int_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    #[test]
    fn membership_is_the_distinct_pk_set() {
        // hash 7 appears twice; membership counts it once.
        let index = GenPkIndex::from_hashed([7, 7, 9]);
        assert_eq!(index.len(), 2);
        let mut set = HashSet::new();
        index.extend_into(&mut set);
        assert_eq!(set, HashSet::from([7, 9]));
    }

    #[test]
    fn from_batches_collapses_within_gen_duplicates() {
        // Two rows share pk=1 (a within-gen duplicate); pk=2 is unique.
        let batches = [int_batch(&[1, 2, 1])];
        let index = GenPkIndex::from_batches(&batches, &["id".to_string()]).unwrap();
        assert_eq!(index.len(), 2); // distinct pks: 1, 2
    }

    #[test]
    fn compute_blocked_sets_accumulates_newer_generations() {
        // Newest gen: pk 1, 2. Older gen: pk 1, 3.
        let gen_new = GenPkIndex::from_hashed([1, 2]);
        let gen_old = GenPkIndex::from_hashed([1, 3]);

        let (per_gen, full_union) = compute_blocked_sets(&[&gen_new, &gen_old]);

        // The newest generation is superseded by nothing.
        assert!(per_gen[0].is_empty());
        // The older generation is superseded by the newer one's membership.
        assert_eq!(per_gen[1], HashSet::from([1, 2]));
        // The full union (base's blocked set) spans both generations.
        assert_eq!(full_union, HashSet::from([1, 2, 3]));
    }

    #[test]
    fn empty_batches_yield_empty_index() {
        let index = GenPkIndex::from_batches(&[int_batch(&[])], &["id".to_string()]).unwrap();
        assert!(index.is_empty());
    }
}
