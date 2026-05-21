// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-generation primary-key index for the LSM vector-search block-list.
//!
//! A [`GenPkIndex`] maps each primary-key hash in one LSM generation to the
//! row address(es) carrying it. It is the membership / block-list unit the
//! vector-search planner threads into each per-source KNN:
//!
//! - the **key set** is the generation's membership, used to block *older*
//!   generations' stale rows (cross-generation supersession), and
//! - the **values** drive the within-generation supersession block (a PK that
//!   occurs more than once in a generation keeps only its newest occurrence).
//!
//! Row addresses are opaque `u64`s whose meaning is fixed by the caller:
//! `_rowaddr` for a flushed generation, the `BatchStore` row position for an
//! active / frozen memtable. Keys are [`compute_pk_hash`] — the same hash the
//! dedup nodes use — so membership probes stay consistent across the system.

use std::collections::{HashMap, HashSet};

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_core::utils::mask::RowAddrTreeMap;

use super::exec::{FreshnessPolarity, compute_pk_hash, resolve_pk_indices};

/// Per-generation PK index. Immutable once built.
#[derive(Debug, Default, Clone)]
pub struct GenPkIndex {
    /// All row addresses for each PK hash in the generation. The `Vec` holds
    /// more than one address only for a within-generation duplicate (the same
    /// PK inserted/updated multiple times before the generation was sealed).
    pk_to_rowaddrs: HashMap<u64, Vec<u64>>,
}

impl GenPkIndex {
    /// Build from a generation's batches, deriving each row's address from
    /// `rowaddr_for(batch_idx, row_idx)`.
    ///
    /// The closure lets the caller pick the address convention: a running
    /// `BatchStore` position for an active/frozen memtable, or the flushed
    /// `_rowaddr` for a flushed generation.
    pub fn from_batches(
        batches: &[RecordBatch],
        pk_columns: &[String],
        mut rowaddr_for: impl FnMut(usize, usize) -> u64,
    ) -> Result<Self> {
        let mut pk_to_rowaddrs: HashMap<u64, Vec<u64>> = HashMap::new();
        for (batch_idx, batch) in batches.iter().enumerate() {
            if batch.num_rows() == 0 {
                continue;
            }
            let pk_indices = resolve_pk_indices(batch, pk_columns)
                .map_err(|e| lance_core::Error::invalid_input(e.to_string()))?;
            for row_idx in 0..batch.num_rows() {
                let pk_hash = compute_pk_hash(batch, &pk_indices, row_idx);
                pk_to_rowaddrs
                    .entry(pk_hash)
                    .or_default()
                    .push(rowaddr_for(batch_idx, row_idx));
            }
        }
        Ok(Self { pk_to_rowaddrs })
    }

    /// Build directly from `(pk_hash, rowaddr)` pairs. Useful for tests and for
    /// callers that already have hashes in hand.
    pub fn from_hashed(rows: impl IntoIterator<Item = (u64, u64)>) -> Self {
        let mut pk_to_rowaddrs: HashMap<u64, Vec<u64>> = HashMap::new();
        for (pk_hash, rowaddr) in rows {
            pk_to_rowaddrs.entry(pk_hash).or_default().push(rowaddr);
        }
        Self { pk_to_rowaddrs }
    }

    /// Number of distinct PKs in the generation.
    pub fn len(&self) -> usize {
        self.pk_to_rowaddrs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pk_to_rowaddrs.is_empty()
    }

    /// Extend `set` with this generation's membership (its PK hashes).
    ///
    /// The planner accumulates `NEWER(G)` newest-to-oldest by folding each
    /// generation's membership into a running set.
    pub fn extend_membership(&self, set: &mut HashSet<u64>) {
        set.extend(self.pk_to_rowaddrs.keys().copied());
    }

    /// Row addresses of rows superseded *within* this generation: every
    /// occurrence of a PK except its newest, where "newest" follows the
    /// source's write polarity — the largest row address for an insert-ordered
    /// source, the smallest for a reverse-written flushed generation.
    pub fn within_gen_superseded(&self, polarity: FreshnessPolarity) -> RowAddrTreeMap {
        let mut blocked = RowAddrTreeMap::new();
        for addrs in self.pk_to_rowaddrs.values() {
            if addrs.len() < 2 {
                continue;
            }
            let newest = match polarity {
                FreshnessPolarity::InsertOrder => addrs.iter().copied().max(),
                FreshnessPolarity::ReverseWrite => addrs.iter().copied().min(),
            }
            .expect("addrs is non-empty (len >= 2 checked above)");
            for &addr in addrs {
                if addr != newest {
                    blocked.insert(addr);
                }
            }
        }
        blocked
    }

    /// Row addresses of rows in this generation whose PK has a newer version in
    /// `newer` (the union of all newer generations' membership). These are the
    /// cross-generation superseded rows to block.
    pub fn superseded_by(&self, newer: &HashSet<u64>) -> RowAddrTreeMap {
        let mut blocked = RowAddrTreeMap::new();
        for (pk_hash, addrs) in &self.pk_to_rowaddrs {
            if newer.contains(pk_hash) {
                for &addr in addrs {
                    blocked.insert(addr);
                }
            }
        }
        blocked
    }
}

/// Compute per-generation block-lists for the generations that carry a
/// [`GenPkIndex`] (flushed generations and active / frozen memtables), processed
/// newest-first.
///
/// A row is blocked if it is superseded *within* its own generation (a within-gen
/// duplicate that is not the newest) or if its PK has a newer version in any later
/// generation. The returned block-lists are aligned with `gens_newest_first`.
///
/// The second return value is the union of every input generation's membership.
/// The base table (generation 0) has no `GenPkIndex`, so a caller blocks base's
/// superseded rows by scanning base and testing each row's PK hash against this
/// set (`base_superseded` is the union of all newer generations relative to base).
pub fn compute_block_lists(
    gens_newest_first: &[(FreshnessPolarity, &GenPkIndex)],
) -> (Vec<RowAddrTreeMap>, HashSet<u64>) {
    let mut newer: HashSet<u64> = HashSet::new();
    let mut blocks = Vec::with_capacity(gens_newest_first.len());
    for (polarity, index) in gens_newest_first {
        // block(S) = within_S_superseded ∪ { rows of S whose PK ∈ NEWER(G) }
        let mut block = index.within_gen_superseded(*polarity);
        block |= index.superseded_by(&newer);
        blocks.push(block);
        // Fold this generation into NEWER for the next (older) generation.
        index.extend_membership(&mut newer);
    }
    (blocks, newer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::mask::RowAddrMask;
    use std::sync::Arc;

    /// `selected(addr) == false` iff `addr` is in the block tree.
    fn blocks(tree: RowAddrTreeMap, addr: u64) -> bool {
        !RowAddrMask::from_block(tree).selected(addr)
    }

    fn int_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids.to_vec()))]).unwrap()
    }

    #[test]
    fn membership_is_the_distinct_pk_set() {
        // pk hash 7 appears twice; membership counts it once.
        let index = GenPkIndex::from_hashed([(7, 100), (7, 200), (9, 300)]);
        assert_eq!(index.len(), 2);
        let mut set = HashSet::new();
        index.extend_membership(&mut set);
        assert_eq!(set, HashSet::from([7, 9]));
    }

    #[test]
    fn within_gen_superseded_keeps_newest_per_polarity() {
        // pk 7 -> {100, 200}; pk 9 -> {300} (unique, never blocked).
        let index = GenPkIndex::from_hashed([(7, 100), (7, 200), (9, 300)]);

        // Insert order: newest = largest (200), so 100 is superseded.
        let block = index.within_gen_superseded(FreshnessPolarity::InsertOrder);
        assert!(blocks(block.clone(), 100));
        assert!(!blocks(block.clone(), 200));
        assert!(!blocks(block, 300));

        // Reverse write: newest = smallest (100), so 200 is superseded.
        let block = index.within_gen_superseded(FreshnessPolarity::ReverseWrite);
        assert!(blocks(block.clone(), 200));
        assert!(!blocks(block.clone(), 100));
        assert!(!blocks(block, 300));
    }

    #[test]
    fn superseded_by_blocks_only_pks_present_in_newer() {
        let index = GenPkIndex::from_hashed([(7, 100), (9, 300), (9, 301)]);
        let newer = HashSet::from([9u64]); // pk 9 was updated in a newer gen
        let block = index.superseded_by(&newer);
        // Both addresses of pk 9 are blocked; pk 7 survives.
        assert!(blocks(block.clone(), 300));
        assert!(blocks(block.clone(), 301));
        assert!(!blocks(block, 100));
    }

    #[test]
    fn from_batches_hashes_match_pk_helper() {
        // Two rows share pk=1 (a within-gen duplicate); pk=2 is unique.
        let batches = [int_batch(&[1, 2, 1])];
        let pk = ["id".to_string()];
        let index = GenPkIndex::from_batches(&batches, &pk, |_, row| row as u64).unwrap();
        assert_eq!(index.len(), 2);

        // The duplicate pk=1 sits at rows 0 and 2; insert-order newest is row 2,
        // so row 0 is the only within-gen superseded address.
        let block = index.within_gen_superseded(FreshnessPolarity::InsertOrder);
        assert!(blocks(block.clone(), 0));
        assert!(!blocks(block.clone(), 2));
        assert!(!blocks(block, 1));
    }

    #[test]
    fn compute_block_lists_handles_within_and_cross_gen() {
        // Newer gen (active, insert order): pk 1 @ addr 10, pk 2 @ addr 11.
        let gen_new = GenPkIndex::from_hashed([(1, 10), (2, 11)]);
        // Older gen (flushed, reverse write): pk 1 @ 100 (superseded by newer
        // gen), pk 3 @ {101, 102} (within-gen dup; reverse-write newest = 101).
        let gen_old = GenPkIndex::from_hashed([(1, 100), (3, 101), (3, 102)]);

        let (block_lists, newer) = compute_block_lists(&[
            (FreshnessPolarity::InsertOrder, &gen_new),
            (FreshnessPolarity::ReverseWrite, &gen_old),
        ]);

        // Newest gen blocks nothing: no within-gen dups, no newer gen above it.
        assert!(!blocks(block_lists[0].clone(), 10));
        assert!(!blocks(block_lists[0].clone(), 11));

        // Older gen blocks the cross-gen superseded pk 1 (@100) and the
        // within-gen older copy of pk 3 (@102), but keeps the newest pk 3 (@101).
        assert!(blocks(block_lists[1].clone(), 100));
        assert!(blocks(block_lists[1].clone(), 102));
        assert!(!blocks(block_lists[1].clone(), 101));

        // Membership union spans both generations.
        assert_eq!(newer, HashSet::from([1, 2, 3]));
    }

    #[test]
    fn empty_batches_yield_empty_index() {
        let index =
            GenPkIndex::from_batches(&[int_batch(&[])], &["id".to_string()], |_, r| r as u64)
                .unwrap();
        assert!(index.is_empty());
    }
}
