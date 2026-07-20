// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion lance-format/lance#7499): a full recursive,
//! self-balancing Bε-tree manifest.
//!
//! A benchmark artifact, not a production format. Verifies the counter-proposal:
//! for add-column / backfill (data files trickle into fragments over *many*
//! commits), a Bε-tree keeps per-commit write cost bounded (≈ the ε-buffer) and
//! self-balances (split on growth, merge on shrink) as fragments × columns grow,
//! whereas the flat manifest rewrites the full growing fragment list every commit.
//!
//! ```text
//!   root (protobuf manifest) = child refs + fragment_actions ε-buffer + metadata
//!        |  msn-tag actions per commit; flush the fullest child's batch on overflow
//!        v
//!   internal nodes (protobuf) = pivots + ε-buffer   (recurse)
//!        v
//!   leaves (Lance files)      = fragment tables      (messages applied here)
//! ```
//!
//! Modules: [`node`] (types + pure logic), [`store`] (copy-on-write node IO),
//! [`tree`] (bootstrap / commit / flush / split / merge / materialize).

pub mod action;
pub mod flat_baseline;
pub mod node;
pub mod store;
pub mod support;
pub mod tree;

pub use node::{BeTreeConfig, DEFAULT_FANOUT, DEFAULT_TARGET_NODE_BYTES};
pub use tree::{BeTree, BootstrapStats, CommitStats};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betree::flat_baseline::FlatBaseline;
    use crate::betree::support::{make_backfill_data_file, make_fragment};
    use crate::format::Fragment;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::cache::LanceCache;
    use lance_core::datatypes::Schema;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use std::sync::Arc;

    fn test_env() -> (Arc<ObjectStore>, Arc<ScanScheduler>, Arc<LanceCache>) {
        let object_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
        let cache = Arc::new(LanceCache::with_capacity(64 * 1024 * 1024));
        (object_store, scheduler, cache)
    }

    fn schema() -> Schema {
        Schema::try_from(&ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]))
        .unwrap()
    }

    /// A multi-column backfill over a multi-level Bε-tree grows leaves past B, so
    /// the tree must SPLIT — and still materialize the exact same fragment state
    /// as the flat manifest.
    #[tokio::test]
    async fn recursive_betree_splits_and_matches_flat() {
        let n: u64 = 3_000;
        let f: u64 = 10; // fragments per commit
        let columns: u32 = 3;
        // Tiny nodes so the tree is several levels deep and splits under backfill.
        let config = BeTreeConfig::new(16 * 1024, 4);

        let tempdir = TempObjDir::default();
        let betree_base = tempdir.clone().join("betree");
        let flat_base = tempdir.clone().join("flat");
        let (object_store, scheduler, cache) = test_env();
        let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();

        let (mut tree, boot) = BeTree::bootstrap(
            object_store.clone(),
            betree_base.clone(),
            scheduler.clone(),
            cache.clone(),
            config,
            fragments.clone(),
            Vec::new(),
        )
        .await
        .unwrap();
        assert!(
            boot.num_leaves > 1 && boot.height >= 2,
            "expect a multi-level tree"
        );

        let mut flat =
            FlatBaseline::new(object_store.clone(), flat_base.clone(), schema(), fragments);
        flat.write().await.unwrap();

        // Backfill `columns` new data files into every fragment, over many small commits.
        let commits = n.div_ceil(f);
        let mut total_splits = 0u64;
        for col in 0..columns {
            for c in 0..commits {
                let start = c * f;
                let end = (start + f).min(n);
                let mut actions = Vec::new();
                let mut flat_adds = Vec::new();
                for id in start..end {
                    let df = make_backfill_data_file(id, col);
                    actions.push(action::add_data_file(id, &df));
                    flat_adds.push((id, df));
                }
                total_splits += tree.commit(actions).await.unwrap().splits;
                flat.commit_add_data_files(&flat_adds).await.unwrap();
            }
        }
        assert!(
            total_splits > 0,
            "growing leaves past B should have split the tree"
        );

        // Materialized Bε-tree state == flat manifest state.
        let mut tree_frags = tree.materialize().await.unwrap();
        let flat_manifest = FlatBaseline::cold_open(&object_store, &flat_base, flat.version())
            .await
            .unwrap();
        let mut flat_frags = flat_manifest.fragments.as_ref().clone();
        tree_frags.sort_by_key(|f| f.id);
        flat_frags.sort_by_key(|f| f.id);

        assert_eq!(tree_frags.len(), n as usize);
        assert_eq!(tree_frags.len(), flat_frags.len());
        for (a, b) in tree_frags.iter().zip(flat_frags.iter()) {
            assert_eq!(a.id, b.id);
            let mut ap: Vec<_> = a.files.iter().map(|f| f.path.clone()).collect();
            let mut bp: Vec<_> = b.files.iter().map(|f| f.path.clone()).collect();
            ap.sort();
            bp.sort();
            assert_eq!(ap, bp, "fragment {} data files differ", a.id);
            assert_eq!(
                a.files.len(),
                1 + columns as usize,
                "fragment {} file count",
                a.id
            );
        }

        // Cold open from storage (root → internal → leaves + buffer overlay) matches.
        let cold = BeTree::cold_open(object_store, betree_base, scheduler, cache)
            .await
            .unwrap();
        assert_eq!(cold.len(), n as usize);
    }

    /// Sparsely deleting 4 of every 5 fragments shrinks each leaf below the merge
    /// floor *without* emptying it, so the tree must MERGE (coalesce underflowing
    /// nodes) — and materialize the correct remainder. Also guards the empty-node
    /// routing invariant: an emptied node must be dropped, not kept with min_key=0.
    #[tokio::test]
    async fn recursive_betree_merges_on_bulk_delete() {
        let n: u64 = 3_000;
        let f: u64 = 20;
        // Small nodes + wide fanout: a shallow tree so removes reach leaves.
        let config = BeTreeConfig::new(4 * 1024, 32);

        let tempdir = TempObjDir::default();
        let base = tempdir.clone().join("betree");
        let (object_store, scheduler, cache) = test_env();
        let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();

        let (mut tree, _boot) = BeTree::bootstrap(
            object_store.clone(),
            base.clone(),
            scheduler.clone(),
            cache.clone(),
            config,
            fragments,
            Vec::new(),
        )
        .await
        .unwrap();

        // Remove 4 of every 5 fragments (keep id % 5 == 4): each leaf keeps ~1/5,
        // dropping below the merge floor → coalesces with neighbors.
        let mut merges = 0u64;
        for c in 0..(n / f) {
            let actions: Vec<_> = (c * f..c * f + f)
                .filter(|id| id % 5 != 4)
                .map(action::remove_fragment)
                .collect();
            merges += tree.commit(actions).await.unwrap().merges;
        }
        assert!(
            merges > 0,
            "shrinking leaves below the merge floor should have coalesced them"
        );

        // Materialize the correct remainder (buffered removes are applied too).
        let mut remaining = tree.materialize().await.unwrap();
        remaining.sort_by_key(|f| f.id);
        assert_eq!(remaining.len(), (n / 5) as usize);
        assert!(
            remaining.iter().all(|f| f.id % 5 == 4),
            "only every fifth fragment survives"
        );
    }

    /// Regression: aggressively flushing a full-prefix delete (tiny min_flush)
    /// fully empties leaves. An emptied node must be *dropped* from its parent,
    /// not kept with min_key=0 — otherwise it sorts to the front and low-id
    /// removes misroute to it and no-op, resurrecting deleted fragments.
    #[tokio::test]
    async fn recursive_betree_empty_leaves_do_not_resurrect() {
        let n: u64 = 3_000;
        let remove: u64 = 2_400; // delete ids [0, 2400); keep [2400, 3000)
        let f: u64 = 20;
        // Force the pathological case: tiny flush gate → leaves empty completely.
        let config = BeTreeConfig {
            target_node_bytes: 4 * 1024,
            fanout: 32,
            min_flush_override: Some(64),
        };

        let tempdir = TempObjDir::default();
        let base = tempdir.clone().join("betree");
        let (object_store, scheduler, cache) = test_env();
        let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();
        let (mut tree, _boot) = BeTree::bootstrap(
            object_store.clone(),
            base.clone(),
            scheduler.clone(),
            cache.clone(),
            config,
            fragments,
            Vec::new(),
        )
        .await
        .unwrap();

        for c in 0..(remove / f) {
            let actions: Vec<_> = (c * f..c * f + f).map(action::remove_fragment).collect();
            tree.commit(actions).await.unwrap();
        }

        let mut remaining = tree.materialize().await.unwrap();
        remaining.sort_by_key(|f| f.id);
        assert_eq!(
            remaining.len(),
            (n - remove) as usize,
            "removes must not be lost"
        );
        assert!(
            remaining.iter().all(|f| f.id >= remove),
            "the deleted prefix must not resurrect"
        );
    }
}
