// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion lance-format/lance#7499): a Bε-tree manifest.
//!
//! This is a benchmark artifact, not a production format. It exists to verify
//! the counter-proposal's central claim: for the add-column / backfill workload
//! (data files trickle into all fragments over *many* commits), a Bε-tree keeps
//! per-commit write cost bounded (≈ the ε-buffer) and total write cost
//! sub-linear in the commit count, whereas the flat manifest rewrites the full
//! growing fragment list on every commit.
//!
//! Shape (two levels; recursion to 3+ levels is future work — see DESIGN §6):
//!
//! ```text
//!   root (protobuf)  = table metadata + child refs + fragment_actions ε-buffer
//!        |  append actions per commit; flush a batch on overflow
//!        v
//!   children (Lance) = contiguous fragment-id ranges, one row per fragment
//! ```
//!
//! A commit appends [`pb::FragmentAction`]s to the root buffer and rewrites only
//! the small root. When the buffer exceeds its cap, [`BeTree::flush`] drains it,
//! routes actions to the owning children by fragment-id range, and rewrites just
//! those children — the batching is where the write-amplification win comes from.

pub mod action;
pub mod child;
pub mod flat_baseline;
pub mod root;
pub mod support;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use object_store::path::Path;

use crate::betree::child::ChildIo;
use crate::betree::root::{RootState, child_path, read_latest_version, read_root, write_root};
use crate::format::{Fragment, pb};
use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;

/// Bε-tree tuning knobs.
#[derive(Debug, Clone)]
pub struct BeTreeConfig {
    /// Root ε-buffer cap. When the encoded buffer exceeds this, the commit flushes.
    pub buffer_cap_bytes: u64,
    /// Number of children the fragment-id space is partitioned into at bootstrap.
    /// (Split/merge to self-balance node size is future work — see DESIGN §6.)
    pub num_children: usize,
}

/// Bytes written while bootstrapping the tree.
#[derive(Debug, Default, Clone, Copy)]
pub struct BootstrapStats {
    pub child_bytes: u64,
    pub root_bytes: u64,
    pub num_children: u64,
}

/// Bytes written / read for a single commit.
#[derive(Debug, Default, Clone, Copy)]
pub struct CommitStats {
    /// Bytes written for the root object this commit (always paid).
    pub root_bytes: u64,
    /// Bytes written for children rewritten by a flush (0 if no flush).
    pub flush_write_bytes: u64,
    /// Bytes read from children during a flush (read-modify-write).
    pub flush_read_bytes: u64,
    /// Number of children rewritten by a flush.
    pub flush_children: u64,
    pub flushed: bool,
}

impl CommitStats {
    /// Total bytes written this commit (root + any flushed children).
    pub fn write_bytes(&self) -> u64 {
        self.root_bytes + self.flush_write_bytes
    }
}

struct FlushStats {
    write_bytes: u64,
    read_bytes: u64,
    children: u64,
}

/// A writer session over a Bε-tree. Holds only child refs + the ε-buffer in
/// memory (never the full fragment list — that is the memory win over flat).
pub struct BeTree {
    object_store: Arc<ObjectStore>,
    base: Path,
    child_io: ChildIo,
    config: BeTreeConfig,
    root: RootState,
}

impl BeTree {
    /// Bootstrap the tree from a full fragment list: partition into contiguous
    /// id-range children, write each as a Lance file, then write the root. This
    /// models "a single commit that writes a manifest with 1M fragments".
    #[allow(clippy::too_many_arguments)]
    pub async fn bootstrap(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
        config: BeTreeConfig,
        mut fragments: Vec<Fragment>,
        schema_pb: Vec<u8>,
    ) -> Result<(Self, BootstrapStats)> {
        if fragments.is_empty() {
            return Err(Error::invalid_input(
                "BeTree::bootstrap requires at least one fragment",
            ));
        }
        fragments.sort_by_key(|f| f.id);
        let child_io = ChildIo::new(object_store.clone(), scheduler, cache);

        // Partition into exactly `c` contiguous, non-empty id-range children (the
        // first `rem` children get one extra fragment). Because the children stay
        // sorted by `min_frag_id` and non-empty, `child_index_for`'s binary search
        // routes correctly.
        let n = fragments.len();
        let c = config.num_children.clamp(1, n);
        let base_len = n / c;
        let rem = n % c;

        let mut children = Vec::with_capacity(c);
        let mut child_bytes = 0u64;
        let mut start = 0usize;
        for i in 0..c {
            let len = base_len + if i < rem { 1 } else { 0 };
            let chunk_frags = &fragments[start..start + len];
            start += len;
            let path = child_path(&base, &format!("c{i}-v1"));
            let res = child_io.write(&path, chunk_frags).await?;
            children.push(make_child_ref(path.to_string(), chunk_frags, res.byte_size));
            child_bytes += res.byte_size;
        }
        let num_children = children.len() as u64;

        let root = RootState {
            version: 1,
            children,
            buffer: Vec::new(),
            buffer_cap_bytes: config.buffer_cap_bytes,
            schema_pb,
        };
        let root_bytes = write_root(&object_store, &base, &root).await?;

        let tree = Self {
            object_store,
            base,
            child_io,
            config,
            root,
        };
        Ok((
            tree,
            BootstrapStats {
                child_bytes,
                root_bytes,
                num_children,
            },
        ))
    }

    /// Append `actions` to the root ε-buffer, flush on overflow, and write the
    /// new (small) root object.
    ///
    /// Prototype limitation: this mutates in-memory state (version, buffer, child
    /// refs) as it writes; a mid-flush I/O error leaves the tree inconsistent.
    /// A production writer would stage the new root/children and swap in only
    /// after all writes succeed. The benchmark aborts on any error, so the
    /// inconsistent state is never observed.
    pub async fn commit(&mut self, actions: Vec<pb::FragmentAction>) -> Result<CommitStats> {
        self.root.buffer.extend(actions);
        self.root.version += 1;

        let mut stats = CommitStats::default();
        if self.root.buffer_encoded_len() as u64 > self.config.buffer_cap_bytes {
            let flush = self.flush().await?;
            stats.flushed = true;
            stats.flush_write_bytes = flush.write_bytes;
            stats.flush_read_bytes = flush.read_bytes;
            stats.flush_children = flush.children;
        }
        stats.root_bytes = write_root(&self.object_store, &self.base, &self.root).await?;
        Ok(stats)
    }

    /// Drain the ε-buffer, route actions to owning children, and rewrite only
    /// the touched children (read-modify-write). This is the amortized step.
    async fn flush(&mut self) -> Result<FlushStats> {
        let mut by_child: HashMap<usize, Vec<pb::FragmentAction>> = HashMap::new();
        for act in std::mem::take(&mut self.root.buffer) {
            let fid = action::target_frag_id(&act).ok_or_else(|| {
                Error::invalid_input("fragment action without a target fragment id")
            })?;
            by_child
                .entry(self.child_index_for(fid))
                .or_default()
                .push(act);
        }

        // Rewrite every touched child concurrently — they are independent files
        // (disjoint id ranges, distinct objects), so a flush's wall-clock is one
        // child's read-modify-write, not the sum across children.
        let base = self.base.clone();
        let version = self.root.version;
        let child_io = &self.child_io;
        let jobs: Vec<(usize, pb::ChildRef, Vec<pb::FragmentAction>)> = by_child
            .into_iter()
            .map(|(idx, actions)| (idx, self.root.children[idx].clone(), actions))
            .collect();
        let tasks = jobs.into_iter().map(|(idx, child_ref, actions)| {
            let base = &base;
            async move {
                let old_path = Path::from(child_ref.node_path.as_str());
                let fragments = child_io.read(&old_path, Some(child_ref.byte_size)).await?;
                let mut map: BTreeMap<u64, Fragment> =
                    fragments.into_iter().map(|f| (f.id, f)).collect();
                for act in actions {
                    action::apply(&mut map, act)?;
                }
                let new_frags: Vec<Fragment> = map.into_values().collect();
                let new_path = child_path(base, &format!("c{idx}-v{version}"));
                let res = child_io.write(&new_path, &new_frags).await?;
                let new_ref = make_child_ref(new_path.to_string(), &new_frags, res.byte_size);
                Ok::<_, Error>((idx, new_ref, res.byte_size, child_ref.byte_size))
            }
        });
        let results = futures::future::try_join_all(tasks).await?;

        let mut flush = FlushStats {
            write_bytes: 0,
            read_bytes: 0,
            children: 0,
        };
        for (idx, new_ref, write_bytes, read_bytes) in results {
            self.root.children[idx] = new_ref;
            flush.write_bytes += write_bytes;
            flush.read_bytes += read_bytes;
            flush.children += 1;
        }
        Ok(flush)
    }

    /// The child index owning `fid`. Relies on the invariant that `children` are
    /// non-empty and sorted ascending by `min_frag_id` with contiguous ranges —
    /// upheld by `bootstrap` and preserved by add-only backfill (ids never change).
    /// Removals that could empty a child are out of scope (see DESIGN §6).
    fn child_index_for(&self, fid: u64) -> usize {
        match self
            .root
            .children
            .binary_search_by(|c| c.min_frag_id.cmp(&fid))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        }
    }

    /// Materialize the full fragment list from the in-memory root: read children
    /// from storage and overlay the buffered actions.
    pub async fn materialize(&self) -> Result<Vec<Fragment>> {
        Self::overlay(&self.child_io, &self.root.children, &self.root.buffer).await
    }

    pub fn version(&self) -> u64 {
        self.root.version
    }

    pub fn num_children(&self) -> usize {
        self.root.children.len()
    }

    /// Cold open from storage: resolve the latest root, read all children, and
    /// overlay the buffered actions to reconstruct the full fragment list.
    pub async fn cold_open(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
    ) -> Result<Vec<Fragment>> {
        let version = read_latest_version(&object_store, &base).await?;
        let root = read_root(&object_store, &base, version).await?;
        let child_io = ChildIo::new(object_store, scheduler, cache);
        Self::overlay(&child_io, &root.children, &root.buffer).await
    }

    /// Read all children (concurrently) and overlay the buffered actions.
    async fn overlay(
        child_io: &ChildIo,
        children: &[pb::ChildRef],
        buffer: &[pb::FragmentAction],
    ) -> Result<Vec<Fragment>> {
        let paths: Vec<Path> = children
            .iter()
            .map(|c| Path::from(c.node_path.as_str()))
            .collect();
        let reads = children
            .iter()
            .zip(&paths)
            .map(|(c, p)| child_io.read(p, Some(c.byte_size)));
        let per_child = futures::future::try_join_all(reads).await?;

        let mut map: BTreeMap<u64, Fragment> = BTreeMap::new();
        for frags in per_child {
            for f in frags {
                map.insert(f.id, f);
            }
        }
        for act in buffer.iter().cloned() {
            action::apply(&mut map, act)?;
        }
        Ok(map.into_values().collect())
    }
}

fn make_child_ref(node_path: String, fragments: &[Fragment], byte_size: u64) -> pb::ChildRef {
    let num_rows: u64 = fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .sum();
    pb::ChildRef {
        node_path,
        min_frag_id: fragments.first().map(|f| f.id).unwrap_or(0),
        max_frag_id: fragments.last().map(|f| f.id).unwrap_or(0),
        num_fragments: fragments.len() as u64,
        num_rows,
        byte_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betree::flat_baseline::FlatBaseline;
    use crate::betree::support::{make_backfill_data_file, make_fragment};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

    /// A backfill run over a Bε-tree must materialize the exact same fragment
    /// state as the flat manifest — fast is only useful if it is also correct.
    #[tokio::test]
    async fn betree_matches_flat_after_backfill() {
        let n: u64 = 2_000;
        let k: u64 = 8; // backfill commits
        let num_children = 4;
        let buffer_cap_bytes = 32 * 1024; // small cap to force several flushes

        let tempdir = TempObjDir::default();
        let betree_base = tempdir.clone().join("betree");
        let flat_base = tempdir.clone().join("flat");
        let object_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
        let cache = Arc::new(LanceCache::with_capacity(64 * 1024 * 1024));

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let fragments: Vec<Fragment> = (0..n).map(make_fragment).collect();

        let (mut tree, bootstrap) = BeTree::bootstrap(
            object_store.clone(),
            betree_base.clone(),
            scheduler.clone(),
            cache.clone(),
            BeTreeConfig {
                buffer_cap_bytes,
                num_children,
            },
            fragments.clone(),
            Vec::new(),
        )
        .await
        .unwrap();
        assert_eq!(bootstrap.num_children, num_children as u64);

        let mut flat = FlatBaseline::new(
            object_store.clone(),
            flat_base.clone(),
            schema.clone(),
            fragments.clone(),
        );
        flat.write().await.unwrap();

        // Backfill: K commits, each adds column 0 to a disjoint window of frags.
        let per = n.div_ceil(k);
        let mut flushes = 0u64;
        for c in 0..k {
            let start = c * per;
            let end = (start + per).min(n);
            let mut actions = Vec::new();
            let mut flat_adds = Vec::new();
            for id in start..end {
                let df = make_backfill_data_file(id, 0);
                actions.push(action::add_data_file(id, &df));
                flat_adds.push((id, df));
            }
            let stats = tree.commit(actions).await.unwrap();
            if stats.flushed {
                flushes += 1;
            }
            flat.commit_add_data_files(&flat_adds).await.unwrap();
        }
        assert!(flushes > 0, "small buffer cap should have forced a flush");

        // Materialized Bε-tree state must equal the flat manifest state.
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
            let mut a_paths: Vec<_> = a.files.iter().map(|f| f.path.clone()).collect();
            let mut b_paths: Vec<_> = b.files.iter().map(|f| f.path.clone()).collect();
            a_paths.sort();
            b_paths.sort();
            assert_eq!(a_paths, b_paths, "fragment {} data files differ", a.id);
            // Every fragment gained exactly the one backfilled data file.
            assert_eq!(
                a.files.len(),
                2,
                "fragment {} should have 2 data files",
                a.id
            );
        }

        // Cold open from storage (root + children + buffer overlay) also works.
        let cold = BeTree::cold_open(object_store.clone(), betree_base, scheduler, cache)
            .await
            .unwrap();
        assert_eq!(cold.len(), n as usize);
    }

    /// Bootstrap must reject an empty fragment list (no valid child partitioning).
    #[tokio::test]
    async fn bootstrap_rejects_empty_fragments() {
        let tempdir = TempObjDir::default();
        let object_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing());
        let cache = Arc::new(LanceCache::with_capacity(0));
        let result = BeTree::bootstrap(
            object_store,
            tempdir.clone().join("empty"),
            scheduler,
            cache,
            BeTreeConfig {
                buffer_cap_bytes: 1024,
                num_children: 4,
            },
            Vec::new(),
            Vec::new(),
        )
        .await;
        assert!(result.is_err(), "empty bootstrap must be rejected");
    }

    /// An `AddDeletionFile` action carrying no deletion file must not clear an
    /// existing one (add-only semantics).
    #[test]
    fn add_deletion_file_none_does_not_clear() {
        use crate::format::pb::fragment_action::Action;
        use crate::format::{DeletionFile, DeletionFileType};

        let mut fragment = make_fragment(0);
        fragment.deletion_file = Some(DeletionFile {
            read_version: 1,
            id: 7,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(3),
            base_id: None,
        });
        let mut frags = BTreeMap::from([(0u64, fragment)]);

        let noop = pb::FragmentAction {
            action: Some(Action::AddDeletionFile(pb::AddDeletionFile {
                frag_id: 0,
                deletion_file: None,
            })),
        };
        action::apply(&mut frags, noop).unwrap();
        assert!(
            frags.get(&0).unwrap().deletion_file.is_some(),
            "empty AddDeletionFile must not clear the existing deletion file"
        );
    }
}
