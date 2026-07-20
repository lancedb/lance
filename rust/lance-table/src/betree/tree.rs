// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): the recursive, self-balancing Bε-tree.
//!
//! Algorithm per `LITERATURE.md`: messages are tagged with a monotonic msn and
//! buffered at the root; a full node flushes the batch destined for its fullest
//! child (gated at `min_flush`), recursing down; a node that overflows splits
//! (root split grows height); underflowing children coalesce with a sibling
//! (root with one child shrinks height). Nodes are immutable — every touched
//! node on the root→leaf path is rewritten (copy-on-write) and the root repoint
//! is the commit.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::betree::node::{self, BeTreeConfig, InternalNode};
use crate::betree::store::NodeStore;
use crate::format::Fragment;
use crate::format::pb;
use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use object_store::path::Path;

/// Accumulated write work over one commit (for stats / benchmark accounting).
#[derive(Debug, Default, Clone, Copy)]
struct WriteAcc {
    io_bytes: u64,
    flushes: u64,
    splits: u64,
    merges: u64,
    /// Deepest tree level at which a flush occurred this commit (0 = root only).
    /// >0 proves multi-level flushing (internal ε-buffers filled and flushed down).
    max_flush_depth: u32,
}

impl WriteAcc {
    fn add(&mut self, o: Self) {
        self.io_bytes += o.io_bytes;
        self.flushes += o.flushes;
        self.splits += o.splits;
        self.merges += o.merges;
        self.max_flush_depth = self.max_flush_depth.max(o.max_flush_depth);
    }
}

/// Result of flushing an internal node: (possibly split) children, residual
/// buffer, and accumulated write work.
type FlushResult = (Vec<pb::ChildRef>, Vec<pb::TaggedAction>, WriteAcc);
/// Result of ingesting into a subtree: the child ref(s) that now represent it
/// (>1 if it split), and accumulated write work.
type IngestResult = (Vec<pb::ChildRef>, WriteAcc);

/// Bytes/structure written while bootstrapping.
#[derive(Debug, Default, Clone, Copy)]
pub struct BootstrapStats {
    pub io_write_bytes: u64,
    pub num_leaves: u64,
    pub height: u32,
}

/// Result of one commit.
#[derive(Debug, Default, Clone, Copy)]
pub struct CommitStats {
    pub io_write_bytes: u64,
    pub flushes: u64,
    pub splits: u64,
    pub merges: u64,
    pub height: u32,
    /// Deepest level flushed this commit (0 = root buffer only; ≥1 = cascaded
    /// into internal nodes — the deep-flush regime).
    pub max_flush_depth: u32,
}

/// A writer session over a recursive Bε-tree. Holds only the root (child refs +
/// ε-buffer + metadata) in memory; interior/leaf nodes are read on demand.
pub struct BeTree {
    store: NodeStore,
    config: BeTreeConfig,
    version: u64,
    children: Vec<pb::ChildRef>,
    buffer: Vec<pb::TaggedAction>,
    next_msn: u64,
    schema_pb: Vec<u8>,
}

impl BeTree {
    /// Bootstrap a balanced tree from a full fragment list (held in memory).
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
        let n = fragments.len() as u64;
        let mut iter = fragments.into_iter();
        Self::bootstrap_generate(
            object_store,
            base,
            scheduler,
            cache,
            config,
            n,
            move |_| iter.next().unwrap(),
            schema_pb,
        )
        .await
    }

    /// Bootstrap by *streaming* `num_fragments` fragments from `gen` (called with
    /// ids 0..num_fragments, in order), packing them into ~0.5 B leaves without
    /// ever holding the whole fragment list — this is what lets the tree reach
    /// billion-data-file scale (fat fragments) within a bounded memory budget.
    #[allow(clippy::too_many_arguments)]
    pub async fn bootstrap_generate<F: FnMut(u64) -> Fragment>(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
        config: BeTreeConfig,
        num_fragments: u64,
        mut gen_fn: F,
        schema_pb: Vec<u8>,
    ) -> Result<(Self, BootstrapStats)> {
        if num_fragments == 0 {
            return Err(Error::invalid_input(
                "BeTree::bootstrap requires at least one fragment",
            ));
        }
        let store = NodeStore::new(object_store, base, scheduler, cache);
        let target = config.split_piece_bytes();

        let mut io = 0u64;
        let mut layer: Vec<pb::ChildRef> = Vec::new();
        let mut buf: Vec<Fragment> = Vec::new();
        let mut buf_bytes = 0u64;
        for id in 0..num_fragments {
            let f = gen_fn(id);
            buf_bytes += node::fragment_logical_bytes(&f);
            buf.push(f);
            if buf_bytes >= target {
                let w = store.write_leaf(&buf).await?;
                io += w.io_bytes;
                layer.push(w.child_ref);
                buf.clear();
                buf_bytes = 0;
            }
        }
        if !buf.is_empty() {
            let w = store.write_leaf(&buf).await?;
            io += w.io_bytes;
            layer.push(w.child_ref);
        }
        let num_leaves = layer.len() as u64;

        // Internal layers until the top fits under fanout.
        while layer.len() as u32 > config.fanout {
            let mut next: Vec<pb::ChildRef> = Vec::new();
            for group in layer.chunks(config.fanout as usize) {
                let w = store.write_internal(group.to_vec(), Vec::new()).await?;
                io += w.io_bytes;
                next.push(w.child_ref);
            }
            layer = next;
        }
        let height = layer.iter().map(|c| c.height).max().unwrap_or(0) + 1;

        let tree = Self {
            store,
            config,
            version: 1,
            children: layer,
            buffer: Vec::new(),
            next_msn: 1,
            schema_pb,
        };
        io += tree.write_root().await?;
        Ok((
            tree,
            BootstrapStats {
                io_write_bytes: io,
                num_leaves,
                height,
            },
        ))
    }

    pub fn height(&self) -> u32 {
        self.children.iter().map(|c| c.height).max().unwrap_or(0) + 1
    }

    fn to_pb_root(&self) -> pb::BeTreeRoot {
        pb::BeTreeRoot {
            version: self.version,
            children: self.children.clone(),
            buffer: self.buffer.clone(),
            next_msn: self.next_msn,
            schema_pb: self.schema_pb.clone(),
            target_node_bytes: self.config.target_node_bytes,
            fanout: self.config.fanout,
        }
    }

    async fn write_root(&self) -> Result<u64> {
        self.store.write_root(&self.to_pb_root()).await
    }

    /// Inject actions into the root buffer (msn-tagged), flush/split/rebalance,
    /// and copy-on-write the touched path + a new root.
    pub async fn commit(&mut self, actions: Vec<pb::FragmentAction>) -> Result<CommitStats> {
        for action in actions {
            self.buffer.push(pb::TaggedAction {
                msn: self.next_msn,
                action: Some(action),
            });
            self.next_msn += 1;
        }
        self.version += 1;

        let mut acc = WriteAcc::default();

        // Flush the root buffer down as far as it will go (root is depth 0).
        let children = std::mem::take(&mut self.children);
        let buffer = std::mem::take(&mut self.buffer);
        let (children, buffer, a) = self.flush_internal(children, buffer, 0).await?;
        acc.add(a);
        self.children = children;
        self.buffer = buffer;

        // Grow: if the root still overflows, split it and lift a new root over the pieces.
        if self.children.len() as u32 > self.config.fanout
            || node::internal_logical_bytes(&self.children, &self.buffer)
                >= self.config.split_ceiling()
        {
            let pieces = node::split_internal(
                std::mem::take(&mut self.children),
                std::mem::take(&mut self.buffer),
                self.config.split_piece_bytes(),
                self.config.fanout,
            );
            let mut new_children = Vec::with_capacity(pieces.len());
            for (ch, buf) in pieces {
                let w = self.store.write_internal(ch, buf).await?;
                acc.io_bytes += w.io_bytes;
                new_children.push(w.child_ref);
            }
            self.children = new_children;
            acc.splits += 1;
        }

        // Coalesce underflowing children (self-balancing on deletes).
        let children = std::mem::take(&mut self.children);
        let (children, a) = self.merge_small_children(children).await?;
        acc.add(a);
        self.children = children;

        // Shrink: a root with a single internal child pulls that child up.
        self.maybe_shrink_root().await?;

        acc.io_bytes += self.write_root().await?;
        Ok(CommitStats {
            io_write_bytes: acc.io_bytes,
            flushes: acc.flushes,
            splits: acc.splits,
            merges: acc.merges,
            height: self.height(),
            max_flush_depth: acc.max_flush_depth,
        })
    }

    /// Flush an internal node's buffer to its children while it overflows,
    /// picking the fullest child each round (gated at `min_flush`). `depth` is
    /// this node's level below the root (0 = root). Returns the (possibly split)
    /// children and the residual buffer.
    fn flush_internal(
        &self,
        mut children: Vec<pb::ChildRef>,
        mut buffer: Vec<pb::TaggedAction>,
        depth: u32,
    ) -> BoxFuture<'_, Result<FlushResult>> {
        Box::pin(async move {
            let mut acc = WriteAcc::default();
            loop {
                if node::internal_logical_bytes(&children, &buffer) < self.config.split_ceiling() {
                    break;
                }
                let mut buckets = node::partition_buffer_by_child(&children, buffer);
                // Fullest child by buffered bytes.
                let (idx, best) = buckets
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, node::buffer_bytes(b)))
                    .max_by_key(|(_, b)| *b)
                    .unwrap_or((0, 0));
                if best < self.config.min_flush_bytes {
                    // No child worth flushing to — reassemble and stop (caller may split).
                    buffer = buckets.into_iter().flatten().collect();
                    break;
                }
                let chosen = std::mem::take(&mut buckets[idx]);
                buffer = buckets.into_iter().flatten().collect();

                let (new_refs, a) = self.ingest(children[idx].clone(), chosen, depth).await?;
                acc.add(a);
                acc.flushes += 1;
                acc.max_flush_depth = acc.max_flush_depth.max(depth);
                children.splice(idx..idx + 1, new_refs);
            }
            Ok((children, buffer, acc))
        })
    }

    /// Push `incoming` messages into the subtree rooted at `child` (at `depth`
    /// below the root); apply at a leaf, recurse+buffer at an internal node;
    /// split on overflow. Returns the child ref(s) that now represent the subtree.
    fn ingest(
        &self,
        child: pb::ChildRef,
        incoming: Vec<pb::TaggedAction>,
        depth: u32,
    ) -> BoxFuture<'_, Result<IngestResult>> {
        Box::pin(async move {
            let mut acc = WriteAcc::default();
            if child.height == 0 {
                // Leaf: apply messages, then split if it overflows.
                let fragments = self.store.read_leaf(&child).await?;
                let mut map: BTreeMap<u64, Fragment> =
                    fragments.into_iter().map(|f| (f.id, f)).collect();
                node::apply_actions(&mut map, incoming)?;
                let new_frags: Vec<Fragment> = map.into_values().collect();

                if node::leaf_logical_bytes(&new_frags) >= self.config.split_ceiling() {
                    let mut refs = Vec::new();
                    for piece in
                        node::split_leaf_fragments(new_frags, self.config.split_piece_bytes())
                    {
                        let w = self.store.write_leaf(&piece).await?;
                        acc.io_bytes += w.io_bytes;
                        refs.push(w.child_ref);
                    }
                    acc.splits += 1;
                    Ok((refs, acc))
                } else {
                    let w = self.store.write_leaf(&new_frags).await?;
                    acc.io_bytes += w.io_bytes;
                    Ok((vec![w.child_ref], acc))
                }
            } else {
                // Internal: buffer, recurse-flush, split if it overflows.
                let InternalNode {
                    children,
                    mut buffer,
                } = self.store.read_internal(&child).await?;
                buffer.extend(incoming);
                // This node is one level deeper than the parent that flushed to it.
                let (children, buffer, a) =
                    self.flush_internal(children, buffer, depth + 1).await?;
                acc.add(a);
                // Rebalance: coalesce any underflowing children before checking split.
                let (children, a) = self.merge_small_children(children).await?;
                acc.add(a);

                if children.len() as u32 > self.config.fanout
                    || node::internal_logical_bytes(&children, &buffer)
                        >= self.config.split_ceiling()
                {
                    let mut refs = Vec::new();
                    for (ch, buf) in node::split_internal(
                        children,
                        buffer,
                        self.config.split_piece_bytes(),
                        self.config.fanout,
                    ) {
                        let w = self.store.write_internal(ch, buf).await?;
                        acc.io_bytes += w.io_bytes;
                        refs.push(w.child_ref);
                    }
                    acc.splits += 1;
                    Ok((refs, acc))
                } else {
                    let w = self.store.write_internal(children, buffer).await?;
                    acc.io_bytes += w.io_bytes;
                    Ok((vec![w.child_ref], acc))
                }
            }
        })
    }

    /// Coalesce runs of adjacent children when one underflows (leaf ≤ 0.25 B;
    /// internal < fanout/4 children), bounded so the merged node stays valid
    /// (leaves ≤ 0.6 B; internal ≤ fanout children). Reads/writes the merged
    /// node(s). Leaves concat fragments; internal nodes concat children + buffers.
    async fn merge_small_children(
        &self,
        children: Vec<pb::ChildRef>,
    ) -> Result<(Vec<pb::ChildRef>, WriteAcc)> {
        let mut acc = WriteAcc::default();
        let mut out: Vec<pb::ChildRef> = Vec::with_capacity(children.len());
        let mut i = 0;
        while i < children.len() {
            if !node::is_underflow(&children[i], &self.config) {
                out.push(children[i].clone());
                i += 1;
                continue;
            }
            // Grow a coalesce group with adjacent siblings, bounded by node kind.
            let is_leaf = children[i].height == 0;
            let mut group = vec![children[i].clone()];
            let mut bytes = children[i].byte_size;
            let mut fan = children[i].fanout_used;
            let mut j = i + 1;
            while j < children.len() {
                let c = &children[j];
                let fits = if is_leaf {
                    bytes + c.byte_size <= self.config.coalesce_ceiling()
                } else {
                    fan + c.fanout_used <= self.config.fanout
                };
                if !fits {
                    break;
                }
                bytes += c.byte_size;
                fan += c.fanout_used;
                group.push(c.clone());
                j += 1;
            }
            if group.len() == 1 {
                out.push(group.pop().unwrap());
            } else {
                let (merged, a) = self.coalesce(group).await?;
                acc.add(a);
                acc.merges += 1;
                out.push(merged);
            }
            i = j;
        }
        Ok((out, acc))
    }

    /// Combine an adjacent group of same-height children into one node.
    async fn coalesce(&self, group: Vec<pb::ChildRef>) -> Result<(pb::ChildRef, WriteAcc)> {
        let mut acc = WriteAcc::default();
        if group[0].height == 0 {
            let mut fragments: Vec<Fragment> = Vec::new();
            for c in &group {
                fragments.extend(self.store.read_leaf(c).await?);
            }
            fragments.sort_by_key(|f| f.id);
            let w = self.store.write_leaf(&fragments).await?;
            acc.io_bytes += w.io_bytes;
            Ok((w.child_ref, acc))
        } else {
            let mut children: Vec<pb::ChildRef> = Vec::new();
            let mut buffer: Vec<pb::TaggedAction> = Vec::new();
            for c in &group {
                let node = self.store.read_internal(c).await?;
                children.extend(node.children);
                buffer.extend(node.buffer);
            }
            let w = self.store.write_internal(children, buffer).await?;
            acc.io_bytes += w.io_bytes;
            Ok((w.child_ref, acc))
        }
    }

    /// If the root has a single internal child, pull that child's children/buffer
    /// up into the root (height −1).
    async fn maybe_shrink_root(&mut self) -> Result<()> {
        while self.children.len() == 1 && self.children[0].height > 0 {
            let node = self.store.read_internal(&self.children[0]).await?;
            self.children = node.children;
            let mut buffer = node.buffer;
            buffer.append(&mut self.buffer);
            self.buffer = buffer;
        }
        Ok(())
    }

    /// Materialize the full fragment list: collect all leaf fragments + every
    /// buffered action in the tree, then overlay actions newest-wins.
    pub async fn materialize(&self) -> Result<Vec<Fragment>> {
        let mut map: BTreeMap<u64, Fragment> = BTreeMap::new();
        let mut actions: Vec<pb::TaggedAction> = self.buffer.clone();
        self.collect(&self.children, &mut map, &mut actions).await?;
        node::apply_actions(&mut map, actions)?;
        Ok(map.into_values().collect())
    }

    /// Walk the tree and collect `(height, logical_bytes)` for every internal node
    /// (root included). Used to measure how full internal ε-buffers are — a node
    /// near `B` is holding a big buffer, a node near its ref-only size is "cold".
    pub async fn internal_node_sizes(&self) -> Result<Vec<(u32, u64)>> {
        let mut out = vec![(
            self.height(),
            node::internal_logical_bytes(&self.children, &self.buffer),
        )];
        self.collect_internal_sizes(self.children.clone(), &mut out)
            .await?;
        Ok(out)
    }

    fn collect_internal_sizes<'a>(
        &'a self,
        children: Vec<pb::ChildRef>,
        out: &'a mut Vec<(u32, u64)>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for c in &children {
                if c.height > 0 {
                    out.push((c.height, c.byte_size));
                    let node = self.store.read_internal(c).await?;
                    self.collect_internal_sizes(node.children, out).await?;
                }
            }
            Ok(())
        })
    }

    fn collect<'a>(
        &'a self,
        children: &'a [pb::ChildRef],
        frags: &'a mut BTreeMap<u64, Fragment>,
        actions: &'a mut Vec<pb::TaggedAction>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for child in children {
                if child.height == 0 {
                    for f in self.store.read_leaf(child).await? {
                        frags.insert(f.id, f);
                    }
                } else {
                    let node = self.store.read_internal(child).await?;
                    actions.extend(node.buffer);
                    self.collect_owned(node.children, frags, actions).await?;
                }
            }
            Ok(())
        })
    }

    fn collect_owned<'a>(
        &'a self,
        children: Vec<pb::ChildRef>,
        frags: &'a mut BTreeMap<u64, Fragment>,
        actions: &'a mut Vec<pb::TaggedAction>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for child in &children {
                if child.height == 0 {
                    for f in self.store.read_leaf(child).await? {
                        frags.insert(f.id, f);
                    }
                } else {
                    let node = self.store.read_internal(child).await?;
                    actions.extend(node.buffer);
                    self.collect_owned(node.children, frags, actions).await?;
                }
            }
            Ok(())
        })
    }

    /// Cold open from storage: read the latest root and materialize.
    pub async fn cold_open(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
    ) -> Result<Vec<Fragment>> {
        let store = NodeStore::new(object_store, base, scheduler, cache);
        let version = store.read_latest_version().await?;
        let root = store.read_root(version).await?;
        let tree = Self {
            store,
            config: BeTreeConfig {
                target_node_bytes: root.target_node_bytes,
                fanout: root.fanout,
                min_flush_bytes: root.target_node_bytes / 16,
            },
            version: root.version,
            children: root.children,
            buffer: root.buffer,
            next_msn: root.next_msn,
            schema_pb: root.schema_pb,
        };
        tree.materialize().await
    }
}
