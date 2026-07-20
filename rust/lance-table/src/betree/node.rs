// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): in-memory Bε-tree nodes and pure node logic.
//!
//! Two node kinds: [`InternalNode`] (child pivots + a message ε-buffer) and
//! [`LeafNode`] (the fragment table). All sizing/split/merge thresholds are in
//! **serialized bytes** because actions and fragments are variable-sized. No IO
//! here — see `store.rs` (node files) and `tree.rs` (algorithms).

use std::collections::BTreeMap;

use prost::Message;

use crate::betree::action;
use crate::format::Fragment;
use crate::format::pb::{self, fragment_action::Action};
use lance_core::{Error, Result};

pub const DEFAULT_TARGET_NODE_BYTES: u64 = 10 * 1024 * 1024;
pub const DEFAULT_FANOUT: u32 = 16;

/// Bε-tree tuning knobs. The two independent knobs are the target node size `B`
/// and the `fanout` (= B^ε); everything else — the flush gate, split pieces,
/// merge floor — derives from them (byte-based, per the literature: ε=1/2,
/// ~10 MiB nodes, fanout 16, split ≥ B into ~0.5 B pieces, merge ≤ 0.25 B).
#[derive(Debug, Clone)]
pub struct BeTreeConfig {
    /// Target node size `B`.
    pub target_node_bytes: u64,
    /// Max children of an internal node before it splits (`B^ε`).
    pub fanout: u32,
    /// Optional override of the flush gate. `None` (the norm) derives it as
    /// `B/fanout` — see [`Self::min_flush_bytes`].
    pub min_flush_override: Option<u64>,
}

impl Default for BeTreeConfig {
    fn default() -> Self {
        Self {
            target_node_bytes: DEFAULT_TARGET_NODE_BYTES,
            fanout: DEFAULT_FANOUT,
            min_flush_override: None,
        }
    }
}

impl BeTreeConfig {
    /// Build a config from the two primary knobs (`B`, `fanout`); the flush gate
    /// derives as `B/fanout`.
    pub fn new(target_node_bytes: u64, fanout: u32) -> Self {
        Self {
            target_node_bytes,
            fanout,
            min_flush_override: None,
        }
    }

    /// A node splits when its logical bytes reach this.
    pub fn split_ceiling(&self) -> u64 {
        self.target_node_bytes
    }
    /// Split output pieces target ~0.5 B (so a just-split node isn't near-full).
    pub fn split_piece_bytes(&self) -> u64 {
        self.target_node_bytes / 2
    }
    /// A node underflows (is a merge candidate) at ≤ 0.25 B.
    pub fn merge_floor(&self) -> u64 {
        self.target_node_bytes / 4
    }
    /// Adjacent children are coalesced while their combined bytes stay ≤ 0.6 B.
    pub fn coalesce_ceiling(&self) -> u64 {
        self.target_node_bytes * 3 / 5
    }
    /// The amortization gate: never flush a child slice smaller than this.
    ///
    /// Derived as `B/fanout` — a full ε-buffer (~B) split across `fanout`
    /// children leaves ~B/fanout in the fullest, so this is the natural "fair
    /// share" threshold and it scales correctly with fanout. (A hardcoded B/16
    /// silently assumed fanout 16: at higher fanout the buffer spreads thinner
    /// than the gate, so flushes never fire and the tree degrades to split-only.)
    pub fn min_flush_bytes(&self) -> u64 {
        self.min_flush_override
            .unwrap_or_else(|| self.target_node_bytes / self.fanout.max(1) as u64)
    }
}

/// A leaf: the fragment table, sorted by fragment id.
#[derive(Debug, Clone)]
pub struct LeafNode {
    pub fragments: Vec<Fragment>,
}

/// An internal node: child pivots (sorted by `min_key`, contiguous) + the
/// ε-buffer. The buffer is kept in msn (insertion) order in memory and sorted by
/// `(key, msn)` only when grouping for a flush.
#[derive(Debug, Clone, Default)]
pub struct InternalNode {
    pub children: Vec<pb::ChildRef>,
    pub buffer: Vec<pb::TaggedAction>,
}

/// Logical (uncompressed) byte size of a single fragment — the split/merge unit.
pub fn fragment_logical_bytes(fragment: &Fragment) -> u64 {
    pb::DataFragment::from(fragment).encoded_len() as u64
}

/// Logical (uncompressed) byte size of a leaf — the split/merge metric.
pub fn leaf_logical_bytes(fragments: &[Fragment]) -> u64 {
    fragments.iter().map(fragment_logical_bytes).sum()
}

/// Logical byte size of an internal node = its encoded protobuf size.
pub fn internal_logical_bytes(children: &[pb::ChildRef], buffer: &[pb::TaggedAction]) -> u64 {
    let node = pb::InternalNode {
        children: children.to_vec(),
        buffer: buffer.to_vec(),
    };
    node.encoded_len() as u64
}

/// The target key of a buffered action (the fragment id it mutates).
pub fn action_key(t: &pb::TaggedAction) -> u64 {
    t.action
        .as_ref()
        .and_then(action::target_frag_id)
        .unwrap_or(0)
}

/// Index of the child owning `key`. `children` are sorted by `min_key` with
/// contiguous ranges, so this is the rightmost child with `min_key <= key`.
pub fn child_index_for(children: &[pb::ChildRef], key: u64) -> usize {
    match children.binary_search_by(|c| c.min_key.cmp(&key)) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i - 1,
    }
}

/// Build a `ChildRef` for a leaf that has just been written.
pub fn leaf_ref(node_path: String, fragments: &[Fragment], byte_size: u64) -> pb::ChildRef {
    pb::ChildRef {
        node_path,
        min_key: fragments.first().map(|f| f.id).unwrap_or(0),
        max_key: fragments.last().map(|f| f.id).unwrap_or(0),
        num_keys: fragments.len() as u64,
        byte_size,
        height: 0,
        fanout_used: 0,
    }
}

/// Build a `ChildRef` for an internal node that has just been written.
pub fn internal_ref(node_path: String, children: &[pb::ChildRef], byte_size: u64) -> pb::ChildRef {
    let num_keys = children.iter().map(|c| c.num_keys).sum();
    let height = children.iter().map(|c| c.height).max().unwrap_or(0) + 1;
    pb::ChildRef {
        node_path,
        min_key: children.first().map(|c| c.min_key).unwrap_or(0),
        max_key: children.last().map(|c| c.max_key).unwrap_or(0),
        num_keys,
        byte_size,
        height,
        fanout_used: children.len() as u32,
    }
}

/// Is this child underflowing (a merge candidate)? Leaves by bytes (≤ 0.25 B),
/// internal nodes by direct-child count (< fanout/4).
pub fn is_underflow(child: &pb::ChildRef, config: &BeTreeConfig) -> bool {
    if child.height == 0 {
        child.byte_size <= config.merge_floor()
    } else {
        child.fanout_used < (config.fanout / 4).max(1)
    }
}

/// Apply buffered actions to an id-keyed fragment map, in `(key, msn)` order
/// (newest-wins). Used at leaves and when materializing.
pub fn apply_actions(
    frags: &mut BTreeMap<u64, Fragment>,
    mut actions: Vec<pb::TaggedAction>,
) -> Result<()> {
    actions.sort_by_key(|t| (action_key(t), t.msn));
    for tagged in actions {
        if let Some(action) = tagged.action {
            apply_one(frags, action)?;
        }
    }
    Ok(())
}

fn apply_one(frags: &mut BTreeMap<u64, Fragment>, action: pb::FragmentAction) -> Result<()> {
    let Some(action) = action.action else {
        return Ok(());
    };
    match action {
        Action::AddFragment(f) => {
            let fragment = Fragment::try_from(f)?;
            frags.insert(fragment.id, fragment);
        }
        Action::RemoveFragment(id) => {
            frags.remove(&id);
        }
        Action::AddDataFile(a) => {
            let file = crate::format::DataFile::try_from(
                a.file
                    .ok_or_else(|| Error::invalid_input("AddDataFile action missing file"))?,
            )?;
            if let Some(fragment) = frags.get_mut(&a.frag_id) {
                fragment.files.push(file);
            }
        }
        Action::RemoveDataFile(a) => {
            if let Some(fragment) = frags.get_mut(&a.frag_id) {
                fragment.files.retain(|f| f.path != a.path);
            }
        }
        Action::AddDeletionFile(a) => {
            if let (Some(df), Some(fragment)) = (a.deletion_file, frags.get_mut(&a.frag_id)) {
                fragment.deletion_file = Some(crate::format::DeletionFile::try_from(df)?);
            }
        }
    }
    Ok(())
}

/// Partition a buffer into one bucket per child (by owning child index). The
/// returned vec has `children.len()` buckets; the buffer is consumed.
pub fn partition_buffer_by_child(
    children: &[pb::ChildRef],
    buffer: Vec<pb::TaggedAction>,
) -> Vec<Vec<pb::TaggedAction>> {
    let mut buckets: Vec<Vec<pb::TaggedAction>> = vec![Vec::new(); children.len()];
    for tagged in buffer {
        let idx = child_index_for(children, action_key(&tagged));
        buckets[idx].push(tagged);
    }
    buckets
}

/// Sum of encoded bytes of a set of buffered actions.
pub fn buffer_bytes(buffer: &[pb::TaggedAction]) -> u64 {
    buffer.iter().map(|t| t.encoded_len() as u64).sum()
}

/// Split a sorted fragment list into `⌈total/piece_bytes⌉` contiguous pieces of
/// roughly equal bytes (~0.5 B each — no tiny tail, so no split-then-merge churn).
pub fn split_leaf_fragments(fragments: Vec<Fragment>, piece_bytes: u64) -> Vec<Vec<Fragment>> {
    let total = leaf_logical_bytes(&fragments);
    let num_pieces = total.div_ceil(piece_bytes.max(1)).max(1) as usize;
    if num_pieces <= 1 || fragments.len() <= 1 {
        return vec![fragments];
    }
    let target = total / num_pieces as u64;
    let mut pieces = Vec::with_capacity(num_pieces);
    let mut cur: Vec<Fragment> = Vec::new();
    let mut cur_bytes = 0u64;
    for f in fragments {
        let fb = pb::DataFragment::from(&f).encoded_len() as u64;
        cur.push(f);
        cur_bytes += fb;
        if cur_bytes >= target && pieces.len() + 1 < num_pieces {
            pieces.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    pieces
}

/// Split an internal node's (children, buffer) into contiguous pieces of roughly
/// equal size (≤ `piece_bytes` and ≤ `fanout` children each). The buffer follows
/// its child by key range. Used when an internal node overflows.
pub fn split_internal(
    children: Vec<pb::ChildRef>,
    buffer: Vec<pb::TaggedAction>,
    piece_bytes: u64,
    fanout: u32,
) -> Vec<(Vec<pb::ChildRef>, Vec<pb::TaggedAction>)> {
    // Enough pieces to satisfy both the byte and fanout ceilings, then cut evenly.
    let total: u64 = children.iter().map(|c| c.byte_size).sum();
    let by_bytes = total.div_ceil(piece_bytes.max(1));
    let by_fanout = (children.len() as u64).div_ceil(fanout.max(1) as u64);
    let num_pieces = by_bytes.max(by_fanout).max(1) as usize;
    let target_bytes = total / num_pieces as u64;
    let target_count = children.len().div_ceil(num_pieces);

    let mut groups: Vec<Vec<pb::ChildRef>> = Vec::new();
    let mut cur: Vec<pb::ChildRef> = Vec::new();
    let mut cur_bytes = 0u64;
    for c in children {
        cur_bytes += c.byte_size;
        cur.push(c);
        if (cur_bytes >= target_bytes || cur.len() >= target_count) && groups.len() + 1 < num_pieces
        {
            groups.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
    }
    if !cur.is_empty() {
        groups.push(cur);
    }

    // Route each buffered action to the group whose key range contains it.
    let group_bounds: Vec<u64> = groups.iter().map(|g| g[0].min_key).collect();
    let mut group_buffers: Vec<Vec<pb::TaggedAction>> = vec![Vec::new(); groups.len()];
    for tagged in buffer {
        let key = action_key(&tagged);
        let gi = match group_bounds.binary_search(&key) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        group_buffers[gi].push(tagged);
    }

    groups.into_iter().zip(group_buffers).collect()
}
