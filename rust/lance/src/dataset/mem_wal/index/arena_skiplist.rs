// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Single-writer, lock-free-read skiplist with no epoch reclamation.
//!
//! Purpose-built for the MemTable scalar index, whose access pattern is:
//! append-only (no per-entry delete), a single writer (serialized by the
//! `ShardWriter` actor), and many concurrent readers. Under that pattern there
//! is nothing to reclaim until the whole index is dropped, so — unlike a
//! general-purpose concurrent skiplist (`crossbeam_skiplist`) — we pay **no
//! epoch pin** on reads. Profiling showed crossbeam's per-operation epoch
//! pinning (`try_pin_loop`) dominating the point-lookup hot path and, worse,
//! contending across threads (the N-thread read-scaling bottleneck). This
//! mirrors RocksDB's arena `InlineSkipList`: nodes live for the index's whole
//! life, readers only do `Acquire` loads, the writer publishes with `Release`.
//!
//! # Safety model
//! - **Single writer.** Only [`SkipListWriter`] mutates; it holds the sole
//!   `&mut`. Callers must serialize writes (the MemTable does so via the actor;
//!   the BTree index additionally guards the writer behind a `Mutex`).
//! - **No free before drop.** Nodes are owned by `SkipListCore` and dropped only
//!   when the core (the whole index generation) is dropped, at which point there
//!   are no live readers. So a reader can never observe a freed node.
//! - **Pointer stability.** Nodes live behind `Box`; the owning `Vec<Box<Node>>`
//!   may reallocate, but the `Box` heap allocations (and thus the `*const Node`
//!   that readers follow) never move.
//! - **Publish/consume.** The writer initializes a node fully, then links it in
//!   with `Release` stores; readers follow links with `Acquire` loads, so a
//!   reader that sees a pointer also sees the fully-initialized node.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Maximum tower height. 16 levels with p=1/4 supports ~4^16 ≈ 4 billion
/// entries before degrading, far beyond any single MemTable generation.
const MAX_HEIGHT: usize = 16;
/// Inverse promotion probability (p = 1/4): a node grows one level with prob
/// 1/4. Matches RocksDB's default `kBranching`.
const BRANCHING: u64 = 4;

/// A skiplist node: a key plus a variable-length tower of forward pointers.
///
/// `next[i]` is the successor at level `i`. The tower length is the node's
/// height. Stored in an `AtomicPtr` so the writer can publish (`Release`) and
/// readers can consume (`Acquire`) without a lock.
struct Node<K> {
    key: K,
    next: Box<[AtomicPtr<Self>]>,
}

/// Shared, append-only core. Owns every node for the index's lifetime.
///
/// `head` is a bare tower (no key) acting as the before-first sentinel; a null
/// node pointer in a traversal means "at head". `nodes` is touched only by the
/// single writer; readers never read it (they only follow `next` pointers).
struct SkipListCore<K> {
    /// Forward pointers out of the head sentinel, one per level (`MAX_HEIGHT`).
    head: Box<[AtomicPtr<Node<K>>]>,
    /// Owns all nodes so they outlive every reader. Writer-only access.
    nodes: UnsafeCell<Vec<Box<Node<K>>>>,
    /// Highest tower level currently in use (1..=MAX_HEIGHT).
    height: AtomicUsize,
    /// Number of entries.
    len: AtomicUsize,
}

// SAFETY: `nodes` (the only non-Sync field) is mutated exclusively by the single
// writer; readers never access it. Reader/writer interaction on `head`/`next`
// goes through atomics with Acquire/Release. `K: Send + Sync` covers the keys
// shared with readers.
unsafe impl<K: Send + Sync> Send for SkipListCore<K> {}
unsafe impl<K: Send + Sync> Sync for SkipListCore<K> {}

impl<K> SkipListCore<K> {
    fn new() -> Self {
        let head = (0..MAX_HEIGHT)
            .map(|_| AtomicPtr::new(ptr::null_mut()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            head,
            nodes: UnsafeCell::new(Vec::new()),
            height: AtomicUsize::new(1),
            len: AtomicUsize::new(0),
        }
    }

    /// The forward-pointer slot at `level` for `node` (null `node` = head).
    #[inline]
    fn next_slot(&self, node: *const Node<K>, level: usize) -> &AtomicPtr<Node<K>> {
        if node.is_null() {
            &self.head[level]
        } else {
            // SAFETY: `node` is non-null and points to a live node owned by
            // `self.nodes` (never freed before `self` drops). `level` is < the
            // node's height at every call site (search descends within height).
            unsafe { &(*node).next[level] }
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }
}

/// Create a paired writer and reader over a fresh, empty skiplist core.
pub fn new_skiplist<K: Ord + Send + Sync>() -> (SkipListWriter<K>, SkipListReader<K>) {
    let core = Arc::new(SkipListCore::new());
    let writer = SkipListWriter {
        core: core.clone(),
        // Nonzero xorshift seed; the writer is single-threaded so a private,
        // deterministic RNG is fine (and keeps tests reproducible).
        rng: 0x9E3779B97F4A7C15,
    };
    let reader = SkipListReader { core };
    (writer, reader)
}

/// The sole mutator of a skiplist. Not `Sync`: only one writer may exist.
pub struct SkipListWriter<K> {
    core: Arc<SkipListCore<K>>,
    rng: u64,
}

impl<K: Ord> SkipListWriter<K> {
    /// Geometric height with p = 1/BRANCHING, capped at `MAX_HEIGHT`.
    fn random_height(&mut self) -> usize {
        // xorshift64
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        let mut height = 1;
        while height < MAX_HEIGHT && x.is_multiple_of(BRANCHING) {
            height += 1;
            x /= BRANCHING;
        }
        height
    }

    /// Insert `key`. Keys must be unique (the MemTable key carries a row
    /// position, making every entry distinct); equal keys are not expected.
    pub fn insert(&mut self, key: K) {
        let cur_height = self.core.height.load(Ordering::Relaxed);

        // Find the predecessor at every level. For levels at/above the current
        // height the predecessor is the head (null); the descent never enters
        // those levels, leaving `pred` null there.
        let mut preds: [*const Node<K>; MAX_HEIGHT] = [ptr::null(); MAX_HEIGHT];
        let mut pred: *const Node<K> = ptr::null();
        for level in (0..MAX_HEIGHT).rev() {
            if level < cur_height {
                loop {
                    let next = self.core.next_slot(pred, level).load(Ordering::Acquire);
                    if !next.is_null() && unsafe { (*next).key < key } {
                        pred = next;
                    } else {
                        break;
                    }
                }
            }
            preds[level] = pred;
        }

        let height = self.random_height();

        // Build the node fully (key + successors) before publishing. Successors
        // are stable: the single writer is the only mutator, so no link changes
        // between read and publish.
        let mut tower: Vec<AtomicPtr<Node<K>>> = Vec::with_capacity(height);
        for (level, pred) in preds.iter().enumerate().take(height) {
            let succ = self.core.next_slot(*pred, level).load(Ordering::Acquire);
            tower.push(AtomicPtr::new(succ));
        }

        let node = Box::new(Node {
            key,
            next: tower.into_boxed_slice(),
        });
        let node_ptr: *mut Node<K> = &*node as *const Node<K> as *mut Node<K>;
        // Hand ownership to the core; the `Box` heap stays put even if the Vec
        // reallocates, so `node_ptr` remains valid for readers.
        // SAFETY: single-writer exclusive access to `nodes`.
        unsafe { (*self.core.nodes.get()).push(node) };

        // Advertise the taller height before linking the top levels: a reader
        // that sees the new height but not yet a top link just finds a null
        // there and descends — still correct.
        if height > cur_height {
            self.core.height.store(height, Ordering::Release);
        }

        // Publish: splice the node in at each level with Release so a reader
        // that loads the pointer also sees the initialized node.
        for (level, pred) in preds.iter().enumerate().take(height) {
            self.core
                .next_slot(*pred, level)
                .store(node_ptr, Ordering::Release);
        }

        self.core.len.fetch_add(1, Ordering::Release);
    }
}

/// A lock-free, pin-free reader. Cheaply cloned and shared across threads.
#[derive(Clone)]
pub struct SkipListReader<K> {
    core: Arc<SkipListCore<K>>,
}

impl<K: Ord> SkipListReader<K> {
    /// Greatest node with `key <= target`, mapped through `f` while it is alive.
    /// Equivalent to crossbeam's `upper_bound(Included(target))`. `None` if no
    /// such node. The closure avoids cloning the key on the hot path.
    pub fn upper_bound_with<R>(&self, target: &K, f: impl FnOnce(&K) -> R) -> Option<R> {
        let node = self.find_le(target);
        if node.is_null() {
            None
        } else {
            // SAFETY: non-null node owned by the core, alive for this call.
            Some(f(unsafe { &(*node).key }))
        }
    }

    /// Greatest node with `key <= target`, or null. Descends the tower.
    fn find_le(&self, target: &K) -> *const Node<K> {
        let height = self.core.height.load(Ordering::Acquire);
        let mut pred: *const Node<K> = ptr::null();
        for level in (0..height).rev() {
            loop {
                let next = self.core.next_slot(pred, level).load(Ordering::Acquire);
                if !next.is_null() && unsafe { (*next).key <= *target } {
                    pred = next;
                } else {
                    break;
                }
            }
        }
        pred
    }

    /// First node with `key >= start`, or null.
    fn lower_bound(&self, start: &K) -> *const Node<K> {
        let height = self.core.height.load(Ordering::Acquire);
        let mut pred: *const Node<K> = ptr::null();
        for level in (0..height).rev() {
            loop {
                let next = self.core.next_slot(pred, level).load(Ordering::Acquire);
                if !next.is_null() && unsafe { (*next).key < *start } {
                    pred = next;
                } else {
                    break;
                }
            }
        }
        self.core.next_slot(pred, 0).load(Ordering::Acquire)
    }

    /// Iterate all keys in ascending order.
    pub fn iter(&self) -> Iter<'_, K> {
        Iter {
            node: self.core.head[0].load(Ordering::Acquire),
            _marker: PhantomData,
        }
    }

    /// Iterate keys in ascending order starting at the first `key >= start`.
    pub fn range_from(&self, start: &K) -> Iter<'_, K> {
        Iter {
            node: self.lower_bound(start),
            _marker: PhantomData,
        }
    }

    /// The smallest key, mapped through `f`, or `None` if empty.
    pub fn front_with<R>(&self, f: impl FnOnce(&K) -> R) -> Option<R> {
        let node = self.core.head[0].load(Ordering::Acquire);
        if node.is_null() {
            None
        } else {
            // SAFETY: non-null node owned by the core, alive for this call.
            Some(f(unsafe { &(*node).key }))
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.core.len()
    }

    /// Whether the index has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Forward iterator over keys in ascending order. Yields `&K` borrowed from the
/// reader: nodes are never freed while the core (and thus the reader) is alive.
pub struct Iter<'a, K> {
    node: *const Node<K>,
    _marker: PhantomData<&'a SkipListReader<K>>,
}

impl<'a, K> Iterator for Iter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        if self.node.is_null() {
            return None;
        }
        // SAFETY: non-null node owned by the core; the borrow lifetime `'a` is
        // bounded by the reader, which keeps the core (and node) alive.
        let node = unsafe { &*self.node };
        self.node = node.next[0].load(Ordering::Acquire);
        Some(&node.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    fn collect(reader: &SkipListReader<i64>) -> Vec<i64> {
        reader.iter().copied().collect()
    }

    #[test]
    fn test_insert_keeps_sorted_order() {
        let (mut w, r) = new_skiplist::<i64>();
        for k in [5, 1, 9, 3, 7, 2, 8, 4, 6, 0] {
            w.insert(k);
        }
        assert_eq!(collect(&r), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(r.len(), 10);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_empty() {
        let (_w, r) = new_skiplist::<i64>();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(collect(&r), Vec::<i64>::new());
        assert_eq!(r.upper_bound_with(&5, |k| *k), None);
        assert_eq!(r.front_with(|k| *k), None);
        assert_eq!(r.range_from(&0).count(), 0);
    }

    #[test]
    fn test_upper_bound_le() {
        let (mut w, r) = new_skiplist::<i64>();
        for k in [10, 20, 30, 40] {
            w.insert(k);
        }
        // Exact hits.
        assert_eq!(r.upper_bound_with(&20, |k| *k), Some(20));
        assert_eq!(r.upper_bound_with(&40, |k| *k), Some(40));
        // Between keys → greatest <= target.
        assert_eq!(r.upper_bound_with(&25, |k| *k), Some(20));
        assert_eq!(r.upper_bound_with(&39, |k| *k), Some(30));
        // Above all.
        assert_eq!(r.upper_bound_with(&999, |k| *k), Some(40));
        // Below all → None.
        assert_eq!(r.upper_bound_with(&5, |k| *k), None);
    }

    #[test]
    fn test_front_and_range_from() {
        let (mut w, r) = new_skiplist::<i64>();
        for k in [3, 1, 4, 1_000, 2] {
            w.insert(k);
        }
        assert_eq!(r.front_with(|k| *k), Some(1));
        assert_eq!(
            r.range_from(&3).copied().collect::<Vec<_>>(),
            vec![3, 4, 1_000]
        );
        // start below first → all; start above last → empty.
        assert_eq!(
            r.range_from(&0).copied().collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 1_000]
        );
        assert_eq!(r.range_from(&2_000).count(), 0);
        // start between → from first >= start.
        assert_eq!(r.range_from(&5).copied().collect::<Vec<_>>(), vec![1_000]);
    }

    #[test]
    fn test_composite_key_dup_values() {
        // Mirrors IndexKey = (value, position): same value, distinct positions.
        let (mut w, r) = new_skiplist::<(i64, u64)>();
        for key in [(7, 0), (3, 1), (7, 2), (3, 0), (7, 1)] {
            w.insert(key);
        }
        let all: Vec<_> = r.iter().copied().collect();
        assert_eq!(all, vec![(3, 0), (3, 1), (7, 0), (7, 1), (7, 2)]);
        // Newest visible position for value 7 with watermark 1 = (7,1).
        assert_eq!(r.upper_bound_with(&(7, 1), |k| *k), Some((7, 1)));
        // Watermark below all of value 3 → falls back to a smaller value.
        assert_eq!(r.upper_bound_with(&(3, 5), |k| *k), Some((3, 1)));
    }

    #[test]
    fn test_concurrent_single_writer_many_readers() {
        // 1 writer inserting 0..N while readers continuously seek. Asserts:
        // every value a reader observes is one the writer has inserted, the
        // observed prefix is contiguous and monotonically non-decreasing (no
        // torn/lost nodes), and the final state is complete and sorted.
        const N: i64 = 50_000;
        let (mut w, r) = new_skiplist::<i64>();
        let done = Arc::new(AtomicBool::new(false));

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let r = r.clone();
                let done = done.clone();
                thread::spawn(move || {
                    let mut max_seen = -1;
                    while !done.load(Ordering::Acquire) {
                        // Largest key present <= N is a contiguous prefix max.
                        if let Some(top) = r.upper_bound_with(&N, |k| *k) {
                            assert!((0..N).contains(&top) || top == N - 1);
                            assert!(top >= max_seen, "visibility went backwards");
                            max_seen = top;
                            // Every key up to `top` must be present (contiguous).
                            assert!(r.upper_bound_with(&top, |k| *k) == Some(top));
                        }
                    }
                    max_seen
                })
            })
            .collect();

        for k in 0..N {
            w.insert(k);
        }
        done.store(true, Ordering::Release);
        for h in readers {
            h.join().unwrap();
        }

        assert_eq!(r.len(), N as usize);
        assert_eq!(collect(&r), (0..N).collect::<Vec<_>>());
    }
}
