// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Minimal communicator abstraction for distributed training

use std::sync::{Arc, Condvar, Mutex};

/// Abstraction for basic collectives needed by distributed KMeans
pub trait Communicator: Send + Sync {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;

    /// Barrier across all workers
    fn barrier(&self);

    /// Allreduce sums of centroid accumulators and counts
    /// Input: local_sums[k][dim], local_counts[k]
    /// Output: global sums & counts
    fn allreduce_sums_counts(
        &self,
        local_sums: &[Vec<f32>],
        local_counts: &[usize],
    ) -> (Vec<Vec<f32>>, Vec<usize>);

    /// Broadcast centroids from root to all workers (in-place update)
    fn bcast_centroids(&self, centroids: &mut Vec<Vec<f32>>, root: usize);
}

/// Shared state for local in-process communicator
#[derive(Default)]
struct SharedState {
    // Barrier state
    barrier_count: usize,
    barrier_generation: u64,
    // Allreduce state
    ar_pending_count: usize,
    ar_acc_sums: Option<Vec<Vec<f32>>>,
    ar_acc_counts: Option<Vec<usize>>,
    ar_result_ready: bool,
    ar_waiters_remaining: usize,
    ar_round_gen: u64,
    // Broadcast state
    bcast_ready: bool,
    bcast_data: Option<Vec<Vec<f32>>>,
    bcast_waiters_remaining: usize,
    bcast_round_gen: u64,
}

/// Local multi-worker communicator for tests / single-process simulation
pub struct LocalCommunicator {
    rank: usize,
    world: usize,
    state: Arc<Mutex<SharedState>>,
    barrier_cv: Arc<Condvar>,
    ar_cv: Arc<Condvar>,
    bcast_cv: Arc<Condvar>,
}

impl LocalCommunicator {
    pub fn new_group(world: usize) -> Vec<Arc<dyn Communicator>> {
        let state = Arc::new(Mutex::new(SharedState::default()));
        let barrier_cv = Arc::new(Condvar::new());
        let ar_cv = Arc::new(Condvar::new());
        let bcast_cv = Arc::new(Condvar::new());
        (0..world)
            .map(|r| {
                Arc::new(Self {
                    rank: r,
                    world,
                    state: state.clone(),
                    barrier_cv: barrier_cv.clone(),
                    ar_cv: ar_cv.clone(),
                    bcast_cv: bcast_cv.clone(),
                }) as Arc<dyn Communicator>
            })
            .collect()
    }
}

impl Communicator for LocalCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn barrier(&self) {
        let mut guard = self.state.lock().unwrap();
        guard.barrier_count += 1;
        if guard.barrier_count == self.world {
            // finalize this barrier generation and release all waiters
            guard.barrier_count = 0;
            guard.barrier_generation = guard.barrier_generation.wrapping_add(1);
            self.barrier_cv.notify_all();
            return;
        }
        while guard.barrier_count != 0 {
            guard = self.barrier_cv.wait(guard).unwrap();
        }
    }

    fn allreduce_sums_counts(
        &self,
        local_sums: &[Vec<f32>],
        local_counts: &[usize],
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut guard = self.state.lock().unwrap();
        let entry_gen = guard.barrier_generation;

        // If a previous round result is still being consumed or a different round is inflight,
        // wait until it fully resets to avoid cross-round interference.
        while guard.ar_result_ready
            || (guard.ar_pending_count != 0 && guard.ar_round_gen != entry_gen)
        {
            guard = self.ar_cv.wait(guard).unwrap();
        }

        // Initialize or accumulate for the current round
        if guard.ar_pending_count == 0 {
            guard.ar_round_gen = entry_gen;
            guard.ar_acc_sums = Some(local_sums.to_vec());
            guard.ar_acc_counts = Some(local_counts.to_vec());
            guard.ar_result_ready = false;
            guard.ar_waiters_remaining = 0;
        } else {
            // Same round accumulation
            if let Some(acc_sums) = guard.ar_acc_sums.as_mut() {
                for (k, row) in local_sums.iter().enumerate() {
                    for (d, v) in row.iter().enumerate() {
                        acc_sums[k][d] += *v;
                    }
                }
            }
            if let Some(acc_cnt) = guard.ar_acc_counts.as_mut() {
                for (k, c) in local_counts.iter().enumerate() {
                    acc_cnt[k] += *c;
                }
            }
        }

        guard.ar_pending_count += 1;
        if guard.ar_pending_count == self.world {
            // finalize and notify all
            guard.ar_result_ready = true;
            guard.ar_waiters_remaining = self.world;
            self.ar_cv.notify_all();
        } else {
            // wait until result is ready for this round
            while !guard.ar_result_ready || guard.ar_round_gen != entry_gen {
                guard = self.ar_cv.wait(guard).unwrap();
            }
        }

        // Clone result for return
        let sums = guard.ar_acc_sums.as_ref().unwrap().clone();
        let cnts = guard.ar_acc_counts.as_ref().unwrap().clone();

        // Mark this waiter as returned; reset when all have taken the result
        guard.ar_waiters_remaining -= 1;
        if guard.ar_waiters_remaining == 0 {
            // reset for next round
            guard.ar_pending_count = 0;
            guard.ar_acc_sums = None;
            guard.ar_acc_counts = None;
            guard.ar_result_ready = false;
            // Notify potential waiters for the next round
            self.ar_cv.notify_all();
        }
        (sums, cnts)
    }

    fn bcast_centroids(&self, centroids: &mut Vec<Vec<f32>>, root: usize) {
        let mut guard = self.state.lock().unwrap();
        let entry_gen = guard.barrier_generation;

        if self.rank == root {
            // Wait until previous broadcast is fully consumed before publishing a new one
            while guard.bcast_ready || guard.bcast_waiters_remaining != 0 {
                guard = self.bcast_cv.wait(guard).unwrap();
            }
            // Root publishes and waits for others to consume
            guard.bcast_data = Some(centroids.clone());
            guard.bcast_round_gen = entry_gen;
            guard.bcast_ready = true;
            guard.bcast_waiters_remaining = self.world;
            self.bcast_cv.notify_all();
        } else {
            // Wait until broadcast for this generation is ready
            while !guard.bcast_ready || guard.bcast_round_gen != entry_gen {
                guard = self.bcast_cv.wait(guard).unwrap();
            }
            if let Some(data) = guard.bcast_data.as_ref() {
                *centroids = data.clone();
            }
        }
        // Mark one waiter consumed (including root)
        guard.bcast_waiters_remaining -= 1;
        if guard.bcast_waiters_remaining == 0 {
            guard.bcast_ready = false;
            guard.bcast_data = None;
            // Notify possible next-round broadcasters
            self.bcast_cv.notify_all();
        }
    }
}
