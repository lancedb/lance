// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::alloc::System;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL: TrackingAllocator<System> = TrackingAllocator::new(System);
static GLOBAL_STATS: AllocStatsAtomic = AllocStatsAtomic::new();

pub fn reset_alloc_stats() {
    GLOBAL_STATS.reset();
}

pub fn get_alloc_stats() -> AllocStats {
    GLOBAL_STATS.get_stats()
}

#[derive(Default, Clone, Debug)]
pub struct AllocStats {
    pub max_bytes_allocated: isize,
    pub total_bytes_allocated: isize,
    pub total_bytes_deallocated: isize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
}

impl AllocStats {
    pub fn net_bytes_allocated(&self) -> isize {
        self.total_bytes_allocated - self.total_bytes_deallocated
    }
}

struct AllocStatsAtomic {
    current_bytes: AtomicIsize,
    max_bytes: AtomicIsize,
    total_allocated: AtomicIsize,
    total_deallocated: AtomicIsize,
    total_alloc_count: AtomicUsize,
    total_dealloc_count: AtomicUsize,
}

impl AllocStatsAtomic {
    const fn new() -> Self {
        Self {
            current_bytes: AtomicIsize::new(0),
            max_bytes: AtomicIsize::new(0),
            total_allocated: AtomicIsize::new(0),
            total_deallocated: AtomicIsize::new(0),
            total_alloc_count: AtomicUsize::new(0),
            total_dealloc_count: AtomicUsize::new(0),
        }
    }

    fn record_alloc(&self, size: isize) {
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        self.total_alloc_count.fetch_add(1, Ordering::Relaxed);

        let current = self.current_bytes.fetch_add(size, Ordering::Relaxed) + size;

        // Update max if current exceeds it
        self.max_bytes.fetch_max(current, Ordering::Relaxed);
    }

    fn record_dealloc(&self, size: isize) {
        self.total_deallocated.fetch_add(size, Ordering::Relaxed);
        self.total_dealloc_count.fetch_add(1, Ordering::Relaxed);
        self.current_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    fn get_stats(&self) -> AllocStats {
        AllocStats {
            max_bytes_allocated: self.max_bytes.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_bytes_deallocated: self.total_deallocated.load(Ordering::Relaxed),
            total_allocations: self.total_alloc_count.load(Ordering::Relaxed),
            total_deallocations: self.total_dealloc_count.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.max_bytes.store(0, Ordering::Relaxed);
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_deallocated.store(0, Ordering::Relaxed);
        self.total_alloc_count.store(0, Ordering::Relaxed);
        self.total_dealloc_count.store(0, Ordering::Relaxed);
    }
}

/// Global allocator wrapper that tracks memory usage
pub struct TrackingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            GLOBAL_STATS.record_alloc(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        GLOBAL_STATS.record_dealloc(layout.size() as isize);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc_zeroed(layout);
        if !ptr.is_null() {
            GLOBAL_STATS.record_alloc(layout.size() as isize);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = self.inner.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            // Record deallocation of old size and allocation of new size
            GLOBAL_STATS.record_dealloc(layout.size() as isize);
            GLOBAL_STATS.record_alloc(new_size as isize);
        }
        new_ptr
    }
}

#[test]
fn check_memory_leak() {
    // Make sure AllocTracker can detect leaks
    let mut leaked = Vec::new();
    reset_alloc_stats();
    let v = vec![0u8; 1024 * 1024];
    leaked.resize(1024, 0u8);
    drop(v);

    let stats = get_alloc_stats();
    assert_eq!(stats.total_allocations, 2);
    assert_eq!(stats.total_deallocations, 1);
    assert_eq!(stats.max_bytes_allocated, (1024 * 1024) + 1024);
    assert_eq!(stats.total_bytes_allocated, (1024 * 1024) + 1024);
    assert_eq!(stats.total_bytes_deallocated, (1024 * 1024));

    assert_eq!(stats.net_bytes_allocated(), 1024);
}
