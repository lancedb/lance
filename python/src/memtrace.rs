// Copyright 2025 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg(feature = "memtrace")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
struct MemtraceStats {
    allocations: u64,
    deallocations: u64,
    current_bytes: u64,
    peak_bytes: u64,
}

struct Counters {
    allocations: AtomicU64,
    deallocations: AtomicU64,
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            current_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.allocations.store(0, Ordering::Relaxed);
        self.deallocations.store(0, Ordering::Relaxed);
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
    }

    fn reset_peak_to_current(&self) {
        let current = self.current_bytes.load(Ordering::Relaxed);
        self.peak_bytes.store(current, Ordering::Relaxed);
    }

    fn snapshot(&self) -> MemtraceStats {
        MemtraceStats {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
        }
    }

    fn record_alloc(&self, size: u64) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let old = self
            .current_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_add(size))
            })
            .expect("update function never returns None");
        let current = old.saturating_add(size);
        self.update_peak(current);
    }

    fn record_dealloc(&self, size: u64) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.current_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(size))
            })
            .ok();
    }

    fn record_realloc(&self, old_size: u64, new_size: u64) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.allocations.fetch_add(1, Ordering::Relaxed);

        let old = self
            .current_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                let cur = cur.saturating_sub(old_size);
                Some(cur.saturating_add(new_size))
            })
            .expect("update function never returns None");

        let current = old.saturating_sub(old_size).saturating_add(new_size);
        self.update_peak(current);
    }

    fn update_peak(&self, current: u64) {
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }
}

static COUNTERS: Counters = Counters::new();

pub(crate) struct CountingAlloc<A> {
    inner: A,
}

impl<A> CountingAlloc<A> {
    pub(crate) const fn new(inner: A) -> Self {
        Self { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            COUNTERS.record_alloc(layout.size() as u64);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc_zeroed(layout);
        if !ptr.is_null() {
            COUNTERS.record_alloc(layout.size() as u64);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        if !ptr.is_null() {
            COUNTERS.record_dealloc(layout.size() as u64);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = self.inner.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            COUNTERS.record_realloc(layout.size() as u64, new_size as u64);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc<System> = CountingAlloc::new(System);

#[pyfunction(name = "_memtrace_is_enabled")]
pub(crate) fn memtrace_is_enabled() -> bool {
    true
}

#[pyfunction(name = "_memtrace_reset")]
pub(crate) fn memtrace_reset() {
    COUNTERS.reset();
}

#[pyfunction(name = "_memtrace_reset_peak_to_current")]
pub(crate) fn memtrace_reset_peak_to_current() {
    COUNTERS.reset_peak_to_current();
}

#[pyfunction(name = "_memtrace_get_stats")]
pub(crate) fn memtrace_get_stats() -> (u64, u64, u64, u64) {
    let s = COUNTERS.snapshot();
    (s.allocations, s.deallocations, s.current_bytes, s.peak_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_and_basic_accounting() {
        COUNTERS.reset();
        let before = COUNTERS.snapshot();

        unsafe {
            let layout = Layout::from_size_align(1024, 8).unwrap();
            let p = std::alloc::alloc(layout);
            assert!(!p.is_null());
            std::alloc::dealloc(p, layout);
        }

        let after = COUNTERS.snapshot();
        assert!(after.allocations >= before.allocations + 1);
        assert!(after.deallocations >= before.deallocations + 1);
        assert!(after.peak_bytes >= before.current_bytes);
    }

    #[test]
    fn reset_peak_to_current_semantics() {
        COUNTERS.reset();

        unsafe {
            let layout = Layout::from_size_align(4096, 8).unwrap();
            let p = std::alloc::alloc(layout);
            assert!(!p.is_null());
            COUNTERS.reset_peak_to_current();
            let s = COUNTERS.snapshot();
            assert!(s.peak_bytes >= s.current_bytes);
            std::alloc::dealloc(p, layout);
        }
    }
}
