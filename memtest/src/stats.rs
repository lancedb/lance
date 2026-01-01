use std::sync::atomic::{AtomicU64, Ordering};

/// Global allocation statistics tracked using atomic operations for thread safety
pub struct AllocationStats {
    pub total_allocations: AtomicU64,
    pub total_deallocations: AtomicU64,
    pub total_bytes_allocated: AtomicU64,
    pub total_bytes_deallocated: AtomicU64,
    pub current_bytes: AtomicU64,
    pub peak_bytes: AtomicU64,
}

impl AllocationStats {
    pub const fn new() -> Self {
        Self {
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            total_bytes_allocated: AtomicU64::new(0),
            total_bytes_deallocated: AtomicU64::new(0),
            current_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
        }
    }

    pub fn record_allocation(&self, size: usize) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated
            .fetch_add(size as u64, Ordering::Relaxed);

        let prev = self.current_bytes.fetch_add(size as u64, Ordering::Relaxed);
        let current = prev.saturating_add(size as u64);
        self.peak_bytes.fetch_max(current, Ordering::Relaxed);
    }

    pub fn record_deallocation(&self, size: usize) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_deallocated
            .fetch_add(size as u64, Ordering::Relaxed);

        // Use fetch_update to perform saturating subtraction atomically
        self.current_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size as u64))
            })
            .ok();
    }

    pub fn reset(&self) {
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_deallocations.store(0, Ordering::Relaxed);
        self.total_bytes_allocated.store(0, Ordering::Relaxed);
        self.total_bytes_deallocated.store(0, Ordering::Relaxed);
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
    }
}

/// Global statistics instance
pub static STATS: AllocationStats = AllocationStats::new();
