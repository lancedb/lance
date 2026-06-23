mod allocator;
mod stats;

use stats::STATS;

/// C-compatible statistics struct
#[repr(C)]
pub struct MemtestStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub total_bytes_allocated: u64,
    pub total_bytes_deallocated: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
}

/// Get all statistics in a single call
///
/// # Safety
/// The `stats` pointer must be valid and properly aligned
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memtest_get_stats(stats: *mut MemtestStats) {
    if stats.is_null() {
        return;
    }
    if (stats as usize).wrapping_rem(std::mem::align_of::<MemtestStats>()) != 0 {
        return;
    }

    (*stats).total_allocations = STATS
        .total_allocations
        .load(std::sync::atomic::Ordering::Relaxed);
    (*stats).total_deallocations = STATS
        .total_deallocations
        .load(std::sync::atomic::Ordering::Relaxed);
    (*stats).total_bytes_allocated = STATS
        .total_bytes_allocated
        .load(std::sync::atomic::Ordering::Relaxed);
    (*stats).total_bytes_deallocated = STATS
        .total_bytes_deallocated
        .load(std::sync::atomic::Ordering::Relaxed);
    (*stats).current_bytes = STATS
        .current_bytes
        .load(std::sync::atomic::Ordering::Relaxed);
    (*stats).peak_bytes = STATS.peak_bytes.load(std::sync::atomic::Ordering::Relaxed);
}

/// Reset all statistics to zero
#[unsafe(no_mangle)]
pub extern "C" fn memtest_reset_stats() {
    STATS.reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_stats_null() {
        unsafe {
            memtest_get_stats(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_get_stats_misaligned() {
        unsafe {
            let align = std::mem::align_of::<MemtestStats>();
            let size = std::mem::size_of::<MemtestStats>();
            let buf_size = size.saturating_add(align).saturating_add(align);
            let mut buf = vec![0u8; buf_size];
            let base = buf.as_mut_ptr() as usize;
            let offset = if base.wrapping_rem(align) == 0 { 1 } else { 0 };
            let misaligned = buf.as_mut_ptr().add(offset) as *mut MemtestStats;
            memtest_get_stats(misaligned);
        }
    }
}
