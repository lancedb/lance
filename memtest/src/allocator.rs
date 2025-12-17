use crate::stats::STATS;
use libc::{c_void, size_t};

#[cfg(target_os = "linux")]
mod sys {
    use super::*;

    extern "C" {
        #[link_name = "__libc_malloc"]
        fn libc_malloc(size: size_t) -> *mut c_void;
        #[link_name = "__libc_calloc"]
        fn libc_calloc(count: size_t, element_size: size_t) -> *mut c_void;
        #[link_name = "__libc_realloc"]
        fn libc_realloc(ptr: *mut c_void, size: size_t) -> *mut c_void;
        #[link_name = "__libc_free"]
        fn libc_free(ptr: *mut c_void);
        #[link_name = "__libc_memalign"]
        fn libc_memalign(alignment: size_t, size: size_t) -> *mut c_void;
    }

    pub(super) unsafe fn malloc(size: size_t) -> *mut c_void {
        libc_malloc(size)
    }

    pub(super) unsafe fn calloc(count: size_t, element_size: size_t) -> *mut c_void {
        libc_calloc(count, element_size)
    }

    pub(super) unsafe fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void {
        libc_realloc(ptr, size)
    }

    pub(super) unsafe fn free(ptr: *mut c_void) {
        libc_free(ptr);
    }

    pub(super) unsafe fn memalign(alignment: size_t, size: size_t) -> *mut c_void {
        libc_memalign(alignment, size)
    }
}

#[cfg(target_os = "macos")]
mod sys {
    use super::*;

    #[repr(C)]
    pub(super) struct malloc_zone_t {
        _private: [u8; 0],
    }

    extern "C" {
        fn malloc_default_zone() -> *mut malloc_zone_t;
        fn malloc_zone_malloc(zone: *mut malloc_zone_t, size: size_t) -> *mut c_void;
        fn malloc_zone_calloc(
            zone: *mut malloc_zone_t,
            count: size_t,
            element_size: size_t,
        ) -> *mut c_void;
        fn malloc_zone_memalign(
            zone: *mut malloc_zone_t,
            alignment: size_t,
            size: size_t,
        ) -> *mut c_void;
        fn malloc_zone_realloc(
            zone: *mut malloc_zone_t,
            ptr: *mut c_void,
            size: size_t,
        ) -> *mut c_void;
        fn malloc_zone_free(zone: *mut malloc_zone_t, ptr: *mut c_void);
        fn malloc_zone_from_ptr(ptr: *const c_void) -> *mut malloc_zone_t;
        fn malloc_size(ptr: *const c_void) -> size_t;
    }

    #[inline]
    unsafe fn zone_for_ptr(ptr: *const c_void) -> *mut malloc_zone_t {
        let zone = malloc_zone_from_ptr(ptr);
        if zone.is_null() {
            malloc_default_zone()
        } else {
            zone
        }
    }

    pub(super) unsafe fn malloc(size: size_t) -> *mut c_void {
        malloc_zone_malloc(malloc_default_zone(), size)
    }

    pub(super) unsafe fn calloc(count: size_t, element_size: size_t) -> *mut c_void {
        malloc_zone_calloc(malloc_default_zone(), count, element_size)
    }

    pub(super) unsafe fn memalign(alignment: size_t, size: size_t) -> *mut c_void {
        malloc_zone_memalign(malloc_default_zone(), alignment, size)
    }

    pub(super) unsafe fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void {
        if ptr.is_null() {
            return malloc(size);
        }
        malloc_zone_realloc(zone_for_ptr(ptr), ptr, size)
    }

    pub(super) unsafe fn free(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        malloc_zone_free(zone_for_ptr(ptr), ptr);
    }

    pub(super) unsafe fn usable_size(ptr: *const c_void) -> size_t {
        malloc_size(ptr)
    }
}

// Magic number to identify our allocations
#[cfg(target_os = "linux")]
const MAGIC: u64 = 0xDEADBEEF_CAFEBABE;

/// Header stored before each tracked allocation
#[cfg(target_os = "linux")]
#[repr(C)]
struct AllocationHeader {
    magic: u64,
    size: u64,
    alignment: u64,
    /// For aligned allocations, stores the actual pointer returned by libc_memalign
    /// For unaligned allocations, this is unused (but present for consistent size)
    actual_ptr: u64,
}

#[cfg(target_os = "linux")]
const HEADER_SIZE: usize = std::mem::size_of::<AllocationHeader>();

/// Check if a pointer was allocated by us
#[cfg(target_os = "linux")]
unsafe fn is_ours(virtual_ptr: *mut c_void) -> bool {
    if virtual_ptr.is_null() {
        return false;
    }
    let header_ptr = (virtual_ptr as *mut u8).sub(HEADER_SIZE) as *const AllocationHeader;
    (*header_ptr).magic == MAGIC
}

/// Extract size, alignment, and actual pointer from a virtual pointer
#[cfg(target_os = "linux")]
unsafe fn extract(virtual_ptr: *mut c_void) -> (usize, usize, *mut c_void) {
    let header_ptr = (virtual_ptr as *mut u8).sub(HEADER_SIZE) as *const AllocationHeader;
    let header = &*header_ptr;

    let size = header.size as usize;
    let alignment = header.alignment as usize;

    let actual_ptr = if alignment > 0 {
        // For aligned allocations, the actual pointer is stored in the header
        header.actual_ptr as *mut c_void
    } else {
        // For unaligned allocations, the actual pointer is the header itself
        header_ptr as *mut c_void
    };

    (size, alignment, actual_ptr)
}

/// Take an allocated pointer and size, store header, and return the adjusted pointer
#[cfg(target_os = "linux")]
unsafe fn to_virtual(actual_ptr: *mut c_void, size: usize, alignment: usize) -> *mut c_void {
    if actual_ptr.is_null() {
        return std::ptr::null_mut();
    }

    if alignment > 0 {
        // For aligned allocations:
        // 1. Find the first aligned position after we have room for the header
        // 2. Store the header just before that position
        // 3. Store the actual_ptr in the header so we can free it later

        let actual_addr = actual_ptr as usize;
        // Find the first address >= actual_addr + HEADER_SIZE that is aligned
        let min_virtual_addr = actual_addr.saturating_add(HEADER_SIZE);
        let virtual_addr = (min_virtual_addr.saturating_add(alignment).saturating_sub(1))
            & !(alignment.saturating_sub(1));

        // Write header just before the aligned virtual address
        let header_ptr = (virtual_addr.saturating_sub(HEADER_SIZE)) as *mut AllocationHeader;
        *header_ptr = AllocationHeader {
            magic: MAGIC,
            size: size as u64,
            alignment: alignment as u64,
            actual_ptr: actual_addr as u64,
        };

        virtual_addr as *mut c_void
    } else {
        // Unaligned allocation - header is at the start
        let header_ptr = actual_ptr as *mut AllocationHeader;
        *header_ptr = AllocationHeader {
            magic: MAGIC,
            size: size as u64,
            alignment: 0,
            actual_ptr: 0, // Unused for unaligned allocations
        };
        (actual_ptr as *mut u8).add(HEADER_SIZE) as *mut c_void
    }
}

#[cfg(target_os = "macos")]
#[inline]
fn is_power_of_two(value: usize) -> bool {
    value != 0 && (value & (value - 1)) == 0
}

#[cfg(target_os = "macos")]
#[inline]
fn is_valid_posix_memalign_alignment(alignment: usize) -> bool {
    is_power_of_two(alignment) && alignment >= std::mem::size_of::<*mut c_void>()
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    to_virtual(sys::malloc(size.saturating_add(HEADER_SIZE)), size, 0)
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn calloc(size: size_t, element_size: size_t) -> *mut c_void {
    let Some(total_size) = size.checked_mul(element_size) else {
        return std::ptr::null_mut();
    };
    STATS.record_allocation(total_size);
    to_virtual(
        sys::calloc(total_size.saturating_add(HEADER_SIZE), 1),
        total_size,
        0,
    )
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_malloc(size: size_t) -> *mut c_void {
    let ptr = sys::malloc(size);
    if !ptr.is_null() {
        STATS.record_allocation(sys::usable_size(ptr) as usize);
    }
    ptr
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_calloc(count: size_t, element_size: size_t) -> *mut c_void {
    let Some(_total_size) = count.checked_mul(element_size) else {
        return std::ptr::null_mut();
    };
    let ptr = sys::calloc(count, element_size);
    if !ptr.is_null() {
        STATS.record_allocation(sys::usable_size(ptr) as usize);
    }
    ptr
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    if is_ours(ptr) {
        // It's ours - extract size and track
        let (size, _alignment, actual_ptr) = extract(ptr);
        STATS.record_deallocation(size);
        sys::free(actual_ptr);
    } else {
        // Not ours - just free it without tracking
        sys::free(ptr);
    }
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    STATS.record_deallocation(sys::usable_size(ptr) as usize);
    sys::free(ptr);
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void {
    let (old_size, actual_ptr) = if ptr.is_null() || !is_ours(ptr) {
        // Either null or not ours - don't track
        if ptr.is_null() {
            (0, std::ptr::null_mut())
        } else {
            // Not ours - just realloc without tracking
            return sys::realloc(ptr, size);
        }
    } else {
        let (s, _align, a) = extract(ptr);
        (s, a)
    };

    STATS.record_deallocation(old_size);
    STATS.record_allocation(size);

    let new_ptr = sys::realloc(actual_ptr, size.saturating_add(HEADER_SIZE));
    to_virtual(new_ptr, size, 0)
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_realloc(ptr: *mut c_void, size: size_t) -> *mut c_void {
    if ptr.is_null() {
        let new_ptr = sys::realloc(std::ptr::null_mut(), size);
        if !new_ptr.is_null() {
            STATS.record_allocation(sys::usable_size(new_ptr) as usize);
        }
        return new_ptr;
    }

    let old_size = sys::usable_size(ptr);
    let new_ptr = sys::realloc(ptr, size);
    if new_ptr.is_null() {
        // For size == 0, some implementations free and return NULL.
        if size == 0 {
            STATS.record_deallocation(old_size as usize);
        }
        return std::ptr::null_mut();
    }

    STATS.record_deallocation(old_size as usize);
    STATS.record_allocation(sys::usable_size(new_ptr) as usize);
    new_ptr
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn memalign(alignment: size_t, size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    // Allocate extra space for header + padding to maintain alignment
    // We need: header (24 bytes) + actual_ptr (8 bytes) + padding to reach alignment
    let extra = alignment.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = sys::memalign(alignment, size.saturating_add(extra));
    to_virtual(actual_ptr, size, alignment)
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn posix_memalign(
    memptr: *mut *mut c_void,
    alignment: size_t,
    size: size_t,
) -> i32 {
    STATS.record_allocation(size);
    let extra = alignment.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = sys::memalign(alignment, size.saturating_add(extra));
    if actual_ptr.is_null() {
        return libc::ENOMEM;
    }
    *memptr = to_virtual(actual_ptr, size, alignment);
    0
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn aligned_alloc(alignment: size_t, size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    let extra = alignment.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = sys::memalign(alignment, size.saturating_add(extra));
    to_virtual(actual_ptr, size, alignment)
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn valloc(size: size_t) -> *mut c_void {
    STATS.record_allocation(size);
    let page_size = libc::sysconf(libc::_SC_PAGESIZE) as size_t;
    let extra = page_size.saturating_add(HEADER_SIZE).saturating_add(8);
    let actual_ptr = sys::memalign(page_size, size.saturating_add(extra));
    to_virtual(actual_ptr, size, page_size)
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_posix_memalign(
    memptr: *mut *mut c_void,
    alignment: size_t,
    size: size_t,
) -> i32 {
    if memptr.is_null() {
        return libc::EINVAL;
    }
    if !is_valid_posix_memalign_alignment(alignment as usize) {
        return libc::EINVAL;
    }

    let ptr = sys::memalign(alignment, size);
    if ptr.is_null() {
        return libc::ENOMEM;
    }
    STATS.record_allocation(sys::usable_size(ptr) as usize);
    *memptr = ptr;
    0
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_aligned_alloc(alignment: size_t, size: size_t) -> *mut c_void {
    if !is_valid_posix_memalign_alignment(alignment as usize) {
        return std::ptr::null_mut();
    }
    if size % alignment != 0 {
        return std::ptr::null_mut();
    }

    let ptr = sys::memalign(alignment, size);
    if !ptr.is_null() {
        STATS.record_allocation(sys::usable_size(ptr) as usize);
    }
    ptr
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_valloc(size: size_t) -> *mut c_void {
    let page_size = libc::sysconf(libc::_SC_PAGESIZE) as size_t;
    let ptr = sys::memalign(page_size, size);
    if !ptr.is_null() {
        STATS.record_allocation(sys::usable_size(ptr) as usize);
    }
    ptr
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn reallocarray(
    old_ptr: *mut c_void,
    count: size_t,
    element_size: size_t,
) -> *mut c_void {
    let Some(size) = count.checked_mul(element_size) else {
        return std::ptr::null_mut();
    };
    realloc(old_ptr, size)
}

#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn malloc_usable_size(ptr: *mut c_void) -> size_t {
    if ptr.is_null() {
        return 0;
    }

    if is_ours(ptr) {
        let (size, _, _) = extract(ptr);
        size
    } else {
        // Not our allocation - return 0 as we don't know the size
        // (there's no __libc_malloc_usable_size to call)
        0
    }
}

#[no_mangle]
#[cfg(target_os = "macos")]
pub unsafe extern "C" fn memtest_malloc_usable_size(ptr: *mut c_void) -> size_t {
    if ptr.is_null() {
        return 0;
    }
    sys::usable_size(ptr)
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct Interpose {
    replacement: *const c_void,
    original: *const c_void,
}

#[cfg(target_os = "macos")]
unsafe impl Sync for Interpose {}

#[cfg(target_os = "macos")]
extern "C" {
    fn malloc(size: size_t) -> *mut c_void;
    fn calloc(count: size_t, element_size: size_t) -> *mut c_void;
    fn realloc(ptr: *mut c_void, size: size_t) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn posix_memalign(memptr: *mut *mut c_void, alignment: size_t, size: size_t) -> i32;
    fn aligned_alloc(alignment: size_t, size: size_t) -> *mut c_void;
    fn valloc(size: size_t) -> *mut c_void;
}

#[cfg(target_os = "macos")]
#[used]
#[link_section = "__DATA,__interpose"]
static INTERPOSE_TABLE: [Interpose; 7] = [
    Interpose {
        replacement: memtest_malloc as *const () as *const c_void,
        original: malloc as *const () as *const c_void,
    },
    Interpose {
        replacement: memtest_calloc as *const () as *const c_void,
        original: calloc as *const () as *const c_void,
    },
    Interpose {
        replacement: memtest_realloc as *const () as *const c_void,
        original: realloc as *const () as *const c_void,
    },
    Interpose {
        replacement: memtest_free as *const () as *const c_void,
        original: free as *const () as *const c_void,
    },
    Interpose {
        replacement: memtest_posix_memalign as *const () as *const c_void,
        original: posix_memalign as *const () as *const c_void,
    },
    Interpose {
        replacement: memtest_aligned_alloc as *const () as *const c_void,
        original: aligned_alloc as *const () as *const c_void,
    },
    Interpose {
        replacement: memtest_valloc as *const () as *const c_void,
        original: valloc as *const () as *const c_void,
    },
];
