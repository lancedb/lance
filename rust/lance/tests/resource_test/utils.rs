use std::alloc::System;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Once};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Registry;
use tracking_allocator::{
    AllocationGroupId, AllocationGroupToken, AllocationLayer, AllocationRegistry,
    AllocationTracker, Allocator,
};

#[global_allocator]
static GLOBAL: Allocator<System> = Allocator::system();

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

static GLOBAL_STATS: LazyLock<Arc<Mutex<HashMap<AllocationGroupId, AllocStats>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

struct MemoryTracker;

impl AllocationTracker for MemoryTracker {
    fn allocated(
        &self,
        _addr: usize,
        _object_size: usize,
        wrapped_size: usize,
        group_id: AllocationGroupId,
    ) {
        if group_id == AllocationGroupId::ROOT {
            // We don't track root allocations
            return;
        }
        let mut guard = GLOBAL_STATS.lock().unwrap();
        let stats = guard.entry(group_id).or_default();
        stats.total_bytes_allocated += wrapped_size as isize;
        stats.total_allocations += 1;
        stats.max_bytes_allocated = stats.max_bytes_allocated.max(stats.net_bytes_allocated());
    }

    fn deallocated(
        &self,
        _addr: usize,
        _object_size: usize,
        wrapped_size: usize,
        source_group_id: AllocationGroupId,
        current_group_id: AllocationGroupId,
    ) {
        let group_id = if source_group_id != AllocationGroupId::ROOT {
            source_group_id
        } else {
            current_group_id
        };
        if group_id == AllocationGroupId::ROOT {
            // We don't track root allocations
            return;
        }
        let mut guard = GLOBAL_STATS.lock().unwrap();
        let stats = guard.entry(group_id).or_default();
        stats.total_bytes_deallocated += wrapped_size as isize;
        stats.total_deallocations += 1;
    }
}

static INIT: Once = Once::new();

// The alloc tracker holds a span and an associated allocation group id.
pub struct AllocTracker {
    group_id: AllocationGroupId,
    span: tracing::Span,
}

impl AllocTracker {
    pub fn init() {
        INIT.call_once(init_memory_tracking);
    }

    pub fn new() -> Self {
        Self::init();

        let token = AllocationGroupToken::register().expect("failed to register token");
        let group_id = token.id();

        let span = tracing::span!(tracing::Level::INFO, "AllocTracker");
        token.attach_to_span(&span);

        Self { group_id, span }
    }

    pub fn enter(&self) -> AllocGuard<'_> {
        AllocGuard::new(self)
    }

    pub fn stats(self) -> AllocStats {
        let mut stats = GLOBAL_STATS.lock().unwrap();
        stats.remove(&self.group_id).unwrap_or_default()
    }
}

pub struct AllocGuard<'a> {
    _guard: tracing::span::Entered<'a>,
}

impl<'a> AllocGuard<'a> {
    #[allow(clippy::print_stderr)]
    pub fn new(tracker: &'a AllocTracker) -> Self {
        if std::env::var("RUST_ALLOC_TIMINGS").is_ok() {
            eprintln!("alloc:enter:{}", chrono::Utc::now().to_rfc3339());
        }
        AllocGuard {
            _guard: tracker.span.enter(),
        }
    }
}

impl Drop for AllocGuard<'_> {
    #[allow(clippy::print_stderr)]
    fn drop(&mut self) {
        if std::env::var("RUST_ALLOC_TIMINGS").is_ok() {
            eprintln!("alloc:exit:{}", chrono::Utc::now().to_rfc3339());
        }
    }
}

pub fn init_memory_tracking() {
    let registry = Registry::default().with(AllocationLayer::new());
    tracing::subscriber::set_global_default(registry)
        .expect("failed to install tracing subscriber");

    let tracker = MemoryTracker;
    AllocationRegistry::set_global_tracker(tracker).expect("failed to set global tracker");
    AllocationRegistry::enable_tracking();
}

#[test]
fn check_memory_leak() {
    // Make sure AllocTracker can detect leaks
    let mut leaked = Vec::new();
    let tracker = AllocTracker::new();
    {
        let _guard = tracker.enter();
        let v = vec![0u8; 1024 * 1024];
        leaked.resize(1024, 0u8);
        drop(v);
    }
    let stats = tracker.stats();
    assert_eq!(stats.max_bytes_allocated, (1024 * 1024) + 1024 + 16);
    assert_eq!(stats.total_bytes_allocated, (1024 * 1024) + 1024 + 16);
    assert_eq!(stats.total_bytes_deallocated, (1024 * 1024) + 8);
    assert_eq!(stats.total_allocations, 2);
    assert_eq!(stats.net_bytes_allocated(), 1024 + 8);
}
