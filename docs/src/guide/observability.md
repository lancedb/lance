# Observability

Lance emits structured telemetry via the Rust [`tracing`](https://docs.rs/tracing) crate and exposes in-memory counters through typed stats structs. This design lets you bring your own subscriber: console logging, JSON output, or OpenTelemetry export via `tracing-opentelemetry`.

## Signal Taxonomy

Lance observability is organized into three layers:

| Layer | Targets | What it covers |
|-------|---------|---------------|
| IO | `lance_io::*` | Object store reads/writes, retries, backpressure, connection resets |
| Encoding | `lance_encoding::*` | Batch decoding latency, buffer operations |
| Dataset | `lance::*` | Compaction, fragment distribution, index operations |

## Tracing Targets

All tracing events use the `target:` field for filtering. Key targets:

| Target | Level | Description |
|--------|-------|-------------|
| `lance::compaction` | INFO | Compaction task lifecycle and byte/timing metrics |
| `lance::commit` | WARN | Commit conflict retries and permanent failures |
| `lance::write` | INFO/WARN | Write operations (insert mode, schema validation) |
| `lance::write::retry` | WARN | Write retry loop: conflict retries and exhaustion |
| `lance_io::retry` | WARN/DEBUG | Download retries and permanent failures |
| `lance_io::writer::connection_reset` | WARN | Upload connection reset retries |
| `lance_encoding::decode` | DEBUG | Batch decode span (num_rows) |
| `lance_index::hnsw::build` | INFO/DEBUG | HNSW index build progress (logged every 10k vectors) |
| `lance_index::hnsw::search` | DEBUG | Search strategy selection (flat vs graph) |

### Filtering Examples

```bash
# Show only compaction events
RUST_LOG="lance::compaction=info" cargo run

# Show IO retries and decode timing
RUST_LOG="lance_io::retry=debug,lance_encoding::decode=debug" cargo run

# Show everything at debug level
RUST_LOG=debug cargo run
```

## In-Memory Stats

### IO Stats (`IoStats`)

Available via `IOTracker::stats()` (snapshot) or `IOTracker::incremental_stats()` (reset-on-read):

| Field | Type | Description |
|-------|------|-------------|
| `read_iops` | u64 | Total read operations |
| `read_bytes` | u64 | Total bytes read |
| `write_iops` | u64 | Total write operations |
| `written_bytes` | u64 | Total bytes written |
| `read_latency_us` | u64 | Cumulative read latency (microseconds) |
| `write_latency_us` | u64 | Cumulative write latency (microseconds) |
| `read_errors` | u64 | Failed read operations |
| `write_errors` | u64 | Failed write operations |

### IO Scheduler Stats (`ScanStats`)

Available via `ScanScheduler::stats()`:

| Field | Type | Description |
|-------|------|-------------|
| `iops` | u64 | Total I/O operations submitted |
| `requests` | u64 | Total scheduler requests |
| `bytes_read` | u64 | Total bytes read |
| `coalesced_iops` | u64 | IOPs saved by coalescing adjacent ranges |
| `split_iops` | u64 | IOPs added by splitting large ranges |
| `backpressure_warnings` | u64 | Debounced backpressure warning emissions (coarse indicator) |

### Cache Stats

Available via `LanceCache` methods (see `lance-core/src/cache.rs`):

- `cache_hit_miss()` -- typed `CacheHitMiss` with hit/miss counts and `hit_ratio()`
- `sync_cache_stats()` -- capacity, utilization, eviction counts

### Index Metrics (`MetricsCollector`)

Implementable trait with these recording methods:

| Method | Description |
|--------|-------------|
| `record_parts_loaded(usize)` | Partition loads from storage |
| `record_index_loads(usize)` | Index loads from storage |
| `record_comparisons(usize)` | Index comparisons (B-tree, bitmap, etc.) |
| `record_bytes_loaded(u64)` | Bytes loaded for index data |
| `record_partitions_probed(usize)` | Partitions probed during search |
| `record_candidates_evaluated(usize)` | Candidate vectors evaluated |
| `record_search_duration_us(u64)` | Search wall-clock time |

Use `LocalMetricsCollector::snapshot()` for a typed `MetricsSnapshot`.

### Compaction Metrics (`CompactionMetrics`)

Returned by `compact_files()`:

| Field | Type | Description |
|-------|------|-------------|
| `fragments_removed` | usize | Fragments overwritten |
| `fragments_added` | usize | New fragments created |
| `files_removed` | usize | Files removed (including deletion files) |
| `files_added` | usize | Files created |
| `bytes_rewritten` | u64 | Bytes re-encoded |
| `bytes_binary_copied` | u64 | Bytes binary-copied (fast path) |
| `elapsed_ms_sum` | u64 | Cumulative worker time (sum of per-task durations) |

### Retry Stats (`RetrySnapshot`)

Available via `RetryStats::snapshot()` when attached to a `CloudObjectReader`:

| Field | Type | Description |
|-------|------|-------------|
| `total_retries` | u64 | Total retry attempts |
| `total_failures` | u64 | Permanent failures (all retries exhausted) |

## OpenTelemetry Integration

Lance does not depend on OpenTelemetry directly. To export Lance telemetry to an OTEL collector, add `tracing-opentelemetry` in your application:

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-opentelemetry = "0.28"
opentelemetry = "0.27"
opentelemetry-otlp = "0.27"
```

```rust
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

fn init_telemetry() {
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(your_otel_tracer());

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(otel_layer)
        .init();
}
```

All Lance `tracing` events will automatically flow to your OTEL backend. Filter by target prefix (`lance::`, `lance_io::`, `lance_encoding::`, `lance_index::`) to control volume.

## Cardinality Policy

- **Low-cardinality labels only** in default configuration: target, level, operation type
- **No file paths** in metric labels (paths appear in event fields, not span names)
- **No per-row or per-page events** in production builds; these are gated behind `DEBUG` level

## Performance Overhead

- Atomic counter updates: ~2ns amortized (Relaxed ordering)
- Tracing events at INFO/WARN: emitted only for significant lifecycle events
- Tracing events at DEBUG: high-frequency, filter in production using `RUST_LOG`
- IO latency tracking: one `Instant::now()` per ObjectStore call (already behind a Mutex)
