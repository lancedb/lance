// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Black-box reproducer for
//! <https://github.com/lance-format/lance/issues/9575>: "ScanScheduler
//! byte-budget backpressure deadlocks compaction over many small
//! fragments".
//!
//! The report describes a dataset with hundreds of small fragments and a
//! wide schema (many columns, some large) where compaction permanently
//! stalls: every background thread parks, there is no I/O in flight, no
//! error, and no timeout. The suspected mechanism (see
//! `lance-io/src/scheduler.rs`'s `IoQueueState`) is a byte-budget circular
//! wait: the scheduler debits its `io_buffer_size` budget when a read is
//! *delivered*, and only credits it back when the *consumer* pulls that
//! data out. With enough fragments and columns in flight at once, the
//! scheduler can deliver (and debit) reads for batches nobody has consumed
//! yet, until the budget can no longer admit the one read the consumer is
//! actually blocked waiting on - and nothing can consume until that read is
//! admitted. The author's own suggested regression test (issue, suggestion
//! #4) is exactly this: many small fragments x wide schema, with
//! `io_buffer_size` small enough to force the inversion.
//!
//! This test only goes through the public `lance` crate API
//! (`Dataset::write`, `lance::dataset::optimize::{CompactionOptions,
//! compact_files}`) - the same surface a downstream user has - and does not
//! reach into `lance-io` internals.
//!
//! A genuine reproduction is a true deadlock, not an error, so it would
//! hang forever rather than fail fast. Each compaction call below is
//! wrapped in a `tokio::time::timeout`; failing to finish in time is taken
//! as the reproduction signal (mirroring the timeout-based deadlock tests
//! already in `lance-io/src/scheduler.rs`). If the scheduler really does
//! wedge every runtime worker thread (as the issue describes), the timeout
//! can only reclaim the task it wraps - other parked threads may keep the
//! process alive after the test "fails", which is itself consistent with
//! the reported symptom ("killed manually" after 30 minutes).
//!
//! ## Artificial read latency
//!
//! Every occurrence reported in the issue happened reading from S3; a fully
//! local, uncontended in-memory or filesystem store completes reads so fast
//! that very little data ever sits "delivered but unconsumed" at once, so
//! an early version of this test (local temp dir, no injected latency)
//! never reproduced the stall even down to a 1 KiB `io_buffer_size`. To
//! stand in for that network latency, every read is routed through
//! `object_store::throttle::ThrottledStore` - a stock, non-Lance-internal
//! wrapper - bound in via the public [`WrappingObjectStore`] extension
//! point on [`ObjectStoreParams::object_store_wrapper`]. This widens the
//! window in which many fragments' reads can be delivered before any of
//! them are consumed, which is the condition the issue describes.
//!
//! ## Status
//!
//! As of lance `0880bfdeb`, this reproducer has **not** reproduced the
//! deadlock across `io_buffer_size` in `{1 KiB, 4 KiB, 16 KiB}`, with
//! uniform and jittered injected latency, and with `num_threads` up to 8.
//!
//! Turning on the scheduler's own `lance_io::scheduler::state=trace` events
//! (see `diagnostic_trace_scheduler_state_under_tightest_budget` below)
//! shows real, substantial backpressure pressure under the tightest config:
//! across ~12,000 state transitions, `head_task_blocked_by_bytes=true` in
//! roughly half of them, and `bytes_available` (budget `1024`) dropping as
//! low as **-141,664**. So the scheduler is genuinely budget-starved, not
//! just lightly loaded - and it still always resolves. That is consistent
//! with `lance-io/src/scheduler.rs`'s own `stress_backpressure` test, whose
//! comment states the byte-budget scheduler cannot deadlock "as long as all
//! requests are consumed in priority order" (its admission rule always lets
//! through whichever outstanding task has the *lowest* priority, regardless
//! of budget). Uniform injected latency keeps completions in roughly
//! issue-order, so switching to independently-randomized jitter
//! (`JitteredLatencyStore`) was an attempt to break that ordering; it did
//! raise the peak number of concurrently in-flight priority classes
//! (11 -> 19) and deepen the budget excursion, but still not past whatever
//! threshold actually wedges the consumer.
//!
//! That is not strong evidence the bug is fixed or absent: the original
//! report needed real S3 latency against GB-scale, 600+ fragment,
//! 75-column datasets over tens of seconds to minutes before the stall
//! appeared, and this black-box test can only reach the scheduler's
//! behavior through the public knobs above (`CompactionOptions`,
//! `WriteParams`, `ObjectStoreParams`) - it cannot force the exact internal
//! interleaving the issue hypothesizes. Treat this file as reproduction
//! infrastructure to keep tuning rather than as a negative result.
//!
//! ### Structural amplifier: list-column item scheduling
//!
//! Reading `rust/lance/src/io/exec/filtered_read.rs` (compaction's actual
//! read path - *not* `scan.rs`'s `try_new_v2`, which is a different code
//! path) shows its `.buffered(fragment_readahead).try_flatten()` consumes
//! fragments strictly in ascending priority order, and its struct/list
//! decoders (`lance-encoding/src/array_encoding/logical/struct.rs`,
//! `list.rs`) only mark a column "loaded" once real data has arrived, not
//! merely scheduled - so no priority-order violation was found there.
//!
//! One real, confirmed amplifier was found: scheduling a list column's
//! *items* (as opposed to its offsets) happens via a bare, untracked
//! `tokio::spawn` per list column per batch per open fragment
//! (`ListFieldSchedulingJob::schedule_next` ->
//! `indirect_schedule_task`, `lance-encoding/src/array_encoding/logical/
//! list.rs`). Each spawned task becomes its own independently-bypassed
//! in-flight priority class the moment it submits its first item request -
//! individually safe per the admission rule, but a majority-list-column,
//! skewed-length schema multiplies how many such classes can be alive at
//! once. `diagnostic_skewed_list_columns_no_latency` below (zero injected
//! latency - purely structural, confirming timing is not the driver)
//! quantifies the effect by turning this knob up in three steps, each
//! measured via the scheduler's own trace events:
//!
//! | config | list cols | peak concurrent priority classes | budget | deepest overrun | wall time |
//! |---|---|---|---|---|---|
//! | baseline (14 list cols, uniform lengths, ~20 fragments open) | 14 | ~11-21 | 1 KiB | ~140 KiB | ~2 s |
//! | 50 list cols, 10% long (200-800 items) | 50 | 142 | 256 KiB | >1 MiB | 2.5 s |
//! | 65 list cols, 25% long (500-3000 items) | 65 | 172 | 32 KiB | **14 MiB (440x)** | **15 s** |
//!
//! Each step is local, in-memory, zero-latency, yet still completes -
//! consistent with the architecture being formally deadlock-safe as long as
//! `filtered_read.rs` keeps consuming in priority order (which it does).
//! But the trend (6x wall time and ~2x peak concurrency for one more turn
//! of the skew knob, with no sign of flattening) suggests that on a real,
//! much larger dataset this amplifier could plausibly turn "slow" into
//! "indistinguishable from hung" well before any real deadlock condition is
//! needed - especially stacked with real S3 latency (which separately also
//! deepens the excursion, per the jitter experiments above). It's also
//! possible the production stall is a different failure mode entirely -
//! e.g. tokio worker-thread starvation from the sheer *count* of untracked
//! spawned tasks under this amplifier, rather than a logical cycle in
//! `IoQueueState` - which a future experiment could check directly via
//! tokio's runtime task-count metrics during a heavily-skewed run.
//!
//! ### Three more mechanisms ruled out; still no true (0% CPU) deadlock
//!
//! The issue's own gdb snapshot - "every lance runtime thread in
//! futex_wait, no bytes in flight on any socket, memory flat, 0% CPU" - is
//! a genuine stuck-forever condition, not "very slow under load" (which is
//! all the amplifier above demonstrated: wall time climbing, but CPU and
//! trace-event activity never stopping). That reframes the search away
//! from `IoQueueState` (provably non-deadlocking as long as consumption
//! stays priority-ordered, which `filtered_read.rs` does) toward a genuine
//! synchronization-primitive cycle. Three more candidates were tested,
//! each independently motivated, none reproducing a true stall:
//!
//! 1. **Index remap** (`diagnostic_compaction_with_scalar_index`): a real
//!    BTree scalar index, created on a stable-row-ID dataset, with
//!    `defer_index_remap: false` matching the issue's own Python repro
//!    verbatim. No measurable effect - confirmed by code
//!    (`commit_compaction`'s `records_stable_row_id_moves`,
//!    `rust/lance/src/dataset/optimize.rs`: `needs_remapping` requires
//!    `!uses_stable_row_ids`, so the classic remap path is structurally
//!    unreachable for a stable-row-ID table regardless of
//!    `defer_index_remap`) and by debug logs (zero index-related log lines
//!    fire during compaction with the index present).
//! 2. **CPU-pool self-starvation** (`diagnostic_collapsed_cpu_pool_no_
//!    latency`): `lance-core/src/utils/tokio.rs`'s `spawn_cpu` dispatches
//!    CPU-bound work (page encode/decode, index prefilter computation) onto
//!    one global blocking-thread pool that collapses to a single worker on
//!    hosts with <= 3 visible CPUs (its own doc comment names this exact
//!    failure mode, with a real prior incident, PR #7423: a closure that
//!    blocks/waits/parks while occupying the sole thread can starve the
//!    work that would unblock it - 0% CPU, no timeout, no error). Forcing
//!    `LANCE_CPU_THREADS=1` (the same public knob `spawn_cpu`'s own doc
//!    points at) plus the index and skewed schema: ~5x slower (80s vs
//!    ~15s), `user` time still exceeding wall time throughout, no hang.
//! 3. **Real S3** (`diagnostic_skewed_list_columns_with_scalar_index_on_s3`,
//!    `diagnostic_everything_combined_on_s3`): every prior test ran against
//!    `memory://` or a synthetic latency wrapper around it. These run
//!    against a real S3-compatible server over real sockets (tested with a
//!    local single-node Garage instance - MinIO's open-source server binary
//!    was archived and is no longer distributed - see
//!    `s3_store_params_from_env`), optionally with `JitteredLatencyStore`
//!    layered on top (a local server is loopback-fast; real S3 round-trip
//!    time needs to be added back in). The last test stacks *every*
//!    amplifier found in this investigation at once: real S3 sockets +
//!    5-50ms jitter, `LANCE_CPU_THREADS=1`, `LANCE_DEFAULT_FRAGMENT_
//!    READAHEAD` past the fragment count, the skewed 65-list-column schema,
//!    and the BTree index. It completed in 125s at a sustained ~100% CPU
//!    (averaged over the whole run, per `/usr/bin/time`) - consistent with
//!    continuous progress throughout, not a long stall followed by a burst.
//!
//! So: four independently-motivated mechanisms tried, three of them
//! stacked together at once over real sockets, and still no reproduction of
//! a true, CPU-idle deadlock.
//!
//! ### Bimodal fragment sizes and GB-scale data
//!
//! The issue's dataset shape is bimodal, not uniformly tiny: "587 fragments
//! hold < 100 rows each ... 14 hold 1.5k-22k rows (~1 GiB each)". Every
//! test above used only the tiny half. `write_bimodal_fragments_at` writes
//! both: `NUM_FRAGMENTS` tiny fragments, then `BIMODAL_NUM_LARGE_FRAGMENTS`
//! large ones appended on top of the same [`wide_schema_generator`] schema.
//! `diagnostic_bimodal_fragments_with_scalar_index_on_s3` confirmed by
//! listing the S3 bucket after a run that this is genuinely GB-scale
//! (**2.8 GiB** for one run - the issue needed ~1 GiB read before
//! stalling), with `target_rows_per_fragment: 1500` matching the issue's
//! own repro value verbatim (keeping the large fragments out of the
//! rewrite task itself, same as the issue's own repro, while they remain
//! part of the dataset and - once the index is created afterward - part of
//! what it covers). It completed in 12s.
//! `diagnostic_everything_combined_bimodal_on_s3` stacks this GB-scale
//! bimodal shape with every other amplifier at once (real S3 + jitter,
//! collapsed CPU pool, high fragment readahead, BTree index over the whole
//! dataset) - 20s, still no hang. Notably *faster* than the equivalent
//! skewed-schema runs despite moving far more data, which points at
//! CPU-bound scheduling work (the list-item-spawn amplifier, `spawn_cpu`
//! contention) as the dominant cost driver in this investigation so far,
//! not raw byte volume.
//!
//! ### What's still untried
//!
//! Five independently-motivated mechanisms tried now (three of them
//! stacked together, one of those at GB-scale), and still no reproduction
//! of a true, CPU-idle deadlock. What's left: real AWS S3 specifically
//! rather than an S3-compatible server (genuine AWS SDK retry/connection-
//! pool/credential-refresh behavior a from-scratch reimplementation like
//! Garage may not replicate), and combining the bimodal/GB-scale shape with
//! the CPU-expensive skewed-list schema rather than the cheaper
//! `wide_schema_generator` one (untried because generating that much data
//! with 65 skewed list columns is expensive enough to be impractical for
//! routine local iteration).

use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::io::{ObjectStoreParams, StorageOptionsAccessor, WrappingObjectStore};
use lance_datagen::{BatchCount, BatchGeneratorBuilder, ByteCount, RowCount, array, gen_batch};
use lance_file::writer::FileWriterOptions;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use object_store::ObjectStore as OsObjectStore;
use object_store::list::PaginatedListStore;
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use rand::{Rng, SeedableRng};
use rstest::rstest;

/// Injects artificial per-request latency into every object-store read; see
/// "Artificial read latency" above.
#[derive(Debug, Clone, Copy)]
struct LatencyInjectingStore {
    config: ThrottleConfig,
}

impl WrappingObjectStore for LatencyInjectingStore {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OsObjectStore>,
    ) -> Arc<dyn OsObjectStore> {
        Arc::new(ThrottledStore::new(original, self.config))
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

fn latency_injecting_store_params(per_call: Duration, per_byte: Duration) -> ObjectStoreParams {
    ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(LatencyInjectingStore {
            config: ThrottleConfig {
                wait_get_per_call: per_call,
                wait_get_per_byte: per_byte,
                ..Default::default()
            },
        })),
        ..Default::default()
    }
}

/// A `get_opts`/`get_ranges` wrapper that sleeps an *independent random*
/// duration (uniform in `[min, max]`) before every read, instead of
/// `ThrottledStore`'s fixed per-call delay.
///
/// `object_store::throttle::ThrottledStore` applies the *same* latency to
/// every call, so every read effectively stays in roughly the relative
/// completion order it was issued in - priority order is preserved, which
/// `lance-io/src/scheduler.rs`'s own `stress_backpressure` test documents as
/// sufficient to avoid deadlocking the byte-budget scheduler no matter how
/// tight the buffer is. Real S3 does not behave that way: per-request
/// latency varies independently (connection reuse, throttling, tail
/// latency), so a later-priority read can legitimately *finish* before an
/// earlier-priority one that is still in flight. That is what could let a
/// column needed next (whose read didn't happen to finish early) stay stuck
/// behind a byte budget held by other, already-delivered-but-unconsumed
/// reads that merely *finished* out of order - independent of this test's
/// `tokio::time::timeout`-based deadlock check, this is reproduction
/// infrastructure to try the thing the scheduler's own escape valve doesn't
/// protect against.
#[derive(Debug, Clone, Copy)]
struct JitteredLatencyStore {
    min_latency: Duration,
    max_latency: Duration,
}

impl std::fmt::Display for JitteredLatencyStoreImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitteredLatencyStore({})", self.inner)
    }
}

#[derive(Debug)]
struct JitteredLatencyStoreImpl {
    inner: Arc<dyn OsObjectStore>,
    min_latency: Duration,
    max_latency: Duration,
}

impl JitteredLatencyStoreImpl {
    async fn jitter(&self) {
        let nanos = if self.max_latency > self.min_latency {
            rand::rng().random_range(self.min_latency.as_nanos()..=self.max_latency.as_nanos())
        } else {
            self.min_latency.as_nanos()
        };
        tokio::time::sleep(Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)).await;
    }
}

#[async_trait::async_trait]
impl OsObjectStore for JitteredLatencyStoreImpl {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.jitter().await;
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        self.jitter().await;
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

impl WrappingObjectStore for JitteredLatencyStore {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OsObjectStore>,
    ) -> Arc<dyn OsObjectStore> {
        Arc::new(JitteredLatencyStoreImpl {
            inner: original,
            min_latency: self.min_latency,
            max_latency: self.max_latency,
        })
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

fn jittered_latency_store_params(
    min_latency: Duration,
    max_latency: Duration,
) -> ObjectStoreParams {
    ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(JitteredLatencyStore {
            min_latency,
            max_latency,
        })),
        ..Default::default()
    }
}

/// Hundreds of fragments, each tiny - mirrors the issue's dataset shape
/// ("587 fragments hold < 100 rows each ... from small appends").
const NUM_FRAGMENTS: u32 = 300;
const ROWS_PER_FRAGMENT: u64 = 20;

/// 75 columns total: plain scalars, list columns (standing in for the
/// issue's 14 `list<...>` columns), and a handful of larger string columns
/// (standing in for its 8 `lance.blob.v2` columns). The wide per-row
/// footprint is what lets a modest number of concurrently in-flight
/// fragments exhaust a small byte budget.
const NUM_PLAIN_COLUMNS: usize = 53;
const NUM_LIST_COLUMNS: usize = 14;
const NUM_WIDE_COLUMNS: usize = 8;
const WIDE_COLUMN_BYTES: u64 = 256;

/// How long to give one compaction run before treating it as deadlocked.
/// With the injected per-read latency below, normal completion takes on
/// the order of 30-40 seconds on this (intentionally tiny) dataset; this
/// leaves several times that as headroom for slow/loaded CI boxes.
const DEADLOCK_TIMEOUT: Duration = Duration::from_secs(180);

/// The column generator shared by every "wide schema" dataset in this file
/// (not the separate majority-list skewed schema, which builds its own
/// batches by hand - see `skewed_schema`): 75 columns of plain scalars,
/// list columns (standing in for the issue's 14 `list<...>` columns), and
/// larger string columns (standing in for its 8 `lance.blob.v2` columns).
fn wide_schema_generator() -> BatchGeneratorBuilder {
    let mut builder = gen_batch();
    for i in 0..NUM_PLAIN_COLUMNS {
        builder = builder.col(format!("plain_{i}"), array::rand::<Int64Type>());
    }
    for i in 0..NUM_LIST_COLUMNS {
        builder = builder.col(
            format!("list_{i}"),
            array::rand_list(&DataType::Int64, false),
        );
    }
    for i in 0..NUM_WIDE_COLUMNS {
        builder = builder.col(
            format!("wide_{i}"),
            array::rand_utf8(ByteCount::from(WIDE_COLUMN_BYTES), false),
        );
    }
    builder
}

/// Writes a dataset with `NUM_FRAGMENTS` small fragments and a wide,
/// stable-row-ID schema, all in a single `Dataset::write` call (one fragment
/// is produced per generated batch because `max_rows_per_file` equals the
/// per-batch row count).
async fn write_many_small_wide_fragments(store_params: Option<ObjectStoreParams>) -> Dataset {
    let reader = wide_schema_generator().into_reader_rows(
        RowCount::from(ROWS_PER_FRAGMENT),
        BatchCount::from(NUM_FRAGMENTS),
    );

    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        max_rows_per_group: ROWS_PER_FRAGMENT as usize,
        enable_stable_row_ids: true,
        store_params,
        ..Default::default()
    };

    let dataset = Dataset::write(reader, "memory://", Some(write_params))
        .await
        .expect("failed to write fixture dataset");
    assert_eq!(dataset.get_fragments().len(), NUM_FRAGMENTS as usize);

    dataset
}

/// The issue's actual fragment-size shape is bimodal, not uniform: "587
/// fragments hold < 100 rows each (2,255 rows total, from small appends),
/// 14 hold 1.5k-22k rows (~1 GiB each)". Every dataset built elsewhere in
/// this file uses only the first half of that shape (hundreds of uniformly
/// tiny fragments). This writes both halves of the same [`wide_schema_generator`]
/// schema into one dataset: `NUM_FRAGMENTS` tiny fragments (`Create`), then
/// `BIMODAL_NUM_LARGE_FRAGMENTS` large ones appended on top (`Append`) -
/// each generated batch becomes exactly one fragment in both calls, because
/// `max_rows_per_file` is set to each call's own batch row count.
///
/// `target_rows_per_fragment` in the resulting `CompactionOptions` should be
/// set well below `BIMODAL_LARGE_FRAGMENT_ROWS`, as in the issue's own
/// repro (`target_rows_per_fragment: 1500`, well under its smallest large
/// fragment's 1.5k rows) - that excludes the large fragments from the
/// rewrite task the same way the issue's own repro does, while they remain
/// part of the dataset (and, when an index is created afterward, part of
/// what the index covers).
const BIMODAL_NUM_LARGE_FRAGMENTS: u32 = 12;
const BIMODAL_LARGE_FRAGMENT_ROWS: u64 = 6_000;

async fn write_bimodal_fragments_at(uri: &str, store_params: Option<ObjectStoreParams>) -> Dataset {
    let small_reader = wide_schema_generator().into_reader_rows(
        RowCount::from(ROWS_PER_FRAGMENT),
        BatchCount::from(NUM_FRAGMENTS),
    );
    let create_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        max_rows_per_group: ROWS_PER_FRAGMENT as usize,
        enable_stable_row_ids: true,
        store_params: store_params.clone(),
        mode: WriteMode::Create,
        ..Default::default()
    };
    Dataset::write(small_reader, uri, Some(create_params))
        .await
        .expect("failed to write small fixture fragments");

    let large_reader = wide_schema_generator().into_reader_rows(
        RowCount::from(BIMODAL_LARGE_FRAGMENT_ROWS),
        BatchCount::from(BIMODAL_NUM_LARGE_FRAGMENTS),
    );
    let append_params = WriteParams {
        max_rows_per_file: BIMODAL_LARGE_FRAGMENT_ROWS as usize,
        enable_stable_row_ids: true,
        store_params,
        mode: WriteMode::Append,
        ..Default::default()
    };
    let dataset = Dataset::write(large_reader, uri, Some(append_params))
        .await
        .expect("failed to append large fixture fragments");

    assert_eq!(
        dataset.get_fragments().len(),
        (NUM_FRAGMENTS + BIMODAL_NUM_LARGE_FRAGMENTS) as usize
    );
    assert_eq!(
        dataset.count_rows(None).await.unwrap(),
        (NUM_FRAGMENTS as u64 * ROWS_PER_FRAGMENT
            + BIMODAL_NUM_LARGE_FRAGMENTS as u64 * BIMODAL_LARGE_FRAGMENT_ROWS) as usize
    );

    dataset
}

/// Most columns are `list<int64>` with a *skewed* length distribution (90%
/// short, 10% long) rather than the roughly-uniform lengths
/// `lance_datagen::array::rand_list` produces.
const SKEWED_NUM_PLAIN_COLUMNS: usize = 5;
const SKEWED_NUM_LIST_COLUMNS: usize = 65;
const SKEWED_NUM_WIDE_COLUMNS: usize = 5;
const SKEWED_SHORT_LIST_MAX_LEN: usize = 3;
const SKEWED_LONG_LIST_MIN_LEN: usize = 500;
const SKEWED_LONG_LIST_MAX_LEN: usize = 3000;
const SKEWED_LONG_LIST_PROBABILITY: f64 = 0.25;

fn skewed_schema() -> Arc<Schema> {
    let mut fields = Vec::with_capacity(
        SKEWED_NUM_PLAIN_COLUMNS + SKEWED_NUM_LIST_COLUMNS + SKEWED_NUM_WIDE_COLUMNS,
    );
    for i in 0..SKEWED_NUM_PLAIN_COLUMNS {
        fields.push(Field::new(format!("plain_{i}"), DataType::Int64, false));
    }
    for i in 0..SKEWED_NUM_LIST_COLUMNS {
        fields.push(Field::new(
            format!("list_{i}"),
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ));
    }
    for i in 0..SKEWED_NUM_WIDE_COLUMNS {
        fields.push(Field::new(format!("wide_{i}"), DataType::Utf8, false));
    }
    Arc::new(Schema::new(fields))
}

/// Builds one batch (= one fragment, when fed through `max_rows_per_file`)
/// of `num_rows` rows against [`skewed_schema`]. Every list column in a
/// batch independently rolls its own short/long length per row, so
/// different list columns desync from each other within the same fragment
/// (not just across fragments).
fn build_skewed_batch(num_rows: usize, rng: &mut impl rand::Rng) -> RecordBatch {
    let schema = skewed_schema();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

    for _ in 0..SKEWED_NUM_PLAIN_COLUMNS {
        let values: Vec<i64> = (0..num_rows).map(|_| rng.random()).collect();
        columns.push(Arc::new(Int64Array::from(values)));
    }
    for _ in 0..SKEWED_NUM_LIST_COLUMNS {
        let mut builder = ListBuilder::new(Int64Builder::new());
        for _ in 0..num_rows {
            let len = if rng.random_bool(SKEWED_LONG_LIST_PROBABILITY) {
                rng.random_range(SKEWED_LONG_LIST_MIN_LEN..=SKEWED_LONG_LIST_MAX_LEN)
            } else {
                rng.random_range(0..=SKEWED_SHORT_LIST_MAX_LEN)
            };
            for _ in 0..len {
                builder.values().append_value(rng.random());
            }
            builder.append(true);
        }
        columns.push(Arc::new(builder.finish()));
    }
    for _ in 0..SKEWED_NUM_WIDE_COLUMNS {
        let mut builder = StringBuilder::new();
        for _ in 0..num_rows {
            let s: String = (0..WIDE_COLUMN_BYTES).map(|_| 'x').collect();
            builder.append_value(&s);
        }
        columns.push(Arc::new(builder.finish() as StringArray));
    }

    RecordBatch::try_new(schema, columns).expect("failed to build skewed batch")
}

/// Like [`write_many_small_wide_fragments`] but with a majority-list,
/// skewed-length schema (see [`build_skewed_batch`]), and with small
/// on-disk pages/chunks (via `FileWriterOptions`) so a moderate
/// `io_buffer_size` produces genuine multi-request contention instead of
/// forcing every read through the single-head-task priority bypass (every
/// request would trivially exceed a too-tiny budget on its own otherwise).
async fn write_skewed_list_heavy_fragments() -> Dataset {
    write_skewed_list_heavy_fragments_at("memory://", None).await
}

/// Like [`write_skewed_list_heavy_fragments`], but against an arbitrary
/// destination URI and with arbitrary `store_params` - so it can be pointed
/// at a real S3-compatible backend instead of the in-memory store.
async fn write_skewed_list_heavy_fragments_at(
    uri: &str,
    store_params: Option<ObjectStoreParams>,
) -> Dataset {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xdead_beef);
    let batches: Vec<RecordBatch> = (0..NUM_FRAGMENTS)
        .map(|_| build_skewed_batch(ROWS_PER_FRAGMENT as usize, &mut rng))
        .collect();
    let schema = skewed_schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        max_rows_per_group: ROWS_PER_FRAGMENT as usize,
        enable_stable_row_ids: true,
        file_writer_options: Some(FileWriterOptions {
            max_page_bytes: Some(64 * 1024),
            data_cache_bytes: Some(64 * 1024),
            ..Default::default()
        }),
        store_params,
        ..Default::default()
    };

    let dataset = Dataset::write(reader, uri, Some(write_params))
        .await
        .expect("failed to write skewed fixture dataset");
    assert_eq!(dataset.get_fragments().len(), NUM_FRAGMENTS as usize);

    dataset
}

/// Attempts compaction with a small `io_buffer_size`, several concurrent
/// compaction tasks, and enough fragments per task that one task's scan
/// spans a large chunk of the dataset - the shape the issue points at.
///
/// `io_buffer_size` is parametrized across a few small values since the
/// exact threshold at which the byte-budget inversion gets forced depends
/// on internal scheduling details this black-box test can't control
/// directly.
#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_compaction_many_small_wide_fragments_does_not_deadlock(
    #[values(1024, 4 * 1024, 16 * 1024)] io_buffer_size: u64,
) {
    let store_params =
        latency_injecting_store_params(Duration::from_millis(1), Duration::from_micros(1));
    let mut dataset = write_many_small_wide_fragments(Some(store_params)).await;
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        // Bin most of the small fragments together per task, so a single
        // task's merged scan is reading from ~100+ fragments at once.
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(io_buffer_size),
        num_threads: Some(8),
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;

    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with \
             io_buffer_size={io_buffer_size} bytes across {NUM_FRAGMENTS} fragments; this \
             looks like the scan-scheduler backpressure deadlock from \
             https://github.com/lance-format/lance/issues/9575 (many small fragments + a wide \
             schema exhausting the I/O byte budget before anything can be consumed)"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: runs the most aggressive config we
/// have (smallest `io_buffer_size`, injected latency) with the scheduler's
/// own `lance_io::scheduler::state` trace events turned on, so a run that
/// *doesn't* deadlock can still be inspected for how close it got (deeply
/// negative `bytes_available`, `head_task_blocked_by_bytes=true` stretches,
/// etc. - see `lance-io/src/scheduler.rs`'s `SchedulerStateEvent`). Ignored
/// by default since its only purpose is to be run by hand with
/// `RUST_LOG=lance_io::scheduler::state=trace`.
#[ignore = "diagnostic only; run by hand with RUST_LOG=lance_io::scheduler::state=trace"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn diagnostic_trace_scheduler_state_under_tightest_budget() {
    // `test_log::test` only bridges the `log` facade by default (its `trace`
    // feature, which would bridge `tracing` events too, isn't enabled
    // anywhere in this workspace), so `SchedulerStateEvent`'s
    // `tracing::event!` calls need a `tracing-subscriber` wired up directly.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();

    // Jittered (independently randomized per call), not uniform, latency -
    // see `JitteredLatencyStore`'s doc comment for why. Of the ranges tried
    // by hand (200us-1ms uniform [no jitter]; 200us-5ms; 100us-200ms), the
    // 200us-5ms range produced the most scheduler contention (see "Status"
    // above) and is kept here as the default; widening it further did not
    // increase contention further in these runs.
    let store_params =
        jittered_latency_store_params(Duration::from_micros(200), Duration::from_millis(5));
    let mut dataset = write_many_small_wide_fragments(Some(store_params)).await;

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(1024),
        num_threads: Some(8),
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    match result {
        Ok(res) => {
            res.expect("compaction returned an error");
        }
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?}; this looks like the \
             scan-scheduler backpressure deadlock from \
             https://github.com/lance-format/lance/issues/9575"
        ),
    }
}

/// Diagnostic, not a regression test: tests whether *structural* fragment
/// concurrency alone - no injected latency at all - can trigger the
/// deadlock, by cranking up fragment readahead instead of relying on I/O
/// timing.
///
/// Reading `rust/lance/src/io/exec/scan.rs`'s V2 scan path
/// (`LanceStream::try_new_v2`) shows every fragment in one scan shares a
/// *single* `ScanScheduler` (one `io_buffer_size` budget), and fragments are
/// opened/scheduled `frag_parallelism` at a time via `.try_buffered(..)`.
/// `frag_parallelism` defaults to `LANCE_DEFAULT_FRAGMENT_READAHEAD`, which
/// defaults to just **4** - so every earlier test in this file, despite
/// using hundreds of fragments, only ever had 4 of them open and competing
/// for the shared budget at once. Each fragment's I/O carries a priority of
/// `(fragment_index << 64) | top_level_row_number`, so as long as fragments
/// are consumed in the same order they're opened, the lowest-priority
/// (= earliest-needed) request is always admitted regardless of budget -
/// which is exactly why 4-way concurrency never deadlocked. This test
/// raises `LANCE_DEFAULT_FRAGMENT_READAHEAD` well past the fragment count,
/// so (per that same code's own comment - "we don't really need to worry
/// too much about fragment readahead... we also want this to be pretty
/// high") effectively the whole task's fragment set is opened and
/// scheduling I/O at once against one tiny shared budget.
///
/// `LANCE_DEFAULT_FRAGMENT_READAHEAD` is read through a `LazyLock`, cached
/// for the life of the process on first use - so this only works if no
/// scan has happened yet in this process, which holds when this test is
/// run by itself (e.g. `cargo test --test compaction_io_buffer_deadlock
/// diagnostic_high_fragment_readahead_no_latency`), not as part of the
/// whole suite.
#[ignore = "diagnostic only; run in isolation, e.g. `cargo test --test compaction_io_buffer_deadlock diagnostic_high_fragment_readahead_no_latency -- --ignored`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_high_fragment_readahead_no_latency() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    // `scan.rs`'s "fragments at once" line goes through `log::debug!`, not
    // `tracing`, so it needs its own logger to show up.
    let _ = env_logger::try_init();

    // Safety: nothing else touches process env in this test binary
    // concurrently with this call, and this runs before any scan (and
    // therefore before `DEFAULT_FRAGMENT_READAHEAD`'s `LazyLock` is forced)
    // as long as the test is run in isolation per the doc comment above.
    unsafe {
        std::env::set_var(
            "LANCE_DEFAULT_FRAGMENT_READAHEAD",
            (NUM_FRAGMENTS * 2).to_string(),
        );
    }

    // No latency injection at all - plain in-memory store. If this
    // reproduces, timing was never the missing ingredient.
    let mut dataset = write_many_small_wide_fragments(None).await;
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(1024),
        num_threads: Some(8),
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with \
             LANCE_DEFAULT_FRAGMENT_READAHEAD={} and no injected latency; this looks like the \
             scan-scheduler backpressure deadlock from \
             https://github.com/lance-format/lance/issues/9575, and it is driven purely by \
             structural fragment concurrency rather than I/O timing",
            NUM_FRAGMENTS * 2
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: targets a specific, code-grounded
/// amplifier rather than I/O timing or raw fragment concurrency.
///
/// Reading `lance-encoding/src/array_encoding/logical/list.rs` shows that
/// scheduling a list column's *items* (as opposed to its offsets) happens
/// via a bare, untracked `tokio::spawn` per list column per batch per open
/// fragment (`indirect_schedule_task`). Each one becomes its own
/// independently-bypassed "in-flight priority class" per
/// `IoQueueState::can_deliver_without_warning`'s escape valve the moment it
/// submits its first item request - individually safe, but with a
/// majority-list-column, skewed-length (mostly-empty, occasionally-huge)
/// schema across many concurrently-open fragments, the number of
/// simultaneously bypassed classes - and the bytes they collectively
/// reserve - could run far past what a moderate `io_buffer_size` was sized
/// for, for long stretches, even though no single admission technically
/// violates the priority invariant.
///
/// Also addresses a confound in the earlier tests: with `io_buffer_size` as
/// tiny as 1 KiB, essentially every read already exceeds the budget on its
/// own (writer pages default to megabytes), so *everything* was going
/// through the head-task bypass regardless of priority games - there was no
/// room for genuine multi-request admit/reject contention. This test writes
/// with small `max_page_bytes`/`data_cache_bytes` (64 KiB) so a few-hundred
/// KiB `io_buffer_size` is a real, multi-page budget instead of a
/// rounding error.
#[ignore = "diagnostic only; run in isolation, e.g. `cargo test --test compaction_io_buffer_deadlock diagnostic_skewed_list_columns_no_latency -- --ignored`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_skewed_list_columns_no_latency() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    // Combine both confirmed amplifiers: skewed list columns (142 peak
    // concurrent in-flight priority classes and a >1 MiB budget excursion
    // against a 256 KiB budget, measured with readahead at its default of
    // 4) and high fragment readahead (so many more fragments' list columns
    // are spawning indirect item-schedule tasks concurrently, not just 4
    // fragments' worth). No latency injection at all - purely structural.
    //
    // Safety / ordering requirement: see
    // `diagnostic_high_fragment_readahead_no_latency` above - this must run
    // before any scan in the process, i.e. in isolation.
    unsafe {
        std::env::set_var(
            "LANCE_DEFAULT_FRAGMENT_READAHEAD",
            (NUM_FRAGMENTS * 2).to_string(),
        );
    }

    let mut dataset = write_skewed_list_heavy_fragments().await;
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        // Tighter than before (64 KiB, matching the page size) now that
        // pages are realistically sized rather than ~8 MiB.
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with a skewed, \
             majority-list-column schema and no injected latency; this looks like the \
             scan-scheduler backpressure deadlock from \
             https://github.com/lance-format/lance/issues/9575, driven by list-item scheduling \
             concurrency rather than I/O timing"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: adds the one piece of the issue's
/// dataset shape every other test in this file is still missing - "one
/// BTree scalar index" - and runs compaction with `defer_index_remap:
/// false` (eager remap), matching the issue's own Python repro verbatim
/// (`defer_index_remap=False`).
///
/// This matters because every other test here used only `CompactionOptions`
/// knobs that affect the *data* scan. With an index present and stable row
/// IDs (as here), `commit_compaction`'s `records_stable_row_id_moves`
/// (`rust/lance/src/dataset/optimize.rs`) is a no-op when
/// `defer_index_remap` is false and no task recorded row-address moves - so
/// this isn't expected to add extra *index-remapping* I/O after the
/// rewrite. What it *does* add is the index's own files (and whatever
/// `plan_compaction`/`commit_compaction` reads from them, e.g.
/// `load_all_indices`/`load_index_fragmaps`) being open and potentially
/// read *concurrently* with the main data scan - a second, independent
/// operation sharing process state with the first, which is a genuinely
/// different situation than the single-scan priority-ordering safety
/// property the earlier diagnostics relied on (that property says nothing
/// about two independent operations contending for one scheduler).
///
/// No injected latency, no readahead override, no list-column skew - just
/// the baseline many-small-fragments/wide-schema dataset plus an index,
/// isolating the index's effect specifically.
#[ignore = "diagnostic only; run in isolation, e.g. `cargo test --test compaction_io_buffer_deadlock diagnostic_compaction_with_scalar_index -- --ignored`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_compaction_with_scalar_index() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    let mut dataset = write_many_small_wide_fragments(None).await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(1024),
        num_threads: Some(8),
        // Matches the issue's own Python repro: `defer_index_remap=False`.
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with a BTree scalar \
             index present and defer_index_remap=false; this looks like the scan-scheduler \
             backpressure deadlock from https://github.com/lance-format/lance/issues/9575, \
             triggered by the index rather than pure data-scan concurrency"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: combines every amplifier found so far
/// - a BTree scalar index (this message), the skewed majority-list-column
/// schema (`diagnostic_skewed_list_columns_no_latency`), and high fragment
/// readahead (`diagnostic_high_fragment_readahead_no_latency`) - in one run.
/// Still zero injected latency.
#[ignore = "diagnostic only; run in isolation, e.g. `cargo test --test compaction_io_buffer_deadlock diagnostic_skewed_list_columns_with_scalar_index -- --ignored`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_skewed_list_columns_with_scalar_index() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    // Safety: see `diagnostic_high_fragment_readahead_no_latency` - must run
    // before any scan in the process, i.e. in isolation.
    unsafe {
        std::env::set_var(
            "LANCE_DEFAULT_FRAGMENT_READAHEAD",
            (NUM_FRAGMENTS * 2).to_string(),
        );
    }

    let mut dataset = write_skewed_list_heavy_fragments().await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} combining a BTree \
             scalar index, a skewed majority-list-column schema, and high fragment readahead, \
             with no injected latency; this looks like the scan-scheduler backpressure \
             deadlock from https://github.com/lance-format/lance/issues/9575"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Control case: `io_buffer_size = 0` puts the scheduler in
/// `no_backpressure` mode, which the issue reports as a workaround for the
/// deadlock. If this also failed to complete, the dataset shape above
/// wouldn't actually be exercising the code path the issue describes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_compaction_many_small_wide_fragments_with_no_backpressure_completes() {
    let store_params =
        latency_injecting_store_params(Duration::from_millis(1), Duration::from_micros(1));
    let mut dataset = write_many_small_wide_fragments(Some(store_params)).await;
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 4_000,
        batch_size: Some(512),
        io_buffer_size: Some(0),
        num_threads: Some(4),
        ..Default::default()
    };

    let metrics =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None))
            .await
            .expect("compaction with io_buffer_size=0 (no_backpressure) should not deadlock")
            .expect("compaction returned an error");
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: a completely different deadlock class
/// than every other diagnostic in this file - CPU-pool exhaustion, not
/// `IoQueueState` byte-budget admission.
///
/// `lance-core/src/utils/tokio.rs`'s `spawn_cpu` dispatches CPU-bound work
/// (page encode/decode, index prefilter computation, ...) onto one *global*,
/// *process-wide* blocking-thread pool sized by `get_num_compute_intensive_
/// cpus()` = `max(1, num_cpus - LANCE_IO_CORE_RESERVATION)` (reservation
/// defaults to 2). Its own doc comment is explicit: on a host with <= 3
/// visible CPUs (a small cloud VM, a CPU-limited k8s pod - a very plausible
/// shape for an S3-backed production job, much more so than this dev
/// machine), that pool collapses to **exactly one worker thread**. If any
/// `spawn_cpu` closure, directly or transitively, ever blocks/waits/parks
/// (a channel, a lock, I/O, or driving another async task) while occupying
/// that single thread, the pool can starve itself forever: 0% CPU, no I/O,
/// no timeout, no error - which matches the issue's gdb snapshot ("every
/// lance runtime thread in futex_wait... 0% CPU") far more precisely than
/// the byte-budget scheduler's "slow but still visibly working" behavior
/// under the stress tests above. This exact bug class has a confirmed
/// precedent in this codebase (PR #7423, referenced in `spawn_cpu`'s own
/// doc comment) and a related, already-fixed cache deadlock of the same
/// "single-flight loader waits on itself" shape (PR #8620,
/// `rust/lance/src/index.rs`'s `open_frag_reuse_index` tests).
///
/// `LANCE_CPU_THREADS` is the same public, documented env var `spawn_cpu`'s
/// own doc points at for sizing this pool, so forcing it to `1` here
/// reproduces the small-host condition without needing an actual
/// resource-constrained machine. Combines with the index (exercises
/// `index/prefilter.rs`'s `spawn_cpu`-dispatched allow-list computation,
/// cached via `dataset.metadata_cache` - a shape very close to the fixed
/// #8620 bug) and the list-heavy schema (exercises `encode_page`'s
/// `spawn_cpu`, `lance-encoding/src/encodings/logical/primitive.rs:7641`,
/// once per page across many columns). No injected latency.
///
/// Like the fragment-readahead/CPU-pool-sizing diagnostics above,
/// `NUM_COMPUTE_INTENSIVE_CPUS` is a `LazyLock` cached for the process
/// lifetime on first use, so this only works run in isolation, before any
/// other `spawn_cpu` call in the process.
#[ignore = "diagnostic only; run in isolation, e.g. `cargo test --test compaction_io_buffer_deadlock diagnostic_collapsed_cpu_pool_no_latency -- --ignored`"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_collapsed_cpu_pool_no_latency() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    // Safety: must run before any `spawn_cpu` call in the process (which
    // forces `NUM_COMPUTE_INTENSIVE_CPUS`'s `LazyLock`), i.e. in isolation -
    // see the doc comment above.
    unsafe {
        std::env::set_var("LANCE_CPU_THREADS", "1");
    }

    let mut dataset = write_skewed_list_heavy_fragments().await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with \
             LANCE_CPU_THREADS=1 (a collapsed CPU pool), a BTree scalar index, and a \
             list-heavy schema; this looks like the CPU-pool self-starvation deadlock \
             documented in lance-core/src/utils/tokio.rs's spawn_cpu (see PR #7423), not the \
             byte-budget scheduler from https://github.com/lance-format/lance/issues/9575 - \
             check CPU usage during the hang to confirm (0% CPU = this; spinning = something \
             else)"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Builds `ObjectStoreParams` for a real, S3-compatible backend (tested
/// against a local single-node Garage server - a maintained, S3-API-
/// compatible object store; MinIO's open-source server binary was archived
/// and is no longer distributed) from environment variables, so no
/// credentials are hardcoded here:
///
/// - `LANCE_TEST_S3_ENDPOINT` (e.g. `http://127.0.0.1:3900`)
/// - `LANCE_TEST_S3_REGION`
/// - `LANCE_TEST_S3_ACCESS_KEY_ID`
/// - `LANCE_TEST_S3_SECRET_ACCESS_KEY`
///
/// Path-style addressing and plain HTTP are forced on, matching a typical
/// local S3-compatible server with no TLS and no per-bucket subdomain DNS.
fn s3_store_params_from_env() -> ObjectStoreParams {
    s3_store_params_from_env_with_jitter(None)
}

/// Like [`s3_store_params_from_env`], optionally layering
/// [`JitteredLatencyStore`] (`(min, max)`) on top of the real S3 connection.
/// A local Garage server is loopback-fast (sub-millisecond), nothing like
/// real S3's actual round-trip time, so this combines genuine socket/HTTP
/// semantics with a realistic latency distribution instead of having to
/// choose one or the other.
fn s3_store_params_from_env_with_jitter(jitter: Option<(Duration, Duration)>) -> ObjectStoreParams {
    let endpoint = std::env::var("LANCE_TEST_S3_ENDPOINT")
        .expect("LANCE_TEST_S3_ENDPOINT must be set to run the S3-backed diagnostic");
    let region = std::env::var("LANCE_TEST_S3_REGION").unwrap_or_else(|_| "garage".to_string());
    let access_key_id = std::env::var("LANCE_TEST_S3_ACCESS_KEY_ID")
        .expect("LANCE_TEST_S3_ACCESS_KEY_ID must be set to run the S3-backed diagnostic");
    let secret_access_key = std::env::var("LANCE_TEST_S3_SECRET_ACCESS_KEY")
        .expect("LANCE_TEST_S3_SECRET_ACCESS_KEY must be set to run the S3-backed diagnostic");

    let mut options = std::collections::HashMap::new();
    options.insert("aws_endpoint".to_string(), endpoint);
    options.insert("aws_region".to_string(), region);
    options.insert("aws_access_key_id".to_string(), access_key_id);
    options.insert("aws_secret_access_key".to_string(), secret_access_key);
    options.insert("aws_allow_http".to_string(), "true".to_string());
    options.insert(
        "aws_virtual_hosted_style_request".to_string(),
        "false".to_string(),
    );

    ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            options,
        ))),
        object_store_wrapper: jitter.map(|(min_latency, max_latency)| {
            Arc::new(JitteredLatencyStore {
                min_latency,
                max_latency,
            }) as Arc<dyn WrappingObjectStore>
        }),
        ..Default::default()
    }
}

/// Diagnostic, not a regression test: everything in this file so far has run
/// against `memory://` or a synthetic latency wrapper around it. The issue
/// was only ever observed against real S3; this is the first test that
/// actually goes over a real socket, against a local S3-compatible server
/// (see [`s3_store_params_from_env`]). Real network I/O has genuinely
/// different completion-order and connection-pooling behavior than anything
/// a local process can synthesize, so this exercises a dimension none of
/// the earlier diagnostics could.
///
/// Combines the richest dataset shape tried so far (skewed majority-list
/// schema + BTree scalar index) with a real S3-compatible backend. Requires
/// `LANCE_TEST_S3_ENDPOINT`/`_REGION`/`_ACCESS_KEY_ID`/`_SECRET_ACCESS_KEY`
/// to be set (see `s3_store_params_from_env`); skipped (via `assert!`
/// panic with a clear message) rather than silently no-op'd if they aren't,
/// since a silently-skipped "did not deadlock" would be misleading.
#[ignore = "diagnostic only; requires a running S3-compatible server, see s3_store_params_from_env"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_skewed_list_columns_with_scalar_index_on_s3() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    let bucket =
        std::env::var("LANCE_TEST_S3_BUCKET").unwrap_or_else(|_| "lance-deadlock-test".into());
    let run_id = uuid::Uuid::new_v4();
    let uri = format!("s3://{bucket}/compaction-deadlock-repro/{run_id}");

    let mut dataset =
        write_skewed_list_heavy_fragments_at(&uri, Some(s3_store_params_from_env())).await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} against a real \
             S3-compatible backend with a BTree scalar index and a skewed majority-list-column \
             schema; this looks like the scan-scheduler backpressure deadlock from \
             https://github.com/lance-format/lance/issues/9575 - check CPU/network activity \
             during the hang: 0% CPU and no bytes in flight matches the issue precisely"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: stacks every amplifier found across
/// this whole investigation in one run - real S3-compatible sockets with
/// realistic jittered latency (`diagnostic_skewed_list_columns_with_
/// scalar_index_on_s3`), a collapsed CPU pool
/// (`diagnostic_collapsed_cpu_pool_no_latency`), high fragment readahead
/// (`diagnostic_high_fragment_readahead_no_latency`), and the skewed
/// majority-list-column schema + BTree scalar index. If anything in this
/// file reproduces the issue, this combination is the best shot.
#[ignore = "diagnostic only; requires a running S3-compatible server, see s3_store_params_from_env"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_everything_combined_on_s3() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    // Safety: must run before any `spawn_cpu`/scan call in the process - see
    // `diagnostic_collapsed_cpu_pool_no_latency` / `diagnostic_high_fragment_
    // readahead_no_latency` above. Run in isolation.
    unsafe {
        std::env::set_var("LANCE_CPU_THREADS", "1");
        std::env::set_var(
            "LANCE_DEFAULT_FRAGMENT_READAHEAD",
            (NUM_FRAGMENTS * 2).to_string(),
        );
    }

    let bucket =
        std::env::var("LANCE_TEST_S3_BUCKET").unwrap_or_else(|_| "lance-deadlock-test".into());
    let run_id = uuid::Uuid::new_v4();
    let uri = format!("s3://{bucket}/compaction-deadlock-repro/{run_id}");
    let store_params = s3_store_params_from_env_with_jitter(Some((
        Duration::from_millis(5),
        Duration::from_millis(50),
    )));

    let mut dataset = write_skewed_list_heavy_fragments_at(&uri, Some(store_params)).await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with every amplifier \
             combined (real S3 + jitter, collapsed CPU pool, high fragment readahead, skewed \
             list-heavy schema, BTree scalar index); this looks like the scan-scheduler \
             backpressure deadlock from https://github.com/lance-format/lance/issues/9575 - \
             check CPU/network activity during the hang: 0% CPU and no bytes in flight matches \
             the issue precisely"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: the one piece of the issue's dataset
/// shape every other test in this file is still missing - the *bimodal*
/// fragment-size distribution (`write_bimodal_fragments_at`) - combined
/// with real S3 and a BTree scalar index that, this time, covers the large
/// fragments too (built *after* both writes land, so indexing touches
/// `NUM_FRAGMENTS * ROWS_PER_FRAGMENT + BIMODAL_NUM_LARGE_FRAGMENTS *
/// BIMODAL_LARGE_FRAGMENT_ROWS` rows - the largest index any test in this
/// file has built). `target_rows_per_fragment: 1500` matches the issue's
/// own repro value verbatim, which (being far below
/// `BIMODAL_LARGE_FRAGMENT_ROWS`) keeps the large fragments out of the
/// rewrite task itself, exactly as the issue's own repro does - so this
/// isolates whatever effect the large fragments have merely by being
/// *part of the dataset and the index* while compaction runs, as opposed
/// to being scanned by it directly.
#[ignore = "diagnostic only; requires a running S3-compatible server, see s3_store_params_from_env"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_bimodal_fragments_with_scalar_index_on_s3() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    let bucket =
        std::env::var("LANCE_TEST_S3_BUCKET").unwrap_or_else(|_| "lance-deadlock-test".into());
    let run_id = uuid::Uuid::new_v4();
    let uri = format!("s3://{bucket}/compaction-deadlock-repro/{run_id}");
    let store_params = s3_store_params_from_env_with_jitter(Some((
        Duration::from_millis(5),
        Duration::from_millis(50),
    )));

    let mut dataset = write_bimodal_fragments_at(&uri, Some(store_params)).await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        // Matches the issue's own Python repro verbatim: well under
        // `BIMODAL_LARGE_FRAGMENT_ROWS`, so only the small fragments
        // qualify for the rewrite task.
        target_rows_per_fragment: 1_500,
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with a bimodal \
             fragment-size dataset (hundreds tiny + {BIMODAL_NUM_LARGE_FRAGMENTS} at \
             {BIMODAL_LARGE_FRAGMENT_ROWS} rows each), a BTree scalar index covering all of it, \
             real S3 + jitter, and target_rows_per_fragment=1500 (matching the issue's own \
             repro, excluding the large fragments from the rewrite task itself); this looks \
             like the scan-scheduler backpressure deadlock from \
             https://github.com/lance-format/lance/issues/9575 - check CPU/network activity \
             during the hang: 0% CPU and no bytes in flight matches the issue precisely"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}

/// Diagnostic, not a regression test: the final combination - every
/// amplifier found across this whole investigation, on top of the one
/// dataset shape ([`write_bimodal_fragments_at`]) that gets genuinely
/// GB-scale ("the issue needed ~1 GiB read before stalling" - this dataset
/// is ~2.8 GiB, confirmed by listing the bucket after a run). Real S3 +
/// jitter, a collapsed CPU pool, high fragment readahead, the bimodal
/// fragment-size shape, and a BTree scalar index covering the whole
/// dataset including the large fragments. If anything in this file
/// reproduces the issue, this is the best shot.
#[ignore = "diagnostic only; requires a running S3-compatible server, see s3_store_params_from_env"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn diagnostic_everything_combined_bimodal_on_s3() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let _ = env_logger::try_init();

    // Safety: must run before any `spawn_cpu`/scan call in the process - see
    // `diagnostic_collapsed_cpu_pool_no_latency` / `diagnostic_high_fragment_
    // readahead_no_latency` above. Run in isolation.
    unsafe {
        std::env::set_var("LANCE_CPU_THREADS", "1");
        std::env::set_var(
            "LANCE_DEFAULT_FRAGMENT_READAHEAD",
            (NUM_FRAGMENTS * 2).to_string(),
        );
    }

    let bucket =
        std::env::var("LANCE_TEST_S3_BUCKET").unwrap_or_else(|_| "lance-deadlock-test".into());
    let run_id = uuid::Uuid::new_v4();
    let uri = format!("s3://{bucket}/compaction-deadlock-repro/{run_id}");
    let store_params = s3_store_params_from_env_with_jitter(Some((
        Duration::from_millis(5),
        Duration::from_millis(50),
    )));

    let mut dataset = write_bimodal_fragments_at(&uri, Some(store_params)).await;
    dataset
        .create_index(
            &["plain_0"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .expect("failed to create scalar index");
    let total_rows_before = dataset.count_rows(None).await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 1_500,
        io_buffer_size: Some(32 * 1024),
        num_threads: Some(8),
        defer_index_remap: false,
        ..Default::default()
    };

    let result =
        tokio::time::timeout(DEADLOCK_TIMEOUT, compact_files(&mut dataset, options, None)).await;
    let metrics = match result {
        Ok(res) => res.expect("compaction returned an error"),
        Err(_) => panic!(
            "compact_files did not complete within {DEADLOCK_TIMEOUT:?} with every amplifier \
             found in this investigation combined on a ~GB-scale, bimodal-fragment-size \
             dataset: real S3 + jitter, collapsed CPU pool, high fragment readahead, hundreds \
             of tiny fragments plus {BIMODAL_NUM_LARGE_FRAGMENTS} large ones, and a BTree \
             scalar index covering the whole dataset. This looks like the scan-scheduler \
             backpressure deadlock from https://github.com/lance-format/lance/issues/9575 - \
             check CPU/network activity during the hang: 0% CPU and no bytes in flight matches \
             the issue precisely"
        ),
    };
    assert!(metrics.fragments_removed > 0, "compaction made no progress");

    let total_rows_after = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows_after, total_rows_before,
        "compaction changed the row count"
    );
}
