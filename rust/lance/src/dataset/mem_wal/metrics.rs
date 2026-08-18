// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL write-path metrics, emitted through the [`metrics`] facade.
//!
//! Mirrors `lance_io::object_store::metrics`: observations route to whatever
//! [`metrics::Recorder`] the embedding process installed, so this crate takes no
//! position on the exporter. Feature-gated behind `metrics`; with the feature
//! off the emit sites compile away.
//!
//! Only flush *durations* live here. Counts and byte totals stay on
//! [`super::write::WriteStats`], which an embedder can already poll: those are
//! cumulative and lose nothing to sampling. A duration does — an average
//! reconstructed from a running total cannot show a tail — so it needs an
//! observation per flush, which is what this module provides.

/// Flush latency in seconds, by `kind`. One family rather than two because the
/// two flushes are stages of the same write pipeline and are read together.
pub const METRIC_FLUSH_DURATION: &str = "lance_mem_wal_flush_duration_seconds";

/// `kind` on [`METRIC_FLUSH_DURATION`]: the WAL buffer landing in object
/// storage. This is the latency an acked write waits on when `durable_write`.
pub const KIND_WAL: &str = "wal";

/// `kind` on [`METRIC_FLUSH_DURATION`]: a sealed memtable becoming an L0
/// SSTable. Orders of magnitude longer than a WAL flush, hence shared buckets
/// spanning both.
pub const KIND_MEMTABLE: &str = "memtable";

/// Bucket bounds spanning both kinds: a WAL flush is one object-store PUT
/// (single-digit ms to a few hundred), a memtable flush writes a whole dataset
/// (hundreds of ms to tens of seconds).
pub const FLUSH_DURATION_BOUNDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

/// Register the description for [`METRIC_FLUSH_DURATION`].
///
/// Routes through the installed recorder, so call it *after* the recorder is
/// set. Exporters that build a catalog up front (the OpenTelemetry bridge, for
/// one) need this to discover the metric's name, kind and unit.
#[cfg(feature = "metrics")]
pub fn describe_metrics() {
    metrics::describe_histogram!(
        METRIC_FLUSH_DURATION,
        metrics::Unit::Seconds,
        "MemWAL flush latency in seconds, by kind (wal buffer, or memtable to L0)."
    );
}

/// Observe one completed flush. No-op without the `metrics` feature.
#[cfg(feature = "metrics")]
pub(crate) fn record_flush_duration(kind: &'static str, duration: std::time::Duration) {
    metrics::histogram!(METRIC_FLUSH_DURATION, "kind" => kind).record(duration.as_secs_f64());
}

/// Observe one completed flush. No-op without the `metrics` feature.
#[cfg(not(feature = "metrics"))]
pub(crate) fn record_flush_duration(_kind: &'static str, _duration: std::time::Duration) {}
