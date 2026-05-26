// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::{Future, FutureExt, Stream, StreamExt, future::BoxFuture, stream::BoxStream};
use pin_project::pin_project;
use tracing::{Instrument, Span};

#[pin_project]
pub struct InstrumentedStream<I: Stream> {
    #[pin]
    stream: I,
    span: Span,
}

impl<I: Stream> Stream for InstrumentedStream<I> {
    type Item = I::Item;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.project();
        let _guard = this.span.enter();
        this.stream.poll_next(cx)
    }
}

// It would be nice to call the method in_current_span but sadly the Instrumented trait in
// the tracing crate already stole the name for all Sized types
pub trait FutureTracingExt {
    /// All calls to poll the future will be done in the context of the current span (when this method is called)
    fn future_in_current_span(self) -> tracing::instrument::Instrumented<Self>
    where
        Self: Future,
        Self: Sized;

    fn future_in_span(self, span: Span) -> tracing::instrument::Instrumented<Self>
    where
        Self: Future,
        Self: Sized;

    fn boxed_in_current_span(self) -> BoxFuture<'static, <Self as Future>::Output>
    where
        Self: Future + Send + 'static,
        Self: Sized;

    fn boxed_in_span(self, span: Span) -> BoxFuture<'static, <Self as Future>::Output>
    where
        Self: Future + Send + 'static,
        Self: Sized;
}

impl<F: Future> FutureTracingExt for F {
    fn future_in_current_span(self) -> tracing::instrument::Instrumented<Self>
    where
        Self: Future,
        Self: Sized,
    {
        self.future_in_span(Span::current())
    }

    fn future_in_span(self, span: Span) -> tracing::instrument::Instrumented<Self>
    where
        Self: Future,
        Self: Sized,
    {
        self.instrument(span)
    }

    fn boxed_in_current_span(self) -> BoxFuture<'static, <Self as Future>::Output>
    where
        Self: Future + Send + 'static,
        Self: Sized,
    {
        self.boxed_in_span(Span::current())
    }

    fn boxed_in_span(self, span: Span) -> BoxFuture<'static, <Self as Future>::Output>
    where
        Self: Future + Send + 'static,
        Self: Sized,
    {
        self.instrument(span).boxed()
    }
}

pub trait StreamTracingExt {
    /// All calls to poll the stream will be done in the context of the current span (when this method is called)
    fn stream_in_current_span(self) -> InstrumentedStream<Self>
    where
        Self: Stream,
        Self: Sized;

    fn stream_in_span(self, span: Span) -> InstrumentedStream<Self>
    where
        Self: Stream,
        Self: Sized;

    fn boxed_stream_in_current_span(self) -> BoxStream<'static, <Self as Stream>::Item>
    where
        Self: Stream + Send + 'static,
        Self: Sized;

    fn boxed_stream_in_span(self, span: Span) -> BoxStream<'static, <Self as Stream>::Item>
    where
        Self: Stream + Send + 'static,
        Self: Sized;
}

impl<S: Stream> StreamTracingExt for S {
    fn stream_in_current_span(self) -> InstrumentedStream<Self>
    where
        Self: Stream,
        Self: Sized,
    {
        self.stream_in_span(Span::current())
    }

    fn stream_in_span(self, span: Span) -> InstrumentedStream<Self>
    where
        Self: Stream,
        Self: Sized,
    {
        InstrumentedStream { stream: self, span }
    }

    fn boxed_stream_in_current_span(self) -> BoxStream<'static, <Self as Stream>::Item>
    where
        Self: Stream + Send + 'static,
        Self: Sized,
    {
        self.boxed_stream_in_span(Span::current())
    }

    fn boxed_stream_in_span(self, span: Span) -> BoxStream<'static, <Self as Stream>::Item>
    where
        Self: Stream + Send + 'static,
        Self: Sized,
    {
        self.stream_in_span(span).boxed()
    }
}

pub const TRACE_FILE_AUDIT: &str = "lance::file_audit";
pub const AUDIT_MODE_CREATE: &str = "create";
pub const AUDIT_MODE_DELETE: &str = "delete";
pub const AUDIT_MODE_DELETE_UNVERIFIED: &str = "delete_unverified";
pub const AUDIT_TYPE_DELETION: &str = "deletion";
pub const AUDIT_TYPE_MANIFEST: &str = "manifest";
pub const AUDIT_TYPE_INDEX: &str = "index";
pub const AUDIT_TYPE_DATA: &str = "data";
pub const TRACE_FILE_CREATE: &str = "create";
pub const TRACE_IO_EVENTS: &str = "lance::io_events";
pub const IO_TYPE_OPEN_SCALAR: &str = "open_scalar_index";
pub const IO_TYPE_OPEN_VECTOR: &str = "open_vector_index";
pub const IO_TYPE_OPEN_FRAG_REUSE: &str = "open_frag_reuse_index";
pub const IO_TYPE_OPEN_MEM_WAL: &str = "open_mem_wal_index";
pub const IO_TYPE_LOAD_VECTOR_PART: &str = "load_vector_part";
pub const IO_TYPE_LOAD_SCALAR_PART: &str = "load_scalar_part";
pub const IO_TYPE_LOAD_DATA_FILE: &str = "load_data_file";
pub const IO_TYPE_LOAD_MANIFEST: &str = "load_manifest";
pub const IO_TYPE_LOAD_DELETION: &str = "load_deletion";
pub const TRACE_EXECUTION: &str = "lance::execution";
pub const EXECUTION_PLAN_RUN: &str = "plan_run";

/// Target for the per-query aggregated profile event emitted by
/// [`crate::utils::profile::QueryProfileLayer`] when a `query` root span closes.
pub const TRACE_QUERY_PROFILE: &str = "lance::query_profile";
/// Name of the root span the profile layer aggregates under. Created at the
/// scanner stream entry points.
pub const QUERY_PROFILE_ROOT: &str = "query";

/// Names of the high-level phase spans recognized by the profile layer.
/// Spans with any other name are ignored by the aggregator.
///
/// Sub-phase spans use the form `phase.<kind>.<subkind>` and roll up into the
/// corresponding `phase.<kind>` total — both are reported in the summary
/// event.
pub const PHASE_PLAN: &str = "phase.plan";
pub const PHASE_OPEN_INDEX: &str = "phase.open_index";
pub const PHASE_OPEN_INDEX_SCALAR: &str = "phase.open_index.scalar";
pub const PHASE_OPEN_INDEX_VECTOR: &str = "phase.open_index.vector";
pub const PHASE_OPEN_INDEX_FRAG_REUSE: &str = "phase.open_index.frag_reuse";
pub const PHASE_OPEN_INDEX_MEM_WAL: &str = "phase.open_index.mem_wal";
pub const PHASE_INDEX_SEARCH: &str = "phase.index_search";
pub const PHASE_INDEX_SEARCH_SCALAR: &str = "phase.index_search.scalar";
pub const PHASE_INDEX_SEARCH_FTS: &str = "phase.index_search.fts";
pub const PHASE_INDEX_SEARCH_ANN: &str = "phase.index_search.ann";
pub const PHASE_LOAD_DATA: &str = "phase.load_data";
pub const PHASE_LOAD_DATA_FILTERED_READ: &str = "phase.load_data.filtered_read";
pub const PHASE_LOAD_DATA_LANCE_SCAN: &str = "phase.load_data.lance_scan";
pub const PHASE_LOAD_DATA_TAKE: &str = "phase.load_data.take";
pub const PHASE_POSTPROCESS: &str = "phase.postprocess";

/// Per-physical-IO event target. Emitted by the `lance-io` scheduler from
/// each `reader.get_range` completion with `bytes` and `duration_us` fields.
/// Consumed by [`crate::utils::profile::QueryProfileLayer`] to build IO
/// distribution stats (count / total / min / p50 / p95 / max for bytes,
/// duration, and throughput) per file type.
pub const TRACE_IO_PHYSICAL: &str = "lance::io_physical";

pub const TRACE_DATASET_EVENTS: &str = "lance::dataset_events";
pub const DATASET_WRITING_EVENT: &str = "writing";
pub const DATASET_COMMITTED_EVENT: &str = "committed";
pub const DATASET_DROPPING_COLUMN_EVENT: &str = "dropping_column";
pub const DATASET_DELETING_EVENT: &str = "deleting";
pub const DATASET_COMPACTING_EVENT: &str = "compacting";
pub const DATASET_CLEANING_EVENT: &str = "cleaning";
pub const DATASET_LOADING_EVENT: &str = "loading";
