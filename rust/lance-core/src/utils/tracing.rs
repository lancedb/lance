// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::Stream;
use pin_project::pin_project;
use tracing::Span;

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
pub trait StreamTracingExt {
    /// All calls to poll the stream will be done in the context of the current span (when this method is called)
    fn stream_in_current_span(self) -> InstrumentedStream<Self>
    where
        Self: Stream,
        Self: Sized;
}

impl<S: Stream> StreamTracingExt for S {
    fn stream_in_current_span(self) -> InstrumentedStream<Self>
    where
        Self: Stream,
        Self: Sized,
    {
        InstrumentedStream {
            stream: self,
            span: Span::current(),
        }
    }
}
