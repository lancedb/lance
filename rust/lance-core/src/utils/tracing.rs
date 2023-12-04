// Copyright 2023 Lance Developers.
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
