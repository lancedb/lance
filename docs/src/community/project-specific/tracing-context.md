# Tracing Context Propagation

Rust query execution in Lance crosses many async boundaries: boxed futures, boxed streams, `tokio::spawn`, and `spawn_blocking` tasks.
If those boundaries do not preserve the current span, distributed tracing backends (e.g. Tempo) will show `lance-rust` work as orphan root traces instead of attaching it to the original request trace.

## Architecture

### TracedExec

`TracedExec` (in `lance-datafusion/src/exec.rs`) is a DataFusion `ExecutionPlan` wrapper that captures the caller's tracing span at construction time and enters it during `execute()` and stream polls.

Because DataFusion plan nodes are **lazy** (they build inner streams on first poll, not at plan creation), downstream span-aware calls like `boxed_stream_in_current_span()` would observe `Span::current()` as `<none>` without this wrapper.

`execute_plan()` applies `TracedExec` automatically:

```rust
let traced_plan = Arc::new(TracedExec::new(plan.clone(), Span::current()));
let stream = traced_plan.execute(0, task_ctx)?;
```

### DataFusion JoinSet Tracing

`ensure_datafusion_task_tracing()` registers a `LanceJoinSetTracer` with DataFusion so that tasks spawned internally by DataFusion (e.g. in joins, aggregations) carry the current span.

## Rule of Thumb

Do not add new direct `.in_current_span()` calls in tracing-sensitive execution paths.
If a helper does not already exist for the boundary you are working on, add or extend a helper in `lance-core` instead of repeating manual span capture at each call site.

## Preferred Helpers

Use the shared helpers from `lance_core`:

| Boundary | Helper |
|---|---|
| `tokio::spawn(...)` | `spawn_in_current_span(...)` / `spawn_in_span(...)` |
| Non-boxed future | `.future_in_current_span()` / `.future_in_span(...)` |
| Boxed future | `.boxed_in_current_span()` / `.boxed_in_span(...)` |
| Non-boxed stream | `.stream_in_current_span()` / `.stream_in_span(...)` |
| Boxed stream | `.boxed_stream_in_current_span()` / `.boxed_stream_in_span(...)` |

These are provided by the `FutureTracingExt` and `StreamTracingExt` traits in `lance_core::utils::tracing`.

## Common Rewrites

### Capture-then-clone in stream/closure chains

`Span::current()` returns the span that is active **at the moment of the call**.
Inside a stream `.map()` closure, the "current" span depends on whoever is **polling** the stream, which may be a completely different context from the code that created the stream.

The correct pattern is to capture the span **once** before the stream chain, then clone it into each closure iteration:

```rust
let stream_span = Span::current();            // capture at creation time
stream
    .map(move |task| {
        let stream_span = stream_span.clone(); // clone into each iteration
        task.map(move |batch| transform(batch?))
            .boxed_in_span(stream_span)        // use the captured span
    })
    .boxed_stream_in_current_span()
```

**Wrong** — using `_in_current_span()` inside the closure:

```rust
// BUG: Span::current() here is the *poller's* span, not the creator's
stream
    .map(move |task| {
        task.map(move |batch| transform(batch?))
            .boxed_in_current_span()
    })
    .boxed_stream_in_current_span()
```

The same principle applies to any `move` closure that will execute later (e.g. callbacks, `then`, `and_then`).
As a rule: if the closure crosses an async boundary, capture the span outside and pass it in.

### Spawning work

Instead of:

```rust
tokio::spawn(async move {
    do_work().await
}.in_current_span())
```

Use:

```rust
spawn_in_current_span(async move {
    do_work().await
})
```

### Boxing a future

Instead of:

```rust
let fut = async move {
    load_page().await
}
.in_current_span()
.boxed();
```

Use:

```rust
let fut = async move {
    load_page().await
}
.boxed_in_current_span();
```

### Boxing a stream

Instead of:

```rust
stream.stream_in_current_span().boxed()
```

Use:

```rust
stream.boxed_stream_in_current_span()
```

## Regression Tests

The integration test suite includes orphan-span detection tests in `rust/lance/tests/query/tracing.rs`.
These tests use a `SpanTreeRecorder` tracing layer to record all spans and assert that every span is reachable from the test root (no orphans).

Covered query paths:

- Multi-fragment scan
- FTS with filter
- Vector search (IVF-PQ index)
- Hybrid FTS + vector reranker (`fts_rerank`)
- All of the above on local filesystem (exercises `spawn_blocking` in `local.rs`)

Run with:

```bash
cargo test -p lance --features slow_tests --test integration_tests query::tracing::
```

## Code Review Policy

Tracing context rules are enforced through code review (see `.github/copilot-instructions.md`).
When reviewing PRs that touch async execution paths, check that new code follows the patterns described above.

## How to Validate End-to-End

For tracing regressions beyond the in-process tests, validate with a real request and inspect the full Tempo time window around that request.
Do not only search for a single span name.
Success means the window contains one request root trace and `lance-rust` spans remain underneath it instead of appearing as a separate root trace.
