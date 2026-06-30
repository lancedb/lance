Lance publishes metrics through the [`metrics`](https://docs.rs/metrics) crate
facade. Install any recorder (Prometheus, OpenTelemetry, etc.) in your
application and Lance will emit into it; when no recorder is installed, emission
is a cheap no-op. Metrics are only emitted when Lance is built with the
`metrics` feature.

## Object store metrics

These track I/O against the underlying object store. The `scheme` label is the
store scheme (`s3`, `gs`, `azure`, `file`, …), and `operation` is one of
`get`, `put`, `head`, `list`, `delete`, `copy`, or `rename`.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_object_store_requests_total` | counter | `operation`, `scheme` | Object store requests issued. |
| `lance_object_store_request_bytes_total` | counter | `operation`, `scheme` | Bytes transferred by `get`/`put` requests. |
| `lance_object_store_request_duration_seconds` | histogram | `operation`, `scheme` | Per-request latency, in seconds. |
| `lance_object_store_errors_total` | counter | `operation`, `scheme` | Requests that returned an error. |
| `lance_object_store_throttle_total` | counter | `status`, `scheme` | Throttle responses (HTTP 429 / 5xx) seen at the HTTP layer, counted per attempt including retries. The `status` label is the numeric HTTP status (e.g. `429`, `503`). |

`lance_object_store_throttle_total` is recorded only for the native cloud
stores (S3, GCS, Azure); Opendal-backed stores bypass the HTTP client where the
counter is installed, so they report the other object store metrics but not
throttle counts.
