# Observability

Lance can publish operational metrics to your monitoring stack. The table below
is the authoritative catalogue of the metrics Lance emits, shared verbatim with
the Rust [`lance::metrics`](https://docs.rs/lance/latest/lance/metrics/) module
documentation.

--8<-- "rust/lance/src/metrics.md"

## Collecting metrics

Lance emits through the [`metrics`](https://docs.rs/metrics) crate facade, so it
is not tied to a specific backend — you install a recorder/exporter and route
the metrics wherever you like. Metrics are available from **both** the Rust and
Python APIs.

### Rust

Enable the `metrics` feature on the `lance` crate:

```toml
lance = { version = "...", features = ["metrics"] }
```

Then install any `metrics`-compatible recorder once at startup, before opening
datasets. For example, with
[`metrics-exporter-prometheus`](https://docs.rs/metrics-exporter-prometheus):

```rust
metrics_exporter_prometheus::PrometheusBuilder::new()
    .install()
    .expect("install Prometheus recorder");
```

Any recorder works — Prometheus, StatsD, an OpenTelemetry bridge, and so on.
When no recorder is installed, emission is a cheap no-op.

### Python

Unlike Rust, the Python bindings do not let you plug in an arbitrary recorder:
bridging one across the FFI boundary into the Rust `metrics` facade would be
complicated and inefficient. Instead `pylance` standardizes on OpenTelemetry,
which has good Python support, as its recorder.

The `pylance` wheels are built with the `metrics` feature enabled. Install the
OpenTelemetry extra and call `instrument_lance_metrics`, which registers Lance's
metrics as observable instruments on your OpenTelemetry `MeterProvider`:

```bash
pip install "pylance[otel]"
```

```python
from lance.otel import instrument_lance_metrics

# Uses the global MeterProvider; pass meter_provider=... to target a specific one.
instrument_lance_metrics()
```

From there the metrics flow through whatever OpenTelemetry pipeline you have
configured (OTLP, Prometheus, console, …). Because OpenTelemetry has no
asynchronous histogram instrument, histograms are exported Prometheus-style as
three observable counters: `<name>_bucket`, `<name>_count`, and `<name>_sum`.
Each `<name>_bucket` sample carries an `le` ("less than or equal") attribute
giving that bucket's inclusive upper bound in the metric's unit; the bucket
count is cumulative, covering every observation at or below `le`. For example, a
`lance_object_store_request_duration_seconds_bucket` sample with `le="0.5"`
counts all requests that completed in 0.5 seconds or less, while `le="+Inf"` is
the total count.
