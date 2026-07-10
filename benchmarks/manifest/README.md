# Manifest benchmark report

`report.py` is a standard-library-only gate reporter for the manifest codec and end-to-end benchmark JSONL outputs. It merges one or more input files, validates the complete benchmark matrix, pairs protobuf and Lance samples by round, computes medians, and writes a deterministic English Markdown report.

## Usage

From the repository root:

```sh
EXPECTED_COMMIT=0123456789abcdef0123456789abcdef01234567
python3 benchmarks/manifest/report.py \
  --expected-commit "$EXPECTED_COMMIT" \
  /absolute/path/codec.jsonl \
  /absolute/path/e2e-ebs.jsonl \
  /absolute/path/e2e-s3.jsonl \
  --explanations /absolute/path/gate-explanations.json \
  --output /absolute/path/manifest-report.md
```

Exit status `0` means `PASS`, `1` means a complete matrix has a gate `FAIL`, and `2` means `INCOMPLETE`. `INCOMPLETE` takes precedence over `FAIL` because a malformed or unpaired matrix cannot establish the release gate.

`--explanations` is an optional, regenerable JSON sidecar for mechanism and fix
text attached to individual `FAIL` rows. Explanations never change a gate result.
Every unexplained failure is marked `UNEXPLAINED`; malformed, duplicate, stale,
unknown, or non-failing selectors are rejected instead of being silently ignored.
Selectors include the complete run scope so an explanation cannot carry over to
a different revision, seed, or host.

```json
{
  "schema_version": 2,
  "explanations": [
    {
      "selector": {
        "gate": "case",
        "schema_version": 2,
        "suite": "codec",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "seed": 8675309,
        "host": "benchmark-host",
        "scenario": "S1",
        "fragments": 1000,
        "storage": "memory",
        "operation": "encode",
        "metric": "wall_ns"
      },
      "mechanism": "Encoder setup dominates this small manifest.",
      "fix": "Reuse initialized encoder state before the measured path."
    }
  ]
}
```

Case selectors use `gate: "case"` plus the exact case dimensions. Their stable
metric names are `wall_ns`, `bytes`, `peak_rss_bytes`, `status`, `get_requests`,
and `request_counts`; the last name selects the combined conflict-retry GET/PUT
gate on either EBS or S3. Codec scaling selectors use `gate: "scaling"`, omit
`fragments` and `storage`, add `format`, and use the scaling row's `wall_ns` or
`bytes` metric. All selector and explanation fields shown for the corresponding
gate kind are required, and extra fields are rejected.

Run the tests with:

```sh
python3 -m unittest discover -s benchmarks/manifest -p 'test_*.py'
```

Gate runs must start from a clean Git checkout at an explicit full SHA. On a
remote benchmark host, fetch and detach at that SHA, verify it, and invoke only
the repository scripts; do not copy a locally built harness or pass an external
revision label:

```sh
BENCHMARK_COMMIT=0123456789abcdef0123456789abcdef01234567
git fetch origin "$BENCHMARK_COMMIT"
git checkout --detach "$BENCHMARK_COMMIT"
test "$(git rev-parse --verify 'HEAD^{commit}')" = "$BENCHMARK_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Both runners reject tracked or untracked worktree changes. They build their own
`release-with-debug` target, verify that `HEAD` is unchanged after the build,
and record that exact SHA. Benchmark JSONL outputs must stay outside the
worktree so the next runner sees the same clean checkout.

Run the process-isolated end-to-end harness on a local EBS mount with:

```sh
python3 benchmarks/manifest/run_e2e.py \
  --dataset-prefix /mnt/lance-manifest-bench \
  --storage ebs \
  --output /absolute/path/e2e-ebs.jsonl
```

The runner builds the harness with `release-with-debug`, verifies the clean Git
HEAD before and after the build, and passes that SHA to the harness. Callers
cannot override it with `--commit`. Use an `s3://` dataset prefix and
`--storage s3` for S3. The target requires the `metrics` feature so S3 GET and
PUT requests can be separated from HEAD and LIST operations. Every timed
operation runs in a fresh worker process; setup, workload generation, and
deterministic normalization of S2's missing row counts happen outside the timed
region. The unique run prefix is deleted by default, including after a failed
run; pass `--keep-data` only for diagnosis.

The protobuf baseline uses data storage version 2.2. The columnar case uses version 2.3, which requires the columnar manifest container. Setup verifies both the persisted storage version and manifest footer. Time travel stays within two versions of the same format; the harness does not create or test mixed-format history.

## Codec benchmark

`run_codec.py` builds the Rust harness with the repository's
`release-with-debug` profile, then launches separate encode, decode-wall, and
decode-RSS worker processes for every scenario, size, format, and round. The
encode worker passes only an encoded temporary fixture to the other fresh
workers, so decode cannot inherit source construction or encode high-water
memory. The encode and decode-wall workers first run an unmeasured 16-fragment
warm-up in the requested format so one-time runtime, encoder, and decoder
initialization is not mislabeled as steady-state codec wall time. The dedicated
decode-RSS worker never warms up; it loads the main fixture into a memory store,
captures its baseline, and then samples only the full decode.
The gate run uses the fixed S1/S2 workloads, all four required sizes, and five
paired rounds by default:

```sh
python3 benchmarks/manifest/run_codec.py \
  --output /absolute/path/codec.jsonl
```

The S2 source workload contains a deterministic 1% of fragments without
`physical_rows`. The harness resolves those synthetic row counts before timing
and feeds the identical normalized snapshot to both codecs; each JSON record
describes that normalization in an extra `normalization` object. Real data-file
I/O normalization belongs to the end-to-end suite.

For codec records, `peak_rss_bytes` is the RSS increase from immediately before
the measured operation to the highest sampled RSS during that operation. The
extra `rss` object records `baseline_bytes` and absolute `process_peak_bytes`,
so the increment can be audited. The RSS worker has no format-specific warm-up
whose retained allocator or decoder buffers could be subtracted from the Lance
result; its baseline contains only process state and the preloaded encoded input.
Use `--cold` only to produce a separate decode-wall diagnostic; the RSS worker
is cold in both modes. Do not merge cold and warm JSONL files into one report
because they intentionally have the same gate dimensions and round identifiers.

Format selection does not use an environment variable. The protobuf case uses
storage version 2.2 and the Lance case uses storage version 2.3, matching the
storage-version format contract. Both workers validate the footer before
emitting records.

For a non-gating 1K smoke run (one round for S1/S2 and both formats):

```sh
python3 benchmarks/manifest/run_codec.py \
  --smoke \
  --output /absolute/path/codec-smoke.jsonl
```

The Lance worker checks the written footer before recording any metric. If the
writer falls back to protobuf, the worker fails and the runner retains the
already completed records in `OUTPUT.partial`; it never labels fallback output
as Lance. A complete gate run emits 320 JSONL records: four operations for each
of 80 process-isolated samples.

Both runners require and record the full lowercase Git HEAD from a clean
worktree. They reject dirty trees, caller-supplied commit labels, and prebuilt
codec executables, so every gate record is tied to the repository target that
the script built. The report independently requires `--expected-commit` and
rejects any record whose commit differs.

## JSONL schema

Every line must be a JSON object with all fields below. Integer metrics are non-negative. A case is identified by every non-format, non-round dimension, including `commit`, `seed`, and `host`.

| Field | Type | Values or meaning |
|---|---|---|
| `schema_version` | integer | `2`; schema 1 predates the single-copy manifest writer and is rejected |
| `suite` | string | `codec` or `e2e` |
| `scenario` | string | `S1` or `S2` |
| `fragments` | integer | `1000`, `100000`, `1000000`, or `10000000` in the required matrix |
| `format` | string | `protobuf` or `lance` |
| `storage` | string | `memory` for codec; `ebs` or `s3` for end-to-end |
| `operation` | string | Codec: `encode`, `decode`, `size`, `decode_rss`; end-to-end: `open`, `commit`, `conflict_retry`, `time_travel` |
| `round` | integer | Paired sample identifier |
| `wall_ns` | integer | Wall-clock duration in nanoseconds |
| `bytes` | integer | Codec: encoded manifest size. End-to-end: total tracked operation I/O (`read_bytes + write_bytes`) |
| `peak_rss_bytes` | integer | Peak resident memory; codec records the operation-local increment described above |
| `get_requests` | integer | S3: logical GET requests. EBS: tracked read I/O operations, including metadata/list operations |
| `put_requests` | integer | S3: logical PUT or conditional-PUT requests. EBS: tracked write I/O operations, including delete/copy/rename |
| `read_bytes` | integer | Attempted bytes read |
| `write_bytes` | integer | Attempted bytes written, including failed conflict attempts |
| `status` | string | `pass`, `passed`, `ok`, `success`, `fail`, `failed`, or `error` |
| `error` | string or null | Error detail; a non-empty value fails the case |
| `commit` | string | Full lowercase 40- or 64-character Git SHA; must match the report's `--expected-commit` |
| `seed` | integer | Workload seed |
| `host` | non-empty string | Benchmark host identity |

Each required case needs both formats, at least five unique rounds per format, exactly matching round sets, and no duplicate format/round pair. Missing fields, unsupported records, missing matrix cases, or unpaired rounds are explicitly `INCOMPLETE` rather than `PASS`.

## Required matrices and gates

The codec matrix contains S1 and S2 at 1K, 100K, 1M, and 10M fragments for all four codec operations. The end-to-end matrix contains `open` and `time_travel` at all four sizes, plus `commit` and `conflict_retry` at 1M and 10M, for both scenarios and both storage types.

Codec gates compare Lance/protobuf medians: encode wall time at most `1.2x`, decode wall time at most `1.0x`, manifest bytes at most `1.0x`, and decode peak RSS at most `1.0x`. Encode time, decode time, and manifest size growth from 1K to 10M must stay within a symmetric `1.5x` deviation from linear growth.

End-to-end gates require 1K open p50 at most `1.1x` with no GET increase; 1M and 10M open improvements must remain consistent with the codec size ratio, with at least `1.8x` S3 speedup; commit wall time must be at most `1.1x`; conflict retry request behavior must match; and every time-travel sample must succeed. The report also lists the S2/S1 codec cost and conflict write amplification without adding extra thresholds.

`conflict_retry` starts two writers from the same base version and holds both at a barrier immediately before their first commit-handler call. They race the same conditional create: one succeeds, one records an actual conflict, and only the loser retries. `put_requests` includes the failed request. `write_bytes` comes from `IOTracker`, which records each payload before forwarding `put_opts`, so it also includes the failed conditional PUT bytes.

The 100M section is never measured or time-extrapolated. It labels only `10 x` linear estimates derived from measured 10M manifest bytes and commit `write_bytes` medians.
