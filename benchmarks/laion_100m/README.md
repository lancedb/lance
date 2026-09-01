# LAION 100M shared coarse-quantizer benchmark

This benchmark compares two six-segment IVF_RQ indices over the same LAION 100M
dataset. Both sides use identical fragment groups, IVF centroids, and RaBitQ
rotation. The only A/B variable is whether the index metadata enables shared
coarse-quantizer partition ranking.

The default dataset is:

```text
tos://test-lance/mj-lance/LAION_100M_lance_2.1.lance
```

The benchmark assumes 100,000,000 rows, an `id` column, and a 768-dimensional
fixed-size-list `emb` column. The default index parameters are:

| Parameter | Default | Notes |
| --- | ---: | --- |
| Index type | `IVF_RQ` | L2 distance |
| RQ bits | 5 | Shared RaBitQ rotation for both branches |
| Target partition size | 4,096 rows | Used to derive the IVF size |
| IVF partitions/centroids | 24,414 | `100,000,000 // 4,096` |
| Segment target | 16,777,216 rows | `4,096 ** 2` |
| Expected index segments | 6 | Each segment covers a disjoint fragment group |

The baseline branch ranks the IVF centroids independently for each index
segment. The optimized branch records a coarse-quantizer fingerprint and ranks
the identical centroids once per query before reusing the ranking across all six
segments.

## Prerequisites

Run all Python commands through the Lance repository environment:

```bash
cd python
make install
make build
```

`make install` is required once in a fresh checkout. Run `make build` after any
Rust changes so the benchmark uses the current local PyLance extension.

Set TOS credentials without putting their values in shell history, source files,
or result directories:

```bash
export TOS_ACCESS_KEY_ID=...
export TOS_SECRET_ACCESS_KEY=...
export TOS_ENDPOINT=...
export TOS_REGION=...
```

The compatibility variables `EMR_QA_AK` and base64-encoded `EMR_QA_SK` are also
supported. Set `EMR_QA_SK_BASE64=false` only when `EMR_QA_SK` contains the raw
secret instead. Temporary security tokens can be supplied with
`TOS_SECURITY_TOKEN`. Credential values are never written to benchmark output.

### Host and temporary storage

For the full 100M build, start with a dedicated 32-vCPU, 256-GiB host and
100--200 GiB of free SSD/NVMe temporary storage. Query-only comparison typically
needs 128--256 GiB of RAM and little local disk, but it should use the same fixed
CPU allocation for both branches.

IVF construction performs a local disk-backed shuffle even though the source
dataset and final index live in TOS. Point `TMPDIR` at the mounted fast disk
before building:

```bash
mkdir -p /data00/lance-tmp
export TMPDIR=/data00/lance-tmp
```

Keep the OS image, PyLance build, CPU allocation, TOS endpoint, and network path
unchanged throughout an A/B run.

## Workflow

Run the following stages in order. Branch creation is a one-time prerequisite;
index construction uses one common flow with only the shared-quantizer toggle
changed; recall calibration selects the useful nprobes region; comparison then
measures both branches around that region.

### 1. Create branches first

Branch creation is intentionally not part of `build_index.py`. Create the two
branches once from the same unindexed data version before starting either build:

```bash
cd python
uv run python ../benchmarks/laion_100m/prepare_branches.py \
  --source-branch main
```

The source defaults to the latest `main` version. Pass `--source-version` to pin
an explicit immutable starting version. The command refuses to reuse branch names
or branch from an already indexed source unless explicitly allowed.

Use `--dataset-uri` on every command when testing a dataset other than the
default URI. Use `--baseline-branch` and `--optimized-branch` to override the
default branch names.

### 2. Build both indices with one flow

The first invocation trains and checkpoints the global IVF and RQ artifacts.
The second invocation loads exactly the same artifacts.

```bash
cd python

uv run python ../benchmarks/laion_100m/build_index.py \
  --branch ivf-rq-reuse-off \
  --shared-coarse-quantizer false \
  --model-dir /data00/laion-100m-model \
  --checkpoint-dir /data00/laion-100m-checkpoints

uv run python ../benchmarks/laion_100m/build_index.py \
  --branch ivf-rq-reuse-on \
  --shared-coarse-quantizer true \
  --model-dir /data00/laion-100m-model \
  --checkpoint-dir /data00/laion-100m-checkpoints
```

Each build creates six uncommitted segments and commits them together only after
validating complete, non-overlapping fragment coverage. Existing branches and
indices are never deleted or overwritten.

The model directory contains the shared IVF centroid array, the RaBitQ model,
their parameters, and SHA-256 checksums. The checkpoint directory records each
uncommitted segment after it finishes, so an interrupted build can resume by
running the same command with the same arguments. Do not edit or share a model
directory between datasets or parameter sets; parameter and checksum validation
will reject mismatches.

Each invocation writes build measurements to
`<checkpoint-dir>/<branch>/build_metrics.json`. Use `--metrics-file` to select a
different path. Segment measurements are persisted as each segment finishes, so
a resumed build retains the completed work and records how many segments were
loaded from its checkpoint.

| Field | Interpretation |
| --- | --- |
| `model_source` | Whether the latest invocation trained or loaded the shared model |
| `model_prepare_seconds` | Cumulative model training/loading time across invocations |
| `segment_build_seconds` | Wall time for each segment; `null` marks a segment imported from an older checkpoint without timing data |
| `segment_rows` | Physical rows covered by each segment |
| `segments_total_seconds` | Sum of measured segment build times |
| `commit_seconds` | Time to commit all index segments to the branch |
| `index_build_seconds` | Segment plus commit time, excluding shared-model preparation |
| `total_wall_seconds` | Cumulative end-to-end wall time across recorded invocations |
| `process_cpu_seconds` | Cumulative process user plus system CPU time |
| `peak_rss_gib` | Maximum sampled resident memory across invocations |
| `resumed_segments` | Segments loaded from the checkpoint by the latest invocation |
| `runs` | Per-invocation status, wall/CPU/RSS, resumed count, and new segment count |

Compare `index_build_seconds`, not raw end-to-end time, between the two branches:
the first invocation trains the common model while the second normally loads it.
`total_wall_seconds` remains useful for operational cost and interrupted-build
analysis.

### 3. Calibrate recall

Stage `test.parquet` and `neighbors.parquet` locally, or convert them to small
Lance datasets accessible through TOS. The expected fields are `id, emb` and
`id, neighbors_id` respectively. Query IDs are joined to ground truth IDs; input
row order does not have to match. `neighbors_id` must contain at least the
largest requested `k` ground-truth neighbors per query.

```bash
cd python
uv run python ../benchmarks/laion_100m/benchmark.py calibrate \
  --branch ivf-rq-reuse-on \
  --queries /data00/laion/test.parquet \
  --ground-truth /data00/laion/neighbors.parquet \
  --output-dir /data00/laion-results/calibration \
  --nprobes 128 256 512 1024 2048
```

The command writes the selected stable nprobes to `calibration.json`. If the
curve has not reached a plateau, rerun with an additional `4096` point.

A point is considered stable when its recall is within 0.002 of the largest
tested nprobes and the gain at the next tested point is at most 0.001. Selection
must succeed independently for every requested `k`; the final selected value is
the largest stable value across them.

### 4. Compare latency and throughput

For a stable point of 512, benchmark the adjacent powers of two:

```bash
cd python
uv run python ../benchmarks/laion_100m/benchmark.py compare \
  --baseline-branch ivf-rq-reuse-off \
  --optimized-branch ivf-rq-reuse-on \
  --queries /data00/laion/test.parquet \
  --ground-truth /data00/laion/neighbors.parquet \
  --output-dir /data00/laion-results/compare \
  --nprobes 256 512 1024 \
  --concurrency 1 8 32 \
  --duration-seconds 60 \
  --repeats 3
```

Results include Recall@10/100, wall-clock QPS, latency percentiles, error rate,
CPU use, peak RSS, and representative Lance `analyze_plan` output for both branches.
Before timing, the comparison verifies row/schema parity, six index segments,
the expected feature flag on each side, and identical result sets for ten sample
queries. `comparison.csv` and `comparison.json` report median A/B values, QPS gain,
latency reduction, and recall delta for every `(k, nprobes, concurrency)` point.

The load generator is closed-loop: each worker issues its next query after the
previous query completes. Every timed A/B run gets its own warm-up, and A/B order
alternates on each repeat to reduce ordering bias. These are warm-run numbers;
the tool does not flush the OS page cache or remote object-store caches.

The example above performs:

```text
2 k values * 3 nprobes values * 3 concurrencies * 3 repeats * 2 modes
= 108 timed runs
```

At 60 seconds per timed run, that is at least 108 minutes, plus warm-up,
preflight, and object-store variance. Use a unique output directory for every
experiment because `runs.jsonl` is append-only.

## Output files

Calibration writes:

| File | Contents |
| --- | --- |
| `metadata.json` | Command arguments, timestamp, platform, CPU count, and Lance version |
| `runs.jsonl` | One append-only record per measured point |
| `summary.csv` | Per-`(k, nprobes)` recall, QPS, and latency summary |
| `calibration.json` | Stable nprobes per `k` and the selected value |

Comparison additionally writes:

| File | Contents |
| --- | --- |
| `analyze_plan_off.txt/json` | Baseline physical plan and parsed counters |
| `analyze_plan_on.txt/json` | Optimized physical plan and parsed counters |
| `summary.csv` | Every individual repeat for both modes |
| `comparison.csv/json` | Median A/B metrics and calculated improvement |

The main comparison fields are:

| Field | Interpretation |
| --- | --- |
| `qps_gain_percent` | Positive means the optimized branch has higher throughput |
| `latency_*_reduction_percent` | Positive means the optimized branch has lower latency |
| `recall_delta` | Optimized recall minus baseline recall; it should remain near zero |
| `average_cpu_cores` | Process CPU seconds divided by wall seconds |
| `rss_peak_gib` | Peak resident memory of the benchmark process |
| `error_rate` | Failed requests divided by all requests |

Recall@k is the size of the ID intersection between the returned and ground-truth
top-k sets, divided by `k`.

## Expected preflight checks

Before comparison timing begins, the tool requires:

- both branches to have identical row counts and schemas;
- exactly six index segments on each branch;
- no shared coarse-quantizer fingerprint on the baseline;
- a shared coarse-quantizer fingerprint on the optimized branch;
- identical result ID sets for the preflight queries;
- six centroid-ranking calls for the baseline plan and one for the optimized
  plan, with five reused segments.

A preflight failure should be fixed instead of bypassed; otherwise the two sides
are not a controlled A/B comparison.

## Troubleshooting

- **TOS configuration is missing:** verify the access key, secret key, endpoint,
  region, and optional security token environment variables in the same shell
  that runs `uv`.
- **The branch already exists:** choose new branch names or intentionally clean up
  the old experiment outside these scripts. The preparation command does not
  overwrite branches.
- **The index already exists:** use fresh branches. The build command does not
  replace an existing index.
- **The build runs out of local disk:** check `TMPDIR` and free space on its
  filesystem; the shuffle does not use the result directory.
- **Checkpoint mismatch:** rerun with the original arguments or use a fresh
  checkpoint directory. Do not modify checkpoint JSON manually.
- **Recall does not plateau:** add a larger nprobes point, normally 4,096, and
  recalibrate before choosing the final three-point comparison window.
- **Results are noisy:** use a dedicated host, verify that no two benchmark
  processes overlap, retain three or more repeats, and inspect CPU saturation and
  TOS/network behavior alongside the median comparison.

## Tests

After setting up the repository Python environment, run the lightweight unit and
local integration coverage with:

```bash
cd python
uv run pytest ../benchmarks/laion_100m/test_benchmark.py
```
