# Stable Logical Row Address Benchmark

This directory contains the executable release protocol for storage version 2.3.
Every measured operation runs in a fresh process and emits exactly one strict
JSON object. The runner refuses dirty source trees, records the full Git SHA,
and binds every record to canonical workload-matrix and physical-maintenance
policy hashes. The protocol requires Python 3.10 or newer; set `PYTHON` when
the remote host's default `python3` is older.

## Protocol tracks

- `matrix` covers append, clustered/random delete and update, merge insert,
  row-aligned backfill, N-to-one and repeated compaction, scalar/vector index
  take/optimize, cold open/scan/random take, indexed relocation, and the
  `N-to-one pack -> random update/delete -> cold scan/take` chain. Clustered
  delete reclaim must be admitted by default compaction. Before random 50% and
  90% delete reclaim, v2.3 runs `default_compaction_preflight`: it calls only
  the default compaction planner, never executes or commits its tasks, and must
  leave the dataset version and every object-store write counter unchanged.
  The bounded smoke fixture freezes `must_admit`; the 100M-row release fixture
  freezes `must_not_admit`. Paired explicit `Repack`/baseline compaction then
  reaches the same postcondition independently of that plan-only observation.
- `sustained` updates one fixed, exactly sampled hot set. Only the deterministic
  v2.2 no-stable physical policy may create a common maintenance boundary.
  Placement backpressure is a failure.
- `adversarial_natural` independently samples each update round. Every format
  follows the same policy against its own physical state; v2.3 placement
  maintenance never forces baseline compaction. This is the only adversarial
  net-benefit denominator. Every round records cold probes both before and
  after policy maintenance.
- `adversarial_aligned` pairs v2.3 `NormalizePlacement` with forced baseline
  compaction before retrying the rejected update. It is a state/mechanism
  diagnostic and is not a net-benefit denominator.

The release profile freezes 100M rows, 16-byte/128-byte/vector schemas,
100/10K/100K logical fragments, 1/32/1024/10K takes, the specified mutation
ratios, ten paired repeats, and 100-round repeated chains. The smoke profile
keeps the same protocol with bounded fixtures. Release runs require same-region
S3; EBS is intentionally smoke-only.

```bash
python3 benchmarks/stable_row_address/protocol.py \
  --dataset-root /mnt/bench/stable-row-address \
  --output /mnt/results/stable-row-address.jsonl \
  --profile smoke \
  --track matrix

# Run one fixture-local release shard. Run indices 0 through 8 on independent
# workers; the runner adds the shard suffix to both the S3 and local paths.
python3 benchmarks/stable_row_address/protocol.py \
  --dataset-root s3://same-region-bucket/prefix \
  --storage s3 \
  --output /mnt/results/stable-row-address-release.jsonl \
  --profile release \
  --track matrix \
  --track sustained \
  --track adversarial_natural \
  --track adversarial_aligned \
  --shard-count 9 \
  --shard-index 0

# Resume the exact shard with the same arguments after a host restart.
python3 benchmarks/stable_row_address/protocol.py \
  --dataset-root s3://same-region-bucket/prefix \
  --storage s3 \
  --output /mnt/results/stable-row-address-release.jsonl \
  --profile release \
  --track matrix \
  --track sustained \
  --track adversarial_natural \
  --track adversarial_aligned \
  --shard-count 9 \
  --shard-index 0 \
  --resume

python3 benchmarks/stable_row_address/protocol_report.py \
  /mnt/results/stable-row-address-release.shard-000-of-009.jsonl \
  --markdown /mnt/results/stable-row-address-release.shard-000-of-009.md \
  --json /mnt/results/stable-row-address-release.shard-000-of-009.report.json

python3 benchmarks/stable_row_address/protocol_aggregate.py \
  /mnt/results/stable-row-address-release.shard-*-of-009.jsonl \
  --markdown /mnt/results/stable-row-address-release.aggregate.md \
  --json /mnt/results/stable-row-address-release.aggregate.json

# Install a reboot-persistent serial nine-shard release service. The checkout
# must exactly match EXPECTED_COMMIT; existing shard evidence is resumed.
DATASET_ROOT=s3://same-region-bucket/prefix \
RESULT_ROOT=/mnt/results/stable-row-address-release \
AWS_REGION=us-east-2 \
EXPECTED_COMMIT=0123456789abcdef0123456789abcdef01234567 \
PYTHON=/usr/bin/python3.11 \
benchmarks/stable_row_address/release_remote_systemd.sh install
```

Use repeated `--case`, `--case-filter`, or `--variant` arguments for focused
runs. Canonical unindexed and indexed fixtures are created once per
format/schema/fragment layout and shallow-cloned for every case/repeat, sharing
unchanged data and index objects. The immutable `.fixture_lineage.jsonl`
records every source/target edge; the report verifies it against worker records.
Dataset objects are intentionally preserved. The JSONL, `.protocol.json`,
`.summary.json`, lineage, take-ID, and maintenance-plan artifacts are immutable
evidence for one run.
The sidecar records canonical unique-payload, no-dedup logical-payload, and
minimum full-scan projections before execution; these are recomputed by the
report instead of trusted. The full-scan projection includes the untimed
business-ID-to-row-reference scan required by every cold-take probe.
Nine release shards match the nine canonical data-fixture layouts and therefore
do not duplicate canonical data across workers. A checkpoint is atomically
written before and after every worker result and after every case or repeated
round. A crash after sidecar creation but before the first checkpoint resumes
as an empty shard; later resume skips only validated, fsynced JSONL records. If a crash leaves an
in-flight mutation without a durable result, the runner refuses to guess whether
the commit happened and requires a fresh shard prefix.
The report returns `0` for `PASS`, `1` for `FAIL`, and `2` for `INCOMPLETE`.
It derives the expected grid from the frozen sidecar; absent formats, repeats,
probes, boundaries, trigger follow-ups, or metrics cannot be silently skipped.
For the adversarial natural-policy track it independently re-evaluates every
policy boundary, preserves every round-prefix total in the JSON report, and
reports PMR rounds, natural-maintenance rounds, and terminal physical and
placement debt. Missing post-maintenance probes or policy-triggered work is
`INCOMPLETE`; maintenance without a frozen-policy trigger is `FAIL`.
The remote wrapper writes `stable-row-address-release.execution-complete` only
after every runner exits successfully. It writes
`stable-row-address-release.pass` only after all shard reports and the aggregate
return `PASS`; that marker binds the exact Git commit and aggregate SHA-256.
Every executed relocation references one canonical maintenance plan. The
track-owned artifact freezes one total-order physical source group, derives
row boundaries from measured live bytes and the byte target, and records the
exact output row counts. All three formats validate and replay that group and
those boundaries before the first data write; format-native identity order is
allowed inside each output. The runner hashes the plan and the report
independently recomputes its byte formula and hash.

## Measurement contract

Latency includes bounded synthetic stream generation so large fixtures remain
memory bounded. Logical I/O is measured by one path-aware wrapper propagated to
both dataset-root and shallow-clone base stores; S3 attempts are independently
measured at the native HTTP retry boundary. Exact live logical-index coverage
is validated after latency, I/O, and peak-RSS snapshots with a separate
tracker, so deletion-vector verification cannot improve or pollute a measured
operation. Each
record also contains peak RSS, physical/live bytes, manifest and placement
bytes, admission state, mutation counts, and a format-neutral state digest.
Compaction records additionally expose the post-data-file
layout/index-maintenance duration, compacted/index bytes, remapped rows and
indices, coverage reuse, and group admission counts. Indexed relocation gates
require zero v2.3 index object I/O/remap, 100% coverage reuse, a 10x tail-phase
improvement, and a 2x end-to-end improvement when index bytes cover compacted
data bytes. Vector index queries use the exact vector stored at row id zero, so
recall is an exact first-result top-1 check with no unmeasured full-table
ground-truth scan. Scalar recall compares the exact returned business-key set.
Coverage comes from current manifest coverage, generation, liveness, and
fragment metadata without opening index objects; it is never a declared
constant. Every indexed take must report full effective coverage, meet the
recall floor, and not regress from either v2.2 baseline.

Native `UpdateBuilder` is used for range updates. Exact uniform-without-
replacement repeated updates use a precomputed Floyd sample and a matched-only
merge executor because the public update API accepts only SQL predicates. This
is explicit in `implementation_path=exact_selection_matched_merge`; the runner
never presents it as native-update cost. Delete fixtures use a deterministic
keyed permutation stored in the initial `value` column, so threshold predicates
select an exact random-size set without an auxiliary sidecar.

Cold take setup performs a deterministic bottom-k sample over the immutable
business `id`, records both business ids and format-specific row references,
and rejects the pair before timing if the formats selected different rows. The
setup scan is outside the timed take and its artifact is preserved; the timed
operation remains a fresh-process direct `take_rows` call. Standard operation
latency gates bootstrap the p95 of paired ratios rather than their median.
