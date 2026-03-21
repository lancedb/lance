# Distributed Indexing

!!! warning
    Lance exposes public APIs that can be integrated into an external
    distributed index build workflow, but Lance itself does not provide a full
    distributed scheduler or end-to-end orchestration layer.

    This page describes the current model, terminology, and execution flow so
    that callers can integrate these APIs correctly.

## Overview

Distributed index build in Lance follows the same high-level pattern as distributed
write:

1. multiple workers build index data in parallel
2. the caller invokes Lance segment build APIs for one distributed build
3. Lance discovers the relevant worker outputs, then plans and builds index artifacts
4. the built artifacts are committed into the dataset manifest

For vector indices, the worker outputs are temporary shard directories under a
shared UUID. Internally, Lance can turn these shard outputs into one or more
built physical segments.

![Distributed Vector Segment Build](../images/distributed_vector_segment_build.svg)

## Terminology

This guide uses the following terms consistently:

- **Staging root**: the shared UUID directory used during distributed index build
- **Partial shard**: one worker output written under the staging root as
  `partial_<uuid>/`
- **Built segment**: one physical index segment produced during segment build and
  ready to be committed into the manifest
- **Logical index**: the user-visible index identified by name; a logical index
  may contain one or more built segments

For example, a distributed vector build may create a layout like:

```text
indices/<staging_uuid>/
├── partial_<shard_0>/
│   ├── index.idx
│   └── auxiliary.idx
├── partial_<shard_1>/
│   ├── index.idx
│   └── auxiliary.idx
└── partial_<shard_2>/
    ├── index.idx
    └── auxiliary.idx
```

After segment build, Lance produces one or more segment directories:

```text
indices/<segment_uuid_0>/
├── index.idx
└── auxiliary.idx

indices/<segment_uuid_1>/
├── index.idx
└── auxiliary.idx
```

These physical segments are then committed together as one logical index.

## Roles

There are two parties involved in distributed indexing:

- **Workers** build partial shards
- **The caller** launches workers, chooses when a distributed build should be
  turned into built segments, provides any additional inputs requested by the
  segment build APIs, and
  commits the final result

Lance does not provide a distributed scheduler. The caller is responsible for
launching workers and driving the overall workflow.

## Current Model

The current model for distributed vector indexing has two layers of parallelism.

### Shard Build

First, multiple workers build partial shards in parallel:

1. on each worker, call an uncommitted shard-build API such as
   `create_index_builder(...).fragments(...).index_uuid(staging_index_uuid).execute_uncommitted()`
   or Python `create_index_uncommitted(..., fragment_ids=..., index_uuid=...)`
2. each worker writes one `partial_<uuid>/` under the shared staging root

### Segment Build

Then the caller turns that staging root into one or more built segments:

1. open the staging root with `create_index_segment_builder(staging_index_uuid)`
2. provide partial index metadata with `with_partial_indices(...)`
3. optionally choose a grouping policy with `with_target_segment_bytes(...)`
4. call `plan()` to get `Vec<IndexSegmentPlan>`

At that point the caller has two execution choices:

- call `build(plan)` for each plan and run those builds on one or more
  distributed segment-build workers

After the segments are built, publish them with
`commit_existing_index_segments(...)`.

## Internal Segmented Finalize Model

Internally, Lance models distributed vector segment build as:

1. **plan** which partial shards should become each built segment
2. **build** each segment from its selected partial shards
3. **commit** the resulting physical segments as one logical index

The plan step is driven by the staging root and any additional shard metadata
required by the segment build APIs.

This is intentionally a storage-level model:

- partial shards are temporary worker outputs
- built segments are durable physical artifacts
- the logical index identity is attached only at commit time

## Segment Grouping

When Lance builds segments from a staging root, it may either:

- keep shard boundaries, so each partial shard becomes one built segment
- group multiple partial shards into a larger built segment

The grouping decision is separate from shard build. Workers only build partial
shards; Lance applies the segment build policy when it plans built segments.

## Responsibility Boundaries

The caller is expected to know:

- which distributed build is ready for segment build
- any additional shard metadata requested by the segment build APIs
- how the resulting built segments should be published

Lance is responsible for:

- writing partial shard artifacts
- discovering partial shards under the staging root
- planning built segments from the discovered shard set
- merging shard storage into built segment artifacts
- committing built segments into the manifest

This split keeps distributed scheduling outside the storage engine while still
letting Lance own the on-disk index format.

## End-to-End Example (Python)

The following recipe uses the public Python APIs to build an `IVF_PQ` index in
three stages:

1. train shared artifacts once
2. build partial shards on workers
3. plan, build, and commit final segments

This example assumes the dataset already exists in object storage and that your
distributed scheduler is responsible for:

- assigning fragment groups to workers
- distributing the shared training artifacts
- collecting the returned partial index metadata on the coordinator

The examples below use an S3 dataset path. See [Object Store
Configuration](object_store.md) for authentication and other storage settings.

### Shared Configuration

```python
import uuid

import lance
from lance.indices import IndicesBuilder
from lance.indices.ivf import IvfModel
from lance.indices.pq import PqModel

uri = "s3://my-bucket/example/distributed-index.lance"
storage_options = {
    "region": "us-east-1",
}
index_name = "vector_idx"
staging_index_uuid = str(uuid.uuid4())
ivf_model_uri = "s3://my-bucket/example/index-training/ivf.lance"
pq_model_uri = "s3://my-bucket/example/index-training/pq.lance"
```

### 1. Train Once and Plan Fragment Groups

One node trains the shared IVF/PQ artifacts and decides how to partition the
dataset fragments across workers. The trained IVF and PQ models can be saved to
object storage and loaded by every worker later, so the scheduler only needs to
distribute the model URIs.

```python
ds = lance.dataset(uri, storage_options=storage_options)

fragments = ds.get_fragments()
fragment_groups = [
    [fragment.fragment_id for fragment in fragments[:2]],
    [fragment.fragment_id for fragment in fragments[2:]],
]

builder = IndicesBuilder(ds, "vector")
ivf_model = builder.train_ivf(
    num_partitions=32,
    distance_type="l2",
)
ivf_model.save(ivf_model_uri)

pq_model = builder.train_pq(ivf_model, num_subvectors=16)
pq_model.save(pq_model_uri)
```

### 2. Build Partial Shards on Each Worker

Each worker receives one fragment group plus the shared training artifacts. The
worker writes one partial shard under the shared staging UUID and returns the
resulting partial index metadata.

```python
ds = lance.dataset(uri, storage_options=storage_options)

fragment_ids = [0, 1]  # assigned by your scheduler for this worker
ivf_model = IvfModel.load(ivf_model_uri)
pq_model = PqModel.load(pq_model_uri)

partial_index = ds.create_index_uncommitted(
    "vector",
    "IVF_PQ",
    name=index_name,
    metric="l2",
    train=True,
    fragment_ids=fragment_ids,
    index_uuid=staging_index_uuid,
    num_partitions=32,
    num_sub_vectors=16,
    ivf_centroids=ivf_model.centroids,
    pq_codebook=pq_model.codebook,
)

# Send partial_index back to the coordinator.
```

### 3. Plan Final Segments on the Coordinator

The coordinator gathers the partial index metadata and decides how partial
shards should be grouped into final physical segments.

```python
ds = lance.dataset(uri, storage_options=storage_options)

# Collected from all workers.
partial_indices = [
    partial_index_0,
    partial_index_1,
]

segment_builder = (
    ds.create_index_segment_builder(staging_index_uuid)
    .with_partial_indices(partial_indices)
    .with_target_segment_bytes(2 * 1024 * 1024 * 1024)  # optional
)

plans = segment_builder.plan()
```

### 4. Build Final Segments on Segment-Build Workers

Segment build can also be distributed. After planning, the coordinator can
assign one `IndexSegmentPlan` to each segment-build worker. Each worker reads
the staged shard outputs directly from object storage and builds one committed
physical segment.

```python
ds = lance.dataset(uri, storage_options=storage_options)

# Assigned by the coordinator.
plan_for_this_worker = plans[0]

segment_builder = ds.create_index_segment_builder(staging_index_uuid)
segment = segment_builder.build(plan_for_this_worker)

# Send segment back to the coordinator.
```

### 5. Commit Final Segments on the Coordinator

After all segment-build workers finish, the coordinator commits the resulting
physical segments as one logical index.

```python
import numpy as np

ds = lance.dataset(uri, storage_options=storage_options)

# Collected from all segment-build workers.
segments = [
    segment_0,
    segment_1,
]

ds = ds.commit_existing_index_segments(
    index_name,
    "vector",
    segments,
)

query = np.random.randn(128).astype(np.float32)
results = ds.to_table(
    nearest={"column": "vector", "q": query, "k": 5, "nprobes": 4}
)
print(results.select(["id", "_distance"]).to_pydict())
```

The same overall flow also applies to `IVF_FLAT` and `IVF_SQ`. In those cases,
train and save only the IVF model, load it on each worker, and pass
`ivf_centroids=ivf_model.centroids` into `create_index_uncommitted(...)`.
