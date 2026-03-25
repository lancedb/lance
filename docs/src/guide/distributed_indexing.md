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
3. Lance plans and builds index artifacts from the worker outputs supplied by the caller
4. the built artifacts are committed into the dataset manifest

For vector indices, the worker outputs are segments stored directly
under `indices/<segment_uuid>/`. Lance can turn these outputs into one or more
physical segments and then commit them as one logical index.

![Distributed Vector Segment Build](../images/distributed_vector_segment_build.svg)

## Terminology

This guide uses the following terms consistently:

- **Segment**: one worker output written by `execute_uncommitted()` under
  `indices/<segment_uuid>/`
- **Physical segment**: one index segment that is ready to be committed into
  the manifest
- **Logical index**: the user-visible index identified by name; a logical index
  may contain one or more physical segments

For example, a distributed vector build may create a layout like:

```text
indices/<segment_uuid_0>/
├── index.idx
└── auxiliary.idx

indices/<segment_uuid_1>/
├── index.idx
└── auxiliary.idx

indices/<segment_uuid_2>/
├── index.idx
└── auxiliary.idx
```

After segment build, Lance produces one or more segment directories:

```text
indices/<physical_segment_uuid_0>/
├── index.idx
└── auxiliary.idx

indices/<physical_segment_uuid_1>/
├── index.idx
└── auxiliary.idx
```

These physical segments are then committed together as one logical index. In the
common no-merge case, the input segments are already the physical
segments and `build_all()` returns them unchanged.

## Roles

There are two parties involved in distributed indexing:

- **Workers** build segments
- **The caller** launches workers, chooses how those segments should be turned
  into physical segments, provides any additional inputs requested by the
  segment build APIs, and
  commits the final result

Lance does not provide a distributed scheduler. The caller is responsible for
launching workers and driving the overall workflow.

## Current Model

The current model for distributed vector indexing has two layers of parallelism.

### Worker Build

First, multiple workers build segments in parallel:

1. on each worker, call a shard-build API such as
   `create_index_builder(...).fragments(...).execute_uncommitted()`
   or Python `create_index_uncommitted(..., fragment_ids=...)`
2. each worker writes one segment under `indices/<segment_uuid>/`

### Segment Build

Then the caller turns those existing segments into one or more physical
segments:

1. create a builder with `create_index_segment_builder()`
2. provide segment metadata with `with_segments(...)`
3. optionally choose a grouping policy with `with_target_segment_bytes(...)`
4. call `plan()` to get `Vec<IndexSegmentPlan>`

At that point the caller has two execution choices:

- call `build(plan)` for each plan and run those builds in parallel
- call `build_all()` to let Lance build every planned segment on the current node

After the physical segments are built, publish them with
`commit_existing_index_segments(...)`.

## Internal Segmented Finalize Model

Internally, Lance models distributed vector segment build as:

1. **plan** which input segments should become each physical segment
2. **build** each segment from its selected input segments
3. **commit** the resulting physical segments as one logical index

The plan step is driven by the segment metadata returned from
`execute_uncommitted()` and any additional inputs requested by the segment
build APIs.

This is intentionally a storage-level model:

- segments are worker outputs that are not yet published
- physical segments are durable artifacts referenced by the manifest
- the logical index identity is attached only at commit time

## Segment Grouping

When Lance builds segments from existing inputs, it may either:

- keep segment boundaries, so each input segment becomes one physical segment
- group multiple input segments into a larger physical segment

The grouping decision is separate from worker build. Workers only build
segments; Lance applies the segment build policy when it plans
physical segments.

## Responsibility Boundaries

The caller is expected to know:

- which distributed build is ready for segment build
- the segment metadata returned by worker builds
- how the resulting physical segments should be published

Lance is responsible for:

- writing segment artifacts
- planning physical segments from the supplied segment set
- merging segment storage into physical segment artifacts
- committing physical segments into the manifest

This split keeps distributed scheduling outside the storage engine while still
letting Lance own the on-disk index format.
