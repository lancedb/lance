# Lance Performance Guide

This guide provides tips and tricks for optimizing the performance of your Lance applications.

## Logging

Lance uses the `log` crate to log messages. Displaying these log messages will depend on the client
library you are using. For rust, you will need to configure a logging subscriber. For more details
ses the [log](https://docs.rs/log/latest/log/) docs. The Python and Java clients configure a default
logging subscriber that logs to stderr.

The Python/Java logger can be configured with several environment variables:

- `LANCE_LOG`: Controls log filtering based on log level and target. See the [env_logger](https://docs.rs/env_logger/latest/env_logger/) docs for more details. The `LANCE_LOG` environment variable replaces the `RUST_LOG` environment variable.
- `LANCE_LOG_STYLE`: Controls whether colors are used in the log messages. Valid values are `auto`, `always`, `never`.
- `LANCE_LOG_TS_PRECISION`: The precision of the timestamp in the log messages. Valid values are `ns`, `us`, `ms`, `s`.
- `LANCE_LOG_FILE`: Redirects Rust log messages to the specified file path instead of stderr. When set, Lance will create the file and any necessary parent directories. If the file cannot be created (e.g., due to permission issues), Lance will fall back to logging to stderr.

## Trace Events

Lance uses tracing to log events. If you are running `pylance` then these events will be emitted to
as log messages. For Rust connections you can use the `tracing` crate to capture these events.

### File Audit

File audit events are emitted when significant files are created or deleted.

| Event               | Parameter | Description                                                                |
| ------------------- | --------- | -------------------------------------------------------------------------- |
| `lance::file_audit` | `mode`    | The mode of I/O operation (create, delete, delete_unverified)              |
| `lance::file_audit` | `type`    | The type of file affected (manifest, data file, index file, deletion file) |

### I/O Events

I/O events are emitted when significant I/O operations are performed, particularly
those related to indices. These events are NOT emitted when the index is loaded from
the in-memory cache. Correct cache utilization is important for performance and these
events are intended to help you debug cache usage.

| Event              | Parameter | Description                                                                                          |
| ------------------ | --------- | ---------------------------------------------------------------------------------------------------- |
| `lance::io_events` | `type`    | The type of I/O operation (open_scalar_index, open_vector_index, load_vector_part, load_scalar_part) |

### Execution Events

Execution events are emitted when an execution plan is run. These events are useful for
debugging query performance.

| Event              | Parameter           | Description                                                    |
| ------------------ | ------------------- | -------------------------------------------------------------- |
| `lance::execution` | `type`              | The type of execution event (plan_run is the only type today)  |
| `lance::execution` | `output_rows`       | The number of rows in the output of the plan                   |
| `lance::execution` | `iops`              | The number of I/O operations performed by the plan             |
| `lance::execution` | `bytes_read`        | The number of bytes read by the plan                           |
| `lance::execution` | `indices_loaded`    | The number of indices loaded by the plan                       |
| `lance::execution` | `parts_loaded`      | The number of index partitions loaded by the plan              |
| `lance::execution` | `index_comparisons` | The number of comparisons performed inside the various indices |

## Threading Model

Lance is designed to be thread-safe and performant. Lance APIs can be called concurrently unless
explicitly stated otherwise. Users may create multiple tables and share tables between threads.
Operations may run in parallel on the same table, but some operations may lead to conflicts. For
details see [conflict resolution](../format/index.md#conflict-resolution).

Most Lance operations will use multiple threads to perform work in parallel. There are two thread
pools in lance: the IO thread pool and the compute thread pool. The IO thread pool is used for
reading and writing data from disk. The compute thread pool is used for performing computations
on data. The number of threads in each pool can be configured by the user.

The IO thread pool is used for reading and writing data from disk. The number of threads in the IO
thread pool is determined by the object store that the operation is working with. Local object stores
will use 8 threads by default. Cloud object stores will use 64 threads by default. This is a fairly
conservative default and you may need 128 or 256 threads to saturate network bandwidth on some cloud
providers. The `LANCE_IO_THREADS` environment variable can be used to override the number of IO
threads. If you increase this variable you may also want to increase the `io_buffer_size`.

The compute thread pool is used for performing computations on data. The number of threads in the
compute thread pool is determined by the number of cores on the machine. The number of threads in
the compute thread pool can be overridden by setting the `LANCE_CPU_THREADS` environment variable.
This is commonly done when running multiple Lance processes on the same machine (e.g when working with
tools like Ray). Keep in mind that decoding data is a compute intensive operation, even if a workload
seems I/O bound (like scanning a table) it may still need quite a few compute threads to achieve peak
performance.

## Memory Requirements

Lance is designed to be memory efficient. Operations should stream data from disk and not require
loading the entire dataset into memory. However, there are a few components of Lance that can use
a lot of memory.

### Metadata Cache

Lance uses a metadata cache to speed up operations. This cache holds various pieces of metadata such
as file metadata, dataset manifests, etc. This cache is an LRU cache that is sized by bytes. The default
size is 1 GiB.

The metadata cache is not shared between tables by default. For best performance you should create a
single table and share it across your application. Alternatively, you can create a single session
and specify it when you open tables.

Keys are often a composite of multiple fields and all keys are scoped to the dataset URI. The following items are stored in the metadata cache:

| Item              | Key                                              | What is stored                      |
| ----------------- | ------------------------------------------------ | ----------------------------------- |
| Dataset Manifests | Dataset URI, version, and etag                   | The manifest for the dataset        |
| Transactions      | Dataset URI, version                             | The transaction for the dataset     |
| Deletion Files    | Dataset URI, fragment_id, version, id, file_type | The deletion vector for a frag      |
| Row Id Mask       | Dataset URI, version                             | The row id sequence for the dataset |
| Row Id Index      | Dataset URI, version                             | The row id index for the dataset    |
| Row Id Sequence   | Dataset URI, fragment_id                         | The row id sequence for a fragment  |
| Index Metadata    | Dataset URI, version                             | The index metadata for the dataset  |
| Index Details¹    | Dataset URI, index uuid                          | The index details for an index      |
| File Global Meta  | Dataset URI, file path                           | The global metadata for a file      |
| File Column Meta  | Dataset URI, file path, column index             | The search cache for a column       |

Notes:

1. This is only stored for very old indexes which don't store their details in the manifest.

### Index Cache

Lance uses an index cache to speed up queries. This caches vector and scalar indices in memory. The
max size of this cache can be configured when creating a `LanceDataset` using the `index_cache_size_bytes`
parameter. This cache is an LRU cached that is sized by bytes. The default size is 6 GiB.
You can view the size of this cache by inspecting the result of `dataset.session().size_bytes()`.

The index cache is not shared between tables. For best performance you should create a single table and
share it across your application.

**Note**: `index_cache_size` (specified in entries) was deprecated since version 0.30.0. Use
`index_cache_size_bytes` (specified in bytes) for new code.

### Scanning Data

Searches (e.g. vector search, full text search) do not use a lot of memory to hold data because they don't
typically return a lot of data. However, scanning data can use a lot of memory. Scanning is a streaming
operation but we need enough memory to hold the data that we are scanning. The amount of memory needed is
largely determined by the `io_buffer_size` and the `batch_size` variables.

Each I/O thread should have enough memory to buffer an entire page of data. Pages today are typically between
8 and 32 MB. This means, as a rule of thumb, you should generally have about 32MB of memory per I/O thread.
The default `io_buffer_size` is 2GB which is enough to buffer 64 pages of data. If you increase the number
of I/O threads you should also increase the `io_buffer_size`.

Scans will also decode data (and run any filtering or compute) in parallel on CPU threads. The amount of data
decoded at any one time is determined by the `batch_size` and the size of your rows. Each CPU thread will
need enough memory to hold one batch. Once batches are delivered to your application, they are no longer tracked
by Lance and so if memory is a concern then you should also be careful not to accumulate memory in your own
application (e.g. by running `to_table` or otherwise collecting all batches in memory.)

The default `batch_size` is 8192 rows. When you are working with mostly scalar data you want to keep batches
around 1MB and so the amount of memory needed by the compute threads is fairly small. However, when working with
large data you may need to turn down the `batch_size` to keep memory usage under control. For example, when
working with 1024-dimensional vector embeddings (e.g. 32-bit floats) then 8192 rows would be 32MB of data. If you
spread that across 16 CPU threads then you would need 512MB of compute memory per scan. You might find working
with 1024 rows per batch is more appropriate.

In summary, scans could use up to `(2 * io_buffer_size) + (batch_size * num_compute_threads)` bytes of memory.
Keep in mind that `io_buffer_size` is a soft limit (e.g. we cannot read less than one page at a time right now)
and so it is not necessarily a bug if you see memory usage exceed this limit by a small margin.

The above limits refer to limits per-scan. There is an additional limit on the number of IOPS that is applied
across the entire process. This limit is specified by the `LANCE_PROCESS_IO_THREADS_LIMIT` environment variable.
The default is 128 which is more than enough for most workloads. You can increase this limit if you are working
with a high-throughput workload. You can even disable this limit entirely by setting it to zero. Note that this
can often lead to issues with excessive retries and timeouts from the object store.

## Indexes

Training and searching indexes can have unique requirements for compute and memory. This section provides some
guidance on what can be expected for different index types.

### BTree Index

The BTree index is a two-level structure that provides efficient range queries and sorted access.
It strikes a balance between an expensive memory structure containing all values and an expensive disk
structure that can't be efficiently searched.

Training a BTree index is done by sorting the column. This is done using an [external sort](https://en.wikipedia.org/wiki/External_sorting) to constrain the total memory usage to a reasonable amount. Updating a BTree index does not
require re-sorting the entire column. The new values are sorted and the existing values are merged into the new sorted
values in linear time.

#### Storage Requirements

The BTree index is essentially a sorted copy of a column. The storage requirements are therefore the same as the column
but an additional 4 bytes per value is required to store the row ID and there is a small lookup structure which
should be roughly 0.001% of the size of the column.

#### Memory Requirements

Training a BTree index requires some RAM but the current implementation spills to disk rather aggressively and so the
total memory usage is fairly low.

When searching a BTree index, the index is loaded into the index cache in pages. Each page contains 4096 values.

#### Performance

The sort stage is the most expensive step in training a BTree index. The time complexity is O(n log n) where n is the number of rows in the column. At very large scales this can be a bottleneck and a distributed sort may be necessary. Lance currently does
not have anything builtin for this but work is underway to add this functionality. Training an index in parts as the data grows
may be slightly more efficient than training the entire index at once if you have the flexibility to do so.

When the BTree index is fully loaded into the index cache, the search time scales linearly with the number of rows that match the
query. When the BTree index is not fully loaded into the index cache, the search time will be controlled by the number of pages
that need to be loaded from disk and the speed of storage. The parts_loaded metric in the execution metrics can tell you how many
pages were loaded from disk to satisfy a query.

### Bitmap Index

The Bitmap index is an inverted lookup table that stores a bitmap for each possible value in the column. These bitmaps are compressed and serialized as a [Roaring Bitmap](https://roaringbitmap.org/).

A bitmap index is currently trained by accumulating the column into a hash map from value to a vector of row ids. Each value
is then serialized into a bitmap and stored in a file.

### Storage Requirements

The size of a bitmap index is difficult to calculate precisely but will generally scale with the number of unique values in the
column since a unique bitmap is required for each value and a single bitmap with all rows will compress more efficiently than
many bitmaps with a small number of rows.

#### Memory Requirements

Since training a bitmap index requires collecting the values into a hash map you will need at least 8 bytes of memory per row.
In addition, if you have many unique values, then you will need additional memory for the keys of the hash map. Training large
bitmaps with many unique values at scale can be memory intensive.

When a bitmap index is searched, bitmaps are loaded into the session cache individually. The size of the bitmap will depend on
the number of rows that match the token.

### Performance

When the bitmap index is fully loaded into the index cache, the search time scales linearly with the number of values that the
query requires. This makes the bitmap very fast for equality queries or very small ranges. Queries against large ranges are
currently extremely slow and the btree index is much faster for large range queries.

When a bitmap index is not fully loaded into the index cache, the search time will be controlled by the number of bitmaps that
need to be loaded from disk and the speed of storage. The parts_loaded metric in the execution metrics can tell you how many
bitmaps were loaded from disk to satisfy a query.
