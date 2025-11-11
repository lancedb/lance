# Lance Table Format

## Overview

The Lance table format organizes datasets as versioned collections of fragments and indices.
Each version is described by an immutable manifest file that references data files, deletion files, transaction file and indices.
The format supports ACID transactions, schema evolution, and efficient incremental updates through Multi-Version Concurrency Control (MVCC).

## Manifest

![Overview](../../images/table_overview.png)

A manifest describes a single version of the dataset.
It contains the complete schema definition including nested fields, the list of data fragments comprising this version, 
a monotonically increasing version number, and an optional reference to the index section that describes a list of index metadata.

<details>
<summary>Manifest protobuf message</summary>

```protobuf
%%% proto.message.Manifest %%%
```

</details>

## Schema & Fields

The schema of the table is written as a series of fields, plus a schema metadata map. 
The data types generally have a 1-1 correspondence with the Apache Arrow data types.
Each field, including nested fields, have a unique integer id. At initial table creation time, fields are assigned ids in depth-first order.
Afterwards, field IDs are assigned incrementally for newly added fields.

Column encoding configurations are specified through field metadata using the `lance-encoding:` prefix.
See [File Format Encoding Specification](../file/encoding.md) for details on available encodings, compression schemes, and configuration options.

<details>
<summary>Field protobuf message</summary>

```protobuf
%%% proto.message.lance.file.Field %%%
```

</details>

## Fragments

![Fragment Structure](../../images/fragment_structure.png)

A fragment represents a horizontal partition of the dataset containing a subset of rows.
Each fragment has a unique `uint32` identifier assigned incrementally based on the dataset's maximum fragment ID.
Each fragment consists of one or more data files storing columns, plus an optional deletion file.
If present, the deletion file stores the positions (0-based) of the rows that have been deleted from the fragment.
The fragment tracks the total row count including deleted rows in its physical rows field.
Column subsets can be read without accessing all data files, and each data file is independently compressed and encoded.

<details>
<summary>DataFragment protobuf message</summary>

```protobuf
%%% proto.message.DataFragment %%%
```

</details>

### Data Evolution

This fragment design enables a new concept called data evolution, which means efficient schema evolution (add column, update column, drop column) with backfill.
For example, when adding a new column, new column data are added by appending new data files to each fragment, with values computed for all existing rows in the fragment.
There is no need to rewrite the entire table to just add data for a single column.
This enables efficient feature engineering and embedding updates for ML/AI workloads.

Each data file should contain a distinct set of field ids. 
It is not required that all field ids in the dataset schema are found in one of the data files. 
If there is no corresponding data file, that column should be read as entirely `NULL`.

Field ids might be replaced with `-2`, a tombstone value. 
In this case that column should be ignored. This used, for example, when rewriting a column: 
The old data file replaces the field id with `-2` to ignore the old data, and a new data file is appended to the fragment.

## Data Files

Data files store column data for a fragment using the Lance file format.
Each data file stores a subset of the columns in the fragment.
Field IDs are assigned either sequentially based on schema position (for Lance file format v1) 
or independently of column indices due to variable encoding widths (for Lance file format v2).

<details>
<summary>DataFile protobuf message</summary>

```protobuf
%%% proto.message.DataFile %%%
```

</details>

## Deletion Files

Deletion files (a.k.a. deletion vectors) track deleted rows without rewriting data files.
Each fragment can have at most one deletion file per version.

Deletion files support two storage formats.
The Arrow IPC format (`.arrow` extension) stores a flat Int32Array of deleted row offsets and is efficient for sparse deletions.
The Roaring Bitmap format (`.bin` extension) stores a compressed roaring bitmap and is efficient for dense deletions.
Readers must filter rows whose offsets appear in the deletion file for the fragment.

Deletions can be materialized by rewriting data files with deleted rows removed.
However, this invalidates row addresses and requires rebuilding indices, which can be expensive.

<details>
<summary>DeletionFile protobuf message</summary>

```protobuf
%%% proto.message.DeletionFile %%%
```

</details>

## Related Specifications

### Storage Layout

File organization, base path system, and multi-location storage.

See [Storage Layout Specification](layout.md)

### Transactions

MVCC, commit protocol, transaction types, and conflict resolution.

See [Transaction Specification](transaction.md)

### Row Lineage

Row address, Stable row ID, row version tracking, and change data feed.

See [Row ID & Lineage Specification](row_id_lineage.md)

### Indices

Vector indices, scalar indices, full-text search, and index management.

See [Index Specification](index/index.md)

### Versioning

Feature flags and format version compatibility.

See [Format Versioning Specification](versioning.md)
