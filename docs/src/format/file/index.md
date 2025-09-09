# Lance File Format

## File Structure

Each `.lance` file is the container for the actual data.

![Format Overview](../../images/format_overview.png)

At the tail of the file, `ColumnMetadata` protobuf blocks are used to describe the encoding of the columns in the file.

```protobuf
%%% proto.message.ColumnMetadata %%%
```

A `Footer` describes the overall layout of the file. The entire file layout is described here:

```protobuf
// Note: the number of buffers (BN) is independent of the number of columns (CN)
//       and pages.
//
//       Buffers often need to be aligned.  64-byte alignment is common when
//       working with SIMD operations.  4096-byte alignment is common when
//       working with direct I/O.  In order to ensure these buffers are aligned
//       writers may need to insert padding before the buffers.
//       
//       If direct I/O is required then most (but not all) fields described
//       below must be sector aligned.  We have marked these fields with an
//       asterisk for clarity.  Readers should assume there will be optional
//       padding inserted before these fields.
//
//       All footer fields are unsigned integers written with  little endian
//       byte order.
//
// ├──────────────────────────────────┤
// | Data Pages                       |
// |   Data Buffer 0*                 |
// |   ...                            |
// |   Data Buffer BN*                |
// ├──────────────────────────────────┤
// | Column Metadatas                 |
// | |A| Column 0 Metadata*           |
// |     Column 1 Metadata*           |
// |     ...                          |
// |     Column CN Metadata*          |
// ├──────────────────────────────────┤
// | Column Metadata Offset Table     |
// | |B| Column 0 Metadata Position*  |
// |     Column 0 Metadata Size       |
// |     ...                          |
// |     Column CN Metadata Position  |
// |     Column CN Metadata Size      |
// ├──────────────────────────────────┤
// | Global Buffers Offset Table      |
// | |C| Global Buffer 0 Position*    |
// |     Global Buffer 0 Size         |
// |     ...                          |
// |     Global Buffer GN Position    |
// |     Global Buffer GN Size        |
// ├──────────────────────────────────┤
// | Footer                           |
// | A u64: Offset to column meta 0   |
// | B u64: Offset to CMO table       |
// | C u64: Offset to GBO table       |
// |   u32: Number of global bufs     |
// |   u32: Number of columns         |
// |   u16: Major version             |
// |   u16: Minor version             |
// |   "LANC"                         |
// ├──────────────────────────────────┤
//
// File Layout-End
```

## File Version

The Lance file format has gone through a number of changes including a breaking change from version 1 to version 2.
There are a number of APIs that allow the file version to be specified.
Using a newer version of the file format will lead to better compression and/or performance.
However, older software versions may not be able to read newer files.

In addition, the latest version of the file format (next) is unstable and should not be used for production use cases.
Breaking changes could be made to unstable encodings and that would mean that files written with these encodings are
no longer readable by any newer versions of Lance. The `next` version should only be used for experimentation and
benchmarking upcoming features.

The following values are supported:

| Version        | Minimal Lance Version | Maximum Lance Version | Description                                                                                                                                  |
| -------------- | --------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1            | Any                   | Any                   | This is the initial Lance format.                                                                                                            |
| 2.0            | 0.16.0                | Any                   | Rework of the Lance file format that removed row groups and introduced null support for lists, fixed size lists, and primitives              |
| 2.1 (unstable) | None                  | Any                   | Enhances integer and string compression, adds support for nulls in struct fields, and improves random access performance with nested fields. |
| legacy         | N/A                   | N/A                   | Alias for 0.1                                                                                                                                |
| stable         | N/A                   | N/A                   | Alias for the latest stable version (currently 2.0)                                                                                          |
| next           | N/A                   | N/A                   | Alias for the latest unstable version (currently 2.1)                                                                                        |

## File Encodings

Lance supports a variety of encodings for different data types.
The encodings are chosen to give both random access and scan performance.
Encodings are added over time and may be extended in the future.
The manifest records a max format version which controls which encodings will be used.
This allows for a gradual migration to a new data format so that old readers can still read new data while a migration is in progress.

Encodings are divided into "field encodings" and "array encodings".
Field encodings are consistent across an entire field of data,
while array encodings are used for individual pages of data within a field.
Array encodings can nest other array encodings (e.g. a dictionary encoding can bitpack the indices)
however array encodings cannot nest field encodings.
For this reason data types such as `Dictionary<UInt8, List<String>>`
are not yet supported (since there is no dictionary field encoding)

### Encodings Available

| Encoding Name   | Encoding Type  | What it does                                                                                                                                | Supported Versions | When it is applied                                                                      |
| --------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------- |
| Basic struct    | Field encoding | Encodes non-nullable struct data                                                                                                            | >= 2.0             | Default encoding for structs                                                            |
| List            | Field encoding | Encodes lists (nullable or non-nullable)                                                                                                    | >= 2.0             | Default encoding for lists                                                              |
| Basic Primitive | Field encoding | Encodes primitive data types using separate validity array                                                                                  | >= 2.0             | Default encoding for primitive data types                                               |
| Value           | Array encoding | Encodes a single vector of fixed-width values                                                                                               | >= 2.0             | Fallback encoding for fixed-width types                                                 |
| Binary          | Array encoding | Encodes a single vector of variable-width data                                                                                              | >= 2.0             | Fallback encoding for variable-width types                                              |
| Dictionary      | Array encoding | Encodes data using a dictionary array and an indices array which is useful for large data types with few unique values                      | >= 2.0             | Used on string pages with fewer than 100 unique elements                                |
| Packed struct   | Array encoding | Encodes a struct with fixed-width fields in a row-major format making random access more efficient                                          | >= 2.0             | Only used on struct types if the field metadata attribute `"packed"` is set to `"true"` |
| Fsst            | Array encoding | Compresses binary data by identifying common substrings (of 8 bytes or less) and encoding them as symbols                                   | >= 2.1             | Used on string pages that are not dictionary encoded                                    |
| Bitpacking      | Array encoding | Encodes a single vector of fixed-width values using bitpacking which is useful for integral types that do not span the full range of values | >= 2.1             | Used on integral types                                                                  |