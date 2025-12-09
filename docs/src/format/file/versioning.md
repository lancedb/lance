# Versioning

The Lance file format has a single version number for both the overall file format and the encoding strategy. The
major number is changed when the file format itself is modified while the minor number is changed when only the encoding
strategy is modified. Newer versions will typically have better performance and compression but may not be readable
by older versions of Lance.

In addition, the latest version of the file format (next) is unstable and should not be used for production use cases.
Breaking changes could be made to unstable encodings and that would mean that files written with these encodings are
no longer readable by any newer versions of Lance. The `next` version should only be used for experimentation and
benchmarking upcoming features.

The following values are supported:

| Version        | Minimal Lance Version | Maximum Lance Version | Description                                                                                                                                  |
| -------------- | --------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1            | Any                   | 0.34 (write)          | This is the initial Lance format. It is no longer writable.                                                                                  |
| 2.0            | 0.16.0                | Any                   | Rework of the Lance file format that removed row groups and introduced null support for lists, fixed size lists, and primitives              |
| 2.1 (unstable) | None                  | Any                   | Enhances integer and string compression, adds support for nulls in struct fields, and improves random access performance with nested fields. |
| legacy         | N/A                   | N/A                   | Alias for 0.1                                                                                                                                |
| stable         | N/A                   | N/A                   | Alias for the latest stable version (currently 2.0)                                                                                          |
| next           | N/A                   | N/A                   | Alias for the latest unstable version (currently 2.1)                                                                                        |
