# MemWAL Index

The MemWAL Index is a system index that serves as the centralized structure for all MemWAL metadata.
It stores configuration (region specs, indexes to maintain), merge progress, and region state snapshots.

A table has at most one MemWAL index.

For the complete specification, see:

- [MemWAL Index Overview](../../mem_wal.md#memwal-index) - Purpose and high-level description
- [MemWAL Index Details](../../mem_wal.md#memwal-index-details) - Storage format, schemas, and staleness handling
- [MemWAL Index Builder](../../mem_wal.md#memwal-index-builder) - Background process and configuration updates
