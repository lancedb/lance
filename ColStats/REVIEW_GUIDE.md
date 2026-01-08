# Column Statistics Feature - File Review Guide

This guide organizes all files by phase for systematic code review. Review files in order, as each phase builds on the previous ones.

---

## 📋 Phase 0: Infrastructure & Refactoring

**Purpose**: Extract shared zone utilities to enable reuse across modules.

### Files to Review:

1. **`rust/lance-core/src/utils/zone.rs`** (NEW - 212 lines)
   - `ZoneBound` struct: Defines zone boundaries (start, length)
   - `ZoneProcessor` trait: Generic interface for processing zones
   - `FileZoneBuilder<P>`: Synchronous zone builder for file-level stats
   - **Key Functions**:
     - `process_chunk()`: Accumulate statistics for a chunk
     - `finish_zone()`: Finalize zone statistics
     - `reset()`: Clear state for next zone

2. **`rust/lance-index/src/scalar/zone_trainer.rs`** (NEW - 876 lines)
   - `ZoneTrainer<P>`: Async zone trainer for index building
   - Handles `_rowaddr` and fragment boundaries
   - Used by zonemap and bloom filter indices
   - **Key Functions**:
     - `process_batch()`: Process data batches
     - `finalize()`: Complete zone training

3. **`rust/lance-index/src/scalar/zoned.rs`** (MODIFIED)
   - Updated to use new zone utilities
   - Re-exports `ZoneBound`, `ZoneProcessor`, `ZoneTrainer`

4. **`rust/lance-core/src/utils.rs`** (MODIFIED)
   - Added `pub mod zone;` declaration

**Review Focus**: 
- ✅ Trait design is generic and reusable
- ✅ Clear separation between sync (FileZoneBuilder) and async (ZoneTrainer)
- ✅ No circular dependencies

---

## 📋 Phase 1: Policy Enforcement

**Purpose**: Enforce dataset-level column statistics policy to ensure consistency.

### Files to Review:

1. **`rust/lance/src/dataset/write.rs`** (MODIFIED - ~111 lines added)
   - **Key Changes**:
     - Added `enable_column_stats: bool` field to `WriteParams`
     - `WriteParams::for_dataset()`: Inherits policy from dataset manifest
     - `WriteParams::validate_column_stats_policy()`: Validates consistency
   - **Lines to Review**: 
     - `WriteParams` struct definition (~line 159)
     - `for_dataset()` method (~line 278)
     - `validate_column_stats_policy()` method (~line 350)

2. **`rust/lance/src/dataset/write/insert.rs`** (MODIFIED - ~185 lines added)
   - **Key Changes**:
     - Sets `lance.column_stats.enabled` in manifest config on dataset creation
     - Only when `WriteMode::Create` and `enable_column_stats=true`
   - **Lines to Review**:
     - `build_transaction()` method (~line 200-250)
     - Look for `config_upsert_values` and `lance.column_stats.enabled`
   - **Tests**: 
     - `test_column_stats_policy_set_on_create` (~line 300+)
     - `test_column_stats_policy_not_set_when_disabled` (~line 350+)

3. **`rust/lance/src/dataset/write/update.rs`** (MODIFIED)
   - **Key Changes**:
     - Removed `enable_column_stats` field (now uses `WriteParams::for_dataset()`)
     - Uses policy inheritance instead of explicit parameter

**Review Focus**:
- ✅ Policy is set correctly on dataset creation
- ✅ Policy inheritance works via `for_dataset()`
- ✅ Validation prevents mixed-stat datasets
- ✅ Error messages are clear and helpful

---

## 📋 Phase 2: Per-Fragment Statistics Writer

**Purpose**: Collect and store column statistics in each data file.

### Files to Review:

1. **`rust/lance-file/src/writer.rs`** (MODIFIED - ~407 lines added)
   - **Key Changes**:
     - `build_column_statistics()`: Creates column-oriented RecordBatch
     - Uses `FileZoneBuilder` with DataFusion accumulators
     - Stores stats as Arrow IPC in global buffer
   - **Lines to Review**:
     - `FileWriter` struct: Added `column_stats_processors` field (~line 100)
     - `build_column_statistics()` method (~line 600-800)
     - Zone size: 1 million rows (constant)
     - Column-oriented layout: One row per dataset column
   - **Key Functions**:
     - `build_column_statistics()`: Main entry point
     - Uses `ListBuilder` for column-oriented storage
     - Serializes to Arrow IPC format

2. **`rust/lance-file/Cargo.toml`** (MODIFIED)
   - **Dependencies Added**:
     - `arrow-ipc.workspace = true`
     - `datafusion.workspace = true`
     - `datafusion-expr.workspace = true`
   - **Review**: Ensure dependencies are correct versions

**Review Focus**:
- ✅ Column-oriented layout (one row per dataset column)
- ✅ Zone size is 1 million rows
- ✅ Stats stored in global buffer with metadata key
- ✅ Forward/backward compatible (can add new stats later)
- ✅ Uses DataFusion accumulators for min/max

---

## 📋 Phase 3: Per-Fragment Statistics Reader

**Purpose**: Read column statistics from individual data files.

### Files to Review:

1. **`rust/lance-file/src/reader.rs`** (MODIFIED - ~305 lines added)
   - **Key Changes**:
     - `has_column_stats()`: Checks if file has stats
     - `read_column_stats()`: Reads and deserializes stats
   - **Lines to Review**:
     - `has_column_stats()` method (~line 500-510)
     - `read_column_stats()` method (~line 510-600)
     - Arrow IPC deserialization logic
     - Error handling for missing/malformed stats
   - **Key Functions**:
     - `has_column_stats()`: Quick check via metadata
     - `read_column_stats()`: Full read and deserialize
     - Handles multi-part buffers correctly

**Review Focus**:
- ✅ Efficient check via metadata (no file read)
- ✅ Correct Arrow IPC deserialization
- ✅ Handles missing stats gracefully
- ✅ Returns `Option<RecordBatch>` for safety

---

## 📋 Phase 4: Consolidation Core Module

**Purpose**: Consolidate per-fragment stats into a single dataset-level file.

### Files to Review:

1. **`rust/lance/src/dataset/column_stats.rs`** (NEW - 1,049 lines)
   - **Key Functions**:
     - `consolidate_column_stats()`: Main consolidation function
     - `fragment_has_stats()`: Check if fragment has stats
     - `read_fragment_column_stats()`: Read stats from fragment file
     - `build_consolidated_batch()`: Build column-oriented consolidated batch
     - `write_stats_file()`: Write consolidated stats to Lance file
   - **Lines to Review**:
     - `consolidate_column_stats()` (~line 60-150): Main logic
     - All-or-nothing policy check (~line 70-85)
     - Global offset calculation (~line 90-110)
     - `read_fragment_column_stats()` (~line 190-280): Parsing logic
     - `build_consolidated_batch()` (~line 280-400): Batch construction
     - `write_stats_file()` (~line 400-450): File writing
   - **Tests** (~line 540-1000):
     - `test_consolidation_all_fragments_have_stats`
     - `test_global_offset_calculation`
     - `test_empty_dataset`
     - `test_multiple_column_types`
     - `test_consolidation_single_fragment`
     - `test_consolidation_large_dataset`
     - `test_consolidation_with_nullable_columns`
   - **Key Data Structures**:
     - `ZoneStats`: Represents consolidated zone statistics
   - **Review Focus**:
     - ✅ All-or-nothing policy enforced correctly
     - ✅ Global offset calculation is correct
     - ✅ Column-oriented consolidated batch schema
     - ✅ File path resolution using `data_file_dir()`
     - ✅ Error handling for missing files

2. **`rust/lance/src/dataset.rs`** (MODIFIED)
   - **Changes**:
     - Added `pub mod column_stats;` declaration
   - **Review**: Just module declaration

**Review Focus**:
- ✅ All-or-nothing policy logic
- ✅ Global offset calculation correctness
- ✅ Column-oriented schema (7 rows: fragment_ids, zone_starts, zone_lengths, null_counts, nan_counts, min_values, max_values)
- ✅ File path handling with `data_file_dir()`
- ✅ Error messages are clear

---

## 📋 Phase 5: ColumnStatsReader with Auto Type Dispatch

**Purpose**: High-level API for reading consolidated stats with automatic type conversion.

### Files to Review:

1. **`rust/lance/src/dataset/column_stats_reader.rs`** (NEW - 397 lines)
   - **Key Structures**:
     - `ColumnStatsReader`: Main reader struct
     - `ColumnStats`: Result type with strongly-typed statistics
   - **Key Functions**:
     - `read_column_stats()`: Get stats for a column with auto type dispatch
     - `parse_scalar_value()`: Convert string to ScalarValue based on schema
     - `extract_numeric_value()`: Parse numeric strings
     - `extract_string_value()`: Parse string values
   - **Lines to Review**:
     - `ColumnStatsReader::new()` (~line 30-50)
     - `read_column_stats()` (~line 50-150): Main API
     - `parse_scalar_value()` (~line 150-300): Type dispatch logic
     - Supported types: Int8-64, UInt8-64, Float32/64, Utf8, LargeUtf8
   - **Review Focus**:
     - ✅ Type dispatch based on dataset schema
     - ✅ All numeric types handled correctly
     - ✅ String types handled correctly
     - ✅ Error handling for unsupported types
     - ✅ String parsing is robust

2. **`rust/lance/src/dataset.rs`** (MODIFIED)
   - **Changes**:
     - Added `pub mod column_stats_reader;` declaration
   - **Review**: Just module declaration

**Review Focus**:
- ✅ Type dispatch logic is correct for all supported types
- ✅ String parsing handles edge cases
- ✅ Error messages for unsupported types
- ✅ API is easy to use

---

## 📋 Phase 6: Compaction Integration

**Purpose**: Integrate consolidation into compaction workflow.

### Files to Review:

1. **`rust/lance/src/dataset/optimize.rs`** (MODIFIED - ~630 lines added)
   - **Key Changes**:
     - Added `consolidate_column_stats: bool` to `CompactionOptions` (default `true`)
     - Integration in `commit_compaction()` function
     - Separate `UpdateConfig` transaction for manifest update
   - **Lines to Review**:
     - `CompactionOptions` struct (~line 200-250): Added field
     - `commit_compaction()` method (~line 700-850): Integration logic
     - Consolidation call (~line 800-820)
     - Manifest update transaction (~line 820-850)
   - **Tests** (~line 3716-4000):
     - `test_compaction_with_column_stats_consolidation`
     - `test_compaction_skip_consolidation_when_disabled`
     - `test_compaction_with_deletions_preserves_stats`
     - `test_compaction_multiple_rounds_updates_stats`
     - `test_compaction_with_stable_row_ids_and_stats`
     - `test_compaction_no_fragments_to_compact_preserves_stats`
   - **Review Focus**:
     - ✅ Consolidation happens after rewrite transaction
     - ✅ Separate UpdateConfig transaction for safety
     - ✅ Consolidation can be disabled via options
     - ✅ Stats file path stored in manifest config
     - ✅ All compaction scenarios tested

**Review Focus**:
- ✅ Integration point is correct (after rewrite, before final commit)
- ✅ Two-phase commit (rewrite + config update) is safe
- ✅ Default behavior is correct (enabled by default)
- ✅ All edge cases handled

---

## 📋 Phase 7: Comprehensive Testing

**Purpose**: Ensure all scenarios are covered with comprehensive tests.

### Test Files to Review:

1. **`rust/lance/src/dataset/write/insert.rs`** (Tests section)
   - `test_column_stats_policy_set_on_create`
   - `test_column_stats_policy_not_set_when_disabled`

2. **`rust/lance/src/dataset/column_stats.rs`** (Tests section - ~line 540-1000)
   - `test_consolidation_all_fragments_have_stats`
   - `test_global_offset_calculation`
   - `test_empty_dataset`
   - `test_multiple_column_types`
   - `test_consolidation_single_fragment`
   - `test_consolidation_large_dataset`
   - `test_consolidation_with_nullable_columns`

3. **`rust/lance/src/dataset/optimize.rs`** (Tests section - ~line 3716-4000)
   - `test_compaction_with_column_stats_consolidation`
   - `test_compaction_skip_consolidation_when_disabled`
   - `test_compaction_with_deletions_preserves_stats`
   - `test_compaction_multiple_rounds_updates_stats`
   - `test_compaction_with_stable_row_ids_and_stats`
   - `test_compaction_no_fragments_to_compact_preserves_stats`

**Review Focus**:
- ✅ All major scenarios covered
- ✅ Edge cases tested
- ✅ Tests are clear and well-documented
- ✅ Tests use proper test infrastructure (TempStrDir, etc.)

---

## 📋 Quick Review Checklist

### Phase 0: Infrastructure
- [ ] `rust/lance-core/src/utils/zone.rs` - Zone utilities
- [ ] `rust/lance-index/src/scalar/zone_trainer.rs` - Zone trainer

### Phase 1: Policy
- [ ] `rust/lance/src/dataset/write.rs` - Policy enforcement
- [ ] `rust/lance/src/dataset/write/insert.rs` - Policy setting on create

### Phase 2: Writer
- [ ] `rust/lance-file/src/writer.rs` - `build_column_statistics()`
- [ ] `rust/lance-file/Cargo.toml` - Dependencies

### Phase 3: Reader
- [ ] `rust/lance-file/src/reader.rs` - `has_column_stats()`, `read_column_stats()`

### Phase 4: Consolidation
- [ ] `rust/lance/src/dataset/column_stats.rs` - Consolidation logic + tests

### Phase 5: Stats Reader
- [ ] `rust/lance/src/dataset/column_stats_reader.rs` - Type dispatch

### Phase 6: Compaction
- [ ] `rust/lance/src/dataset/optimize.rs` - Compaction integration + tests

### Phase 7: Tests
- [ ] All test files - Comprehensive coverage

---

## 📋 Key Design Decisions to Review

1. **Column-Oriented Layout**: One row per dataset column, fields are List types
   - Files: `writer.rs`, `column_stats.rs`
   - Why: 10-1000x faster for selective column reads

2. **All-or-Nothing Policy**: Only consolidate if ALL fragments have stats
   - Files: `column_stats.rs` (consolidate_column_stats)
   - Why: Prevents misleading partial statistics

3. **Global Offsets**: Adjust zone offsets to dataset-wide positions
   - Files: `column_stats.rs` (consolidate_column_stats)
   - Why: Query optimizer needs absolute row positions

4. **Two-Phase Commit**: Separate transactions for rewrite and config update
   - Files: `optimize.rs` (commit_compaction)
   - Why: Safety - failed consolidation doesn't corrupt dataset

5. **Policy Enforcement**: Prevent mixed-stat datasets at write time
   - Files: `write.rs`, `insert.rs`
   - Why: Consistency and user experience

---

## 📋 File Size Reference

- `rust/lance/src/dataset/column_stats.rs`: **1,049 lines** (largest file)
- `rust/lance/src/dataset/column_stats_reader.rs`: **397 lines**
- `rust/lance-file/src/writer.rs`: **+407 lines** (added)
- `rust/lance/src/dataset/optimize.rs`: **+630 lines** (added)
- `rust/lance-file/src/reader.rs`: **+305 lines** (added)

**Total**: ~4,200 lines of production code + tests

---

## 📋 Review Order Recommendation

1. **Start with Phase 0** (Infrastructure) - Understand the building blocks
2. **Phase 1** (Policy) - Understand the enforcement mechanism
3. **Phase 2** (Writer) - See how stats are collected
4. **Phase 3** (Reader) - See how stats are read from files
5. **Phase 4** (Consolidation) - Core consolidation logic
6. **Phase 5** (Stats Reader) - High-level API
7. **Phase 6** (Compaction) - Integration point
8. **Phase 7** (Tests) - Verify coverage

This order ensures you understand each layer before moving to the next.

---

**Last Updated**: December 17, 2024  
**Branch**: `add-column-stats-mvp`  
**Status**: All tests passing ✅
