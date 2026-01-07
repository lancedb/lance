# Column Statistics Design and Implementation Plan

## Overview

Column statistics are collected at two levels in Lance:
1. **Per-Fragment Level**: Statistics stored in each data file's footer
2. **Consolidated Level**: Statistics merged across all fragments during compaction

This document provides a complete design specification and implementation roadmap.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Per-Fragment Statistics](#per-fragment-statistics)
3. [Consolidated Statistics](#consolidated-statistics)
4. [Dataset-Level Policy](#dataset-level-policy)
5. [Reading Consolidated Stats](#reading-consolidated-stats)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Current Status](#current-status)

---

## Design Principles

### Core Requirements
1. ✅ **All-or-Nothing**: Either all fragments have statistics or consolidation is skipped
2. ✅ **Dataset-Level Policy**: `lance.column_stats.enabled` enforced across all writes
3. ✅ **Type-Preserving**: Min/max stored in native Arrow types
4. ✅ **Selective Loading**: Read only columns you need via projection
5. ✅ **Scalable**: Handles millions of columns efficiently
6. ✅ **Global Offsets**: Consolidated stats use dataset-wide row positions

### Key Decisions
- **Zone Size**: 1 million rows per zone (configurable)
- **Statistics Tracked**: min, max, null_count, nan_count per zone
- **Storage Format**: Arrow IPC for per-fragment, Lance file for consolidated
- **Column-Centric**: Stats organized by column for efficient access

---

## Per-Fragment Statistics

### Storage Location
Stored in each Lance data file's **global buffer** (footer section).

### Schema

```rust
Schema {
    fields: [
        Field { name: "column_name", data_type: Utf8, nullable: false },
        Field { name: "zone_start", data_type: UInt64, nullable: false },
        Field { name: "zone_length", data_type: UInt64, nullable: false },
        Field { name: "null_count", data_type: UInt32, nullable: false },
        Field { name: "nan_count", data_type: UInt32, nullable: false },
        Field { name: "min", data_type: Utf8, nullable: false },
        Field { name: "max", data_type: Utf8, nullable: false },
    ],
    metadata: {
        "lance:column_stats:version": "1"
    }
}
```

### Data Example

For a fragment with 2M rows and 3 columns:

```
┌─────────────┬────────────┬─────────────┬────────────┬───────────┬─────────────────┬─────────────────┐
│ column_name │ zone_start │ zone_length │ null_count │ nan_count │ min             │ max             │
├─────────────┼────────────┼─────────────┼────────────┼───────────┼─────────────────┼─────────────────┤
│ "age"       │ 0          │ 1000000     │ 0          │ 0         │ "Int32(18)"     │ "Int32(65)"     │
│ "age"       │ 1000000    │ 1000000     │ 5          │ 0         │ "Int32(20)"     │ "Int32(70)"     │
│ "id"        │ 0          │ 1000000     │ 0          │ 0         │ "Int64(1)"      │ "Int64(1000000)"│
│ "id"        │ 1000000    │ 1000000     │ 0          │ 0         │ "Int64(1000001)"│ "Int64(2000000)"│
│ "name"      │ 0          │ 1000000     │ 100        │ 0         │ "Utf8(\"Alice\")"│ "Utf8(\"Zoe\")"│
│ "name"      │ 1000000    │ 1000000     │ 50         │ 0         │ "Utf8(\"Aaron\")"│ "Utf8(\"Zack\")"│
└─────────────┴────────────┴─────────────┴────────────┴───────────┴─────────────────┴─────────────────┘
```

**Notes**:
- `zone_start` and `zone_length` are **fragment-local** offsets (always start at 0)
- `min` and `max` use Arrow's `ScalarValue` debug format
- Zone size: 1 million rows (configurable via `COLUMN_STATS_ZONE_SIZE`)

### Storage Implementation

```rust
// In FileWriter::build_column_statistics()

// 1. Serialize RecordBatch to Arrow IPC format
let mut buffer = Vec::new();
let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buffer, &stats_batch.schema())?;
writer.write(&stats_batch)?;
writer.finish()?;

// 2. Store as global buffer
let buffer_bytes = Bytes::from(buffer);
let buffer_index = self.add_global_buffer(buffer_bytes).await?;

// 3. Record in schema metadata
self.schema_metadata.insert(
    "lance:column_stats:buffer_index".to_string(),
    buffer_index.to_string(),
);
self.schema_metadata.insert(
    "lance:column_stats:version".to_string(),
    "1".to_string(),
);
```

### Implementation Status
✅ **Complete** - Implemented in `rust/lance-file/src/writer.rs`

---

## Consolidated Statistics

### When Created
During dataset **compaction**, if ALL fragments have column statistics.

### Storage Location
```
_stats/
└── column_stats_v{version}.lance
```

### All-or-Nothing Policy

**Consolidation only happens if ALL fragments have statistics**:

```rust
// Pre-check before consolidation
let total_fragments = dataset.get_fragments().len();
let mut fragments_with_stats = 0;

for fragment in dataset.get_fragments() {
    if fragment_has_stats(fragment) {
        fragments_with_stats += 1;
    }
}

if fragments_with_stats < total_fragments {
    log::info!(
        "Skipping consolidation: only {}/{} fragments have stats",
        fragments_with_stats, total_fragments
    );
    return Ok(None);
}
```

**Rationale**: Partial statistics can mislead the query optimizer. Better to have none than incomplete data.

### Schema Design

**Single Lance file with 7 rows**, where each column represents a dataset column:

```rust
Schema {
    fields: [
        // One field per dataset column
        Field { name: "age", data_type: LargeBinary, nullable: false },
        Field { name: "id", data_type: LargeBinary, nullable: false },
        Field { name: "name", data_type: LargeBinary, nullable: false },
        Field { name: "price", data_type: LargeBinary, nullable: false },
        // ... millions of columns possible
    ],
    metadata: {
        "lance:stats:version": "1",
        "lance:stats:dataset_version": "{version}"
    }
}
```

### Data Layout: 7 Rows

```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ age                     │ id                      │ name                    │
│ (LargeBinary)           │ (LargeBinary)           │ (LargeBinary)           │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ <binary: [0, 1, 2]>     │ <binary: [0, 1, 2]>     │ <binary: [0, 1, 2]>     │  ← Row 0: fragment_ids
│ <binary: [0, 1M, 2M]>   │ <binary: [0, 1M, 2M]>   │ <binary: [0, 1M, 2M]>   │  ← Row 1: zone_starts (GLOBAL)
│ <binary: [1M, 1M, 500K]>│ <binary: [1M, 1M, 500K]>│ <binary: [1M, 1M, 500K]>│  ← Row 2: zone_lengths
│ <binary: [0, 5, 2]>     │ <binary: [0, 0, 0]>     │ <binary: [100, 50, 25]> │  ← Row 3: null_counts
│ <binary: [0, 0, 0]>     │ <binary: [0, 0, 0]>     │ <binary: [0, 0, 0]>     │  ← Row 4: nan_counts
│ <binary: Arrow Array>   │ <binary: Arrow Array>   │ <binary: Arrow Array>   │  ← Row 5: min_values
│ <binary: Arrow Array>   │ <binary: Arrow Array>   │ <binary: Arrow Array>   │  ← Row 6: max_values
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### Binary Encoding Format

Each `LargeBinary` cell contains an **Arrow IPC-encoded array**.

#### Rows 0-4: Numeric Arrays

```rust
// Row 0: fragment_ids (UInt64Array)
let array = UInt64Array::from(vec![0, 1, 2]);
let encoded = encode_arrow_array(&array)?;

// Row 1: zone_starts (UInt64Array) - GLOBAL offsets
let array = UInt64Array::from(vec![0, 1_000_000, 2_000_000]);
let encoded = encode_arrow_array(&array)?;

// Row 2: zone_lengths (UInt64Array)
let array = UInt64Array::from(vec![1_000_000, 1_000_000, 500_000]);
let encoded = encode_arrow_array(&array)?;

// Row 3: null_counts (UInt32Array)
let array = UInt32Array::from(vec![0, 5, 2]);
let encoded = encode_arrow_array(&array)?;

// Row 4: nan_counts (UInt32Array)
let array = UInt32Array::from(vec![0, 0, 0]);
let encoded = encode_arrow_array(&array)?;
```

#### Rows 5-6: Type-Specific Arrays

**For "age" column (Int32)**:
```rust
// Row 5: min_values
let array = Int32Array::from(vec![18, 20, 25]);
let encoded = encode_arrow_array(&array)?;

// Row 6: max_values
let array = Int32Array::from(vec![65, 70, 80]);
let encoded = encode_arrow_array(&array)?;
```

**For "name" column (Utf8)**:
```rust
// Row 5: min_values
let array = StringArray::from(vec!["Alice", "Aaron", "Adam"]);
let encoded = encode_arrow_array(&array)?;

// Row 6: max_values
let array = StringArray::from(vec!["Zoe", "Zack", "Zara"]);
let encoded = encode_arrow_array(&array)?;
```

**For "price" column (Float64)**:
```rust
// Row 5: min_values
let array = Float64Array::from(vec![9.99, 5.50, 12.00]);
let encoded = encode_arrow_array(&array)?;

// Row 6: max_values
let array = Float64Array::from(vec![99.99, 150.00, 200.00]);
let encoded = encode_arrow_array(&array)?;
```

### Encoding/Decoding Helpers

```rust
fn encode_arrow_array(array: &dyn Array) -> Result<Vec<u8>> {
    let field = Field::new("values", array.data_type().clone(), false);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array.to_owned())])?;
    
    let mut buffer = Vec::new();
    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buffer, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    
    Ok(buffer)
}

fn decode_arrow_array(bytes: &[u8]) -> Result<ArrayRef> {
    let mut reader = arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None)?;
    let batch = reader.next().unwrap()?;
    Ok(batch.column(0).clone())
}
```

### Why This Design?

1. **Column-Centric Access**: Operations typically need stats for specific columns
   - Query: `WHERE age > 50` only needs "age" column stats
   - Lance projection: `read_all().with_projection(vec!["age"])` reads only that column

2. **Scalable to Millions of Columns**: 
   - Fixed 7 rows regardless of column count
   - Each column is a separate field → selective loading

3. **Type-Preserving**:
   - Min/max stored in native Arrow types (Int32Array, StringArray, etc.)
   - No string parsing or type conversion needed

4. **Efficient Storage**:
   - LargeBinary allows arbitrary-sized arrays
   - Arrow IPC is compact and well-compressed
   - Columnar storage within the file

### Implementation Status
⏳ **Planned** - To be implemented in Phase 3-4

---

## Dataset-Level Policy

### Manifest Configuration

When creating a dataset with column stats:

```rust
manifest.config.insert(
    "lance.column_stats.enabled",
    "true"
);
```

After consolidation:

```rust
manifest.config.insert(
    "lance.column_stats.file",
    "_stats/column_stats_v{version}.lance"
);
```

### Policy Enforcement

All write operations validate against the dataset policy:

```rust
// In write_fragments_internal()
params.validate_column_stats_policy(dataset)?;

// Validation logic
pub fn validate_column_stats_policy(&self, dataset: Option<&Dataset>) -> Result<()> {
    if let Some(dataset) = dataset {
        if let Some(policy_str) = dataset.manifest.config.get("lance.column_stats.enabled") {
            let dataset_policy: bool = policy_str.parse()?;
            
            if self.enable_column_stats != dataset_policy {
                return Err(Error::invalid_input(
                    format!(
                        "Column statistics policy mismatch: dataset requires {}, \
                         but WriteParams has {}. Use WriteParams::for_dataset() \
                         to inherit the correct policy.",
                        dataset_policy,
                        self.enable_column_stats
                    ),
                    location!(),
                ));
            }
        }
    }
    Ok(())
}
```

### Inheriting Policy

```rust
// Helper to create WriteParams that respect dataset policy
impl WriteParams {
    pub fn for_dataset(dataset: &Dataset) -> Self {
        let enable_column_stats = dataset
            .manifest
            .config
            .get("lance.column_stats.enabled")
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        Self {
            enable_column_stats,
            ..Default::default()
        }
    }
}
```

### Update Operations

`UpdateBuilder` automatically reads the policy:

```rust
impl UpdateBuilder {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        // Check if column stats are enabled in dataset config
        let enable_column_stats = dataset
            .manifest
            .config
            .get("lance.column_stats.enabled")
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        Self {
            dataset,
            enable_column_stats,
            // ... other fields
        }
    }
    
    // Can be overridden
    pub fn enable_column_stats(mut self, enable: bool) -> Self {
        self.enable_column_stats = enable;
        self
    }
}
```

### Delete Operations

Delete operations **do not modify data files**:
- They create/update a separate deletion vector file
- The file footer (including column statistics) remains unchanged
- ✅ Already correct - no implementation needed

### Implementation Status
🟡 **Partial** - Validation exists, but manifest config not set on creation (Phase 1)

---

## Reading Consolidated Stats

### Automatic Type Dispatching

The key insight: **Use the dataset schema to automatically determine column types**.

### ColumnStatsReader API

```rust
pub struct ColumnStatsReader {
    dataset_schema: Arc<Schema>,
    stats_batch: RecordBatch,
}

pub struct ColumnStats {
    pub fragment_ids: Vec<u64>,
    pub zone_starts: Vec<u64>,
    pub zone_lengths: Vec<u64>,
    pub null_counts: Vec<u32>,
    pub nan_counts: Vec<u32>,
    pub min_values: Vec<ScalarValue>,
    pub max_values: Vec<ScalarValue>,
}

impl ColumnStatsReader {
    pub fn new(dataset_schema: Arc<Schema>, stats_batch: RecordBatch) -> Self {
        Self { dataset_schema, stats_batch }
    }
    
    /// Read all statistics for a column, with automatic type dispatching
    pub fn read_column_stats(&self, column_name: &str) -> Result<ColumnStats> {
        // 1. Get column type from dataset schema
        let field = self.dataset_schema.field(column_name)?;
        let data_type = field.data_type();
        
        // 2. Get the column from stats batch
        let stats_column = self.stats_batch.column_by_name(column_name)?
            .as_any().downcast_ref::<LargeBinaryArray>()?;
        
        // 3. Decode rows 0-4 (same for all types)
        let fragment_ids = self.decode_u64_array(stats_column.value(0))?;
        let zone_starts = self.decode_u64_array(stats_column.value(1))?;
        let zone_lengths = self.decode_u64_array(stats_column.value(2))?;
        let null_counts = self.decode_u32_array(stats_column.value(3))?;
        let nan_counts = self.decode_u32_array(stats_column.value(4))?;
        
        // 4. Decode rows 5-6 (min/max) based on type - AUTOMATIC!
        let (min_values, max_values) = self.decode_min_max(
            stats_column.value(5),
            stats_column.value(6),
            data_type  // Type from schema
        )?;
        
        Ok(ColumnStats {
            fragment_ids,
            zone_starts,
            zone_lengths,
            null_counts,
            nan_counts,
            min_values,
            max_values,
        })
    }
    
    /// Automatically dispatch min/max decoding based on data type
    fn decode_min_max(
        &self,
        min_bytes: &[u8],
        max_bytes: &[u8],
        data_type: &DataType,
    ) -> Result<(Vec<ScalarValue>, Vec<ScalarValue>)> {
        match data_type {
            DataType::Int32 => {
                let mins = self.decode_typed_array::<Int32Array>(min_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Int32(v))
                    .collect();
                let maxs = self.decode_typed_array::<Int32Array>(max_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Int32(v))
                    .collect();
                Ok((mins, maxs))
            }
            DataType::Int64 => {
                let mins = self.decode_typed_array::<Int64Array>(min_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Int64(v))
                    .collect();
                let maxs = self.decode_typed_array::<Int64Array>(max_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Int64(v))
                    .collect();
                Ok((mins, maxs))
            }
            DataType::Utf8 => {
                let mins = self.decode_typed_array::<StringArray>(min_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Utf8(v.map(|s| s.to_string())))
                    .collect();
                let maxs = self.decode_typed_array::<StringArray>(max_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Utf8(v.map(|s| s.to_string())))
                    .collect();
                Ok((mins, maxs))
            }
            DataType::Float64 => {
                let mins = self.decode_typed_array::<Float64Array>(min_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Float64(v))
                    .collect();
                let maxs = self.decode_typed_array::<Float64Array>(max_bytes)?
                    .iter()
                    .map(|v| ScalarValue::Float64(v))
                    .collect();
                Ok((mins, maxs))
            }
            // ... add all Arrow types
            _ => Err(Error::invalid_input(
                format!("Unsupported type: {:?}", data_type),
                location!()
            ))
        }
    }
}
```

### Usage Example

```rust
// Load consolidated stats
let stats_file = dataset.manifest.config.get("lance.column_stats.file")?;
let reader = FileReader::try_open(object_store, stats_file, None).await?;
let stats_batch = reader.read_all().await?;

// Create reader with dataset schema
let stats_reader = ColumnStatsReader::new(
    dataset.schema().clone(),
    stats_batch
);

// Read "age" stats - type is automatically Int32
let age_stats = stats_reader.read_column_stats("age")?;
// age_stats.min_values[0] is ScalarValue::Int32(Some(18))

// Read "name" stats - type is automatically Utf8
let name_stats = stats_reader.read_column_stats("name")?;
// name_stats.min_values[0] is ScalarValue::Utf8(Some("Alice"))

// Read "price" stats - type is automatically Float64
let price_stats = stats_reader.read_column_stats("price")?;
// price_stats.min_values[0] is ScalarValue::Float64(Some(9.99))

// No manual type dispatching needed! ✨
```

### Selective Column Loading

```rust
// Load stats for only "age" and "price" columns
let stats_batch = reader
    .read_all()
    .with_projection(vec!["age", "price"])  // Lance projection
    .await?;

// Only "age" and "price" columns are read from disk
// Other columns (even if there are millions) are not loaded
```

### Implementation Status
⏳ **Planned** - To be implemented in Phase 4

---

## Consolidation Algorithm

### High-Level Flow

```rust
pub async fn consolidate_column_stats(
    dataset: &Dataset,
    new_version: u64,
) -> Result<Option<String>> {
    
    // Step 1: Pre-check - ALL fragments must have stats (all-or-nothing)
    let total_fragments = dataset.get_fragments().len();
    let mut fragments_with_stats = 0;
    
    for fragment in dataset.get_fragments() {
        if fragment_has_stats(fragment).await? {
            fragments_with_stats += 1;
        }
    }
    
    if fragments_with_stats < total_fragments {
        log::info!(
            "Skipping consolidation: only {}/{} fragments have stats",
            fragments_with_stats, total_fragments
        );
        return Ok(None);
    }
    
    // Step 2: Build fragment offset map (for global offsets)
    let mut fragment_offsets = HashMap::new();
    let mut current_offset = 0u64;
    
    for fragment in dataset.get_fragments() {
        fragment_offsets.insert(fragment.id() as u64, current_offset);
        current_offset += fragment.count_rows().await? as u64;
    }
    
    // Step 3: Collect stats from all fragments
    let mut stats_by_column: HashMap<String, Vec<ZoneStats>> = HashMap::new();
    
    for fragment in dataset.get_fragments() {
        let base_offset = fragment_offsets[&(fragment.id() as u64)];
        
        for data_file in &fragment.metadata().files {
            let file_stats = read_fragment_column_stats(dataset, data_file).await?;
            
            for (col_name, zones) in file_stats {
                // Adjust zone_start to global offset
                let adjusted_zones: Vec<ZoneStats> = zones
                    .into_iter()
                    .map(|z| ZoneStats {
                        fragment_id: fragment.id() as u64,
                        zone_start: base_offset + z.zone_start,  // LOCAL → GLOBAL
                        zone_length: z.zone_length,
                        null_count: z.null_count,
                        nan_count: z.nan_count,
                        min: z.min,
                        max: z.max,
                    })
                    .collect();
                
                stats_by_column
                    .entry(col_name)
                    .or_default()
                    .extend(adjusted_zones);
            }
        }
    }
    
    // Step 4: Build consolidated file (7 rows, N columns)
    let consolidated_batch = build_consolidated_batch(
        stats_by_column,
        dataset.schema()
    )?;
    
    // Step 5: Write as Lance file
    let stats_path = format!("_stats/column_stats_v{}.lance", new_version);
    write_lance_file(
        dataset.object_store(),
        &dataset.base.child(&stats_path),
        consolidated_batch
    ).await?;
    
    log::info!(
        "Consolidated column stats from {} fragments into {}",
        total_fragments,
        stats_path
    );
    
    Ok(Some(stats_path))
}
```

### Building Consolidated RecordBatch

```rust
fn build_consolidated_batch(
    stats_by_column: HashMap<String, Vec<ZoneStats>>,
    dataset_schema: &Schema,
) -> Result<RecordBatch> {
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    
    // For each dataset column
    for field in dataset_schema.fields() {
        let col_name = &field.name;
        let zones = stats_by_column.get(col_name)
            .ok_or_else(|| Error::invalid_input(
                format!("No stats for column {}", col_name),
                location!()
            ))?;
        
        // Build 7 arrays for this column
        let fragment_ids_binary = encode_arrow_array(&UInt64Array::from(
            zones.iter().map(|z| z.fragment_id).collect::<Vec<_>>()
        ))?;
        
        let zone_starts_binary = encode_arrow_array(&UInt64Array::from(
            zones.iter().map(|z| z.zone_start).collect::<Vec<_>>()
        ))?;
        
        let zone_lengths_binary = encode_arrow_array(&UInt64Array::from(
            zones.iter().map(|z| z.zone_length).collect::<Vec<_>>()
        ))?;
        
        let null_counts_binary = encode_arrow_array(&UInt32Array::from(
            zones.iter().map(|z| z.null_count).collect::<Vec<_>>()
        ))?;
        
        let nan_counts_binary = encode_arrow_array(&UInt32Array::from(
            zones.iter().map(|z| z.nan_count).collect::<Vec<_>>()
        ))?;
        
        // Min/max need type-specific encoding
        let (min_binary, max_binary) = encode_min_max_for_type(
            zones,
            field.data_type()
        )?;
        
        // Create column with 7 rows
        let column = LargeBinaryArray::from(vec![
            fragment_ids_binary,
            zone_starts_binary,
            zone_lengths_binary,
            null_counts_binary,
            nan_counts_binary,
            min_binary,
            max_binary,
        ]);
        
        fields.push(Field::new(col_name, DataType::LargeBinary, false));
        columns.push(Arc::new(column) as ArrayRef);
    }
    
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns)
}
```

### Implementation Status
⏳ **Planned** - To be implemented in Phase 3

---

## Implementation Roadmap

### Phase 1: Complete Policy Enforcement (~45 minutes)

**Goal**: Ensure `lance.column_stats.enabled` is set in manifest on dataset creation.

**Files to Modify**:
1. `rust/lance/src/dataset/write/commit.rs` - Set manifest config on first write
2. Add tests for policy enforcement

**Tasks**:
- [ ] Find where manifest is created for new datasets
- [ ] Add logic to set `lance.column_stats.enabled` based on WriteParams
- [ ] Add test: create dataset with stats, verify manifest has config
- [ ] Add test: try to append with different policy, verify error
- [ ] Add test: `WriteParams::for_dataset()` inherits policy

**Success Criteria**:
- ✅ Manifest has `lance.column_stats.enabled` after first write
- ✅ All tests pass
- ✅ Policy validation catches mismatches

---

### Phase 2: Column Stats Reader Module (~30 minutes)

**Goal**: Create infrastructure to read per-fragment statistics from Lance files.

**Files to Create**:
1. `rust/lance-file/src/reader/column_stats.rs`

**Tasks**:
- [ ] Implement `read_column_stats_from_file(reader) -> Result<Option<RecordBatch>>`
- [ ] Implement `has_column_stats(reader) -> bool`
- [ ] Add module to `rust/lance-file/src/reader/mod.rs`

**Success Criteria**:
- ✅ Can read stats from file's global buffer
- ✅ Returns None if file has no stats
- ✅ Parses Arrow IPC correctly

---

### Phase 3: Consolidation Core Module (~2 hours)

**Goal**: Implement the consolidation logic that merges per-fragment stats.

**Files to Create**:
1. `rust/lance/src/dataset/optimize/column_stats.rs`

**Tasks**:
- [ ] Implement `encode_arrow_array(array) -> Result<Vec<u8>>`
- [ ] Implement `decode_arrow_array(bytes) -> Result<ArrayRef>`
- [ ] Implement `StatsCollector` struct
- [ ] Implement `consolidate_column_stats()` function
- [ ] Implement all-or-nothing checking
- [ ] Implement fragment offset calculation
- [ ] Implement stats collection from fragments
- [ ] Implement `build_consolidated_batch()`
- [ ] Implement type-specific min/max encoding
- [ ] Add module to `rust/lance/src/dataset/optimize/mod.rs`

**Success Criteria**:
- ✅ Consolidation skipped if any fragment lacks stats
- ✅ Global offsets calculated correctly
- ✅ 7-row Lance file created with LargeBinary columns
- ✅ Min/max encoded in native Arrow types

---

### Phase 4: Stats Reader with Auto Type Dispatching (~1.5 hours)

**Goal**: Provide clean API to read consolidated stats with automatic type handling.

**Files to Create**:
1. `rust/lance/src/dataset/column_stats_reader.rs`

**Tasks**:
- [ ] Implement `ColumnStatsReader` struct
- [ ] Implement `ColumnStats` struct
- [ ] Implement `read_column_stats(column_name)` with auto type dispatch
- [ ] Implement `decode_min_max()` with match on all Arrow types:
  - [ ] Int8, Int16, Int32, Int64
  - [ ] UInt8, UInt16, UInt32, UInt64
  - [ ] Float32, Float64
  - [ ] Utf8, LargeUtf8
  - [ ] Binary, LargeBinary
  - [ ] Date32, Date64
  - [ ] Timestamp variants
  - [ ] Decimal128, Decimal256
- [ ] Add helper methods: `decode_u64_array()`, `decode_u32_array()`, etc.
- [ ] Add module to `rust/lance/src/dataset/mod.rs`

**Success Criteria**:
- ✅ No manual type specification needed
- ✅ Type deduced from dataset schema
- ✅ All common Arrow types supported
- ✅ Clean API: `reader.read_column_stats("age")?`

---

### Phase 5: Integration into Compaction (~45 minutes)

**Goal**: Wire consolidation into the compaction flow.

**Files to Modify**:
1. `rust/lance/src/dataset/optimize.rs`

**Tasks**:
- [ ] Add `consolidate_column_stats: bool` to `CompactionOptions`
- [ ] Set default to `true` in `CompactionOptions::default()`
- [ ] Find where compaction commits (likely `commit_compaction()`)
- [ ] Call `consolidate_column_stats()` before commit
- [ ] Add stats file path to manifest config if consolidation succeeds

**Success Criteria**:
- ✅ Compaction with `consolidate_column_stats=true` creates stats file
- ✅ Manifest has `lance.column_stats.file` after compaction
- ✅ Can opt out with `consolidate_column_stats=false`

---

### Phase 6: Testing (~2.5 hours)

**Goal**: Comprehensive tests for consolidation feature.

**Files to Create**:
1. `rust/lance/src/dataset/optimize/column_stats_tests.rs` or add to existing test file

**Test Cases**:
- [ ] `test_consolidate_all_fragments_have_stats`
  - Create dataset with 3 fragments, all with stats
  - Run consolidation
  - Verify consolidated file exists
  - Verify stats are correct
  - Verify global offsets are correct

- [ ] `test_consolidate_skipped_when_fragments_lack_stats`
  - Create dataset with mixed stats/no-stats fragments
  - Run consolidation
  - Verify consolidation was skipped
  - Verify no consolidated file created

- [ ] `test_consolidate_different_column_types`
  - Create dataset with Int32, Int64, Float64, Utf8 columns
  - All fragments with stats
  - Run consolidation
  - Verify each column type preserved correctly

- [ ] `test_stats_reader_automatic_type_dispatch`
  - Create consolidated stats
  - Read with ColumnStatsReader
  - Verify no manual type specification needed
  - Verify correct types returned

- [ ] `test_selective_column_loading`
  - Create dataset with 100 columns
  - Consolidate
  - Read stats for only 2 columns via projection
  - Verify API works (hard to verify actual I/O savings)

- [ ] `test_consolidation_offset_calculation`
  - Create dataset with 3 fragments of different sizes
  - Fragment 0: 500K rows
  - Fragment 1: 1M rows
  - Fragment 2: 750K rows
  - Consolidate
  - Verify zone_starts are [0, 500K, 1.5M] for each column

- [ ] `test_compaction_with_consolidation`
  - Create dataset with many small fragments
  - Enable column stats
  - Run compaction with `consolidate_column_stats=true`
  - Verify both compacted AND consolidated

- [ ] `test_policy_enforcement_across_operations`
  - Create dataset with stats enabled
  - Try insert with stats disabled -> error
  - Try update with stats disabled -> error
  - Update with stats enabled -> success

**Success Criteria**:
- ✅ All test cases pass
- ✅ Good coverage of edge cases
- ✅ Tests are maintainable and well-documented

---

## Timeline Estimates

| Phase | Description            | Time      | Cumulative  |
| ----- | ---------------------- | --------- | ----------- |
| 1     | Policy enforcement     | 45 min    | 45 min      |
| 2     | Stats reader module    | 30 min    | 1h 15min    |
| 3     | Consolidation core     | 2 hours   | 3h 15min    |
| 4     | Stats reader API       | 1.5 hours | 4h 45min    |
| 5     | Compaction integration | 45 min    | 5h 30min    |
| 6     | Testing                | 2.5 hours | **8 hours** |

**Total estimated effort**: ~8 hours of focused implementation time

---

## Current Status

### ✅ Completed
1. Per-fragment statistics in file writer
   - Location: `rust/lance-file/src/writer.rs`
   - Feature: `ColumnStatisticsProcessor`, `FileZoneBuilder`
   
2. Dataset-level policy validation
   - Location: `rust/lance/src/dataset/write.rs`
   - Feature: `WriteParams::for_dataset()`, `validate_column_stats_policy()`

3. Update operations support
   - Location: `rust/lance/src/dataset/write/update.rs`
   - Feature: Respects `lance.column_stats.enabled` from manifest

4. Test for update with column stats
   - Location: `rust/lance/src/dataset/write/update.rs`
   - Test: `test_update_with_column_stats()`

### 🟡 Partial
- Policy enforcement: Validation exists but manifest config not set on creation

### ⏳ Pending
- Complete policy enforcement (Phase 1)
- Column stats reader module (Phase 2)
- Consolidation core (Phase 3)
- Stats reader with auto dispatch (Phase 4)
- Compaction integration (Phase 5)
- Comprehensive testing (Phase 6)

---

## Key Design Trade-offs

### 1. All-or-Nothing vs Partial Stats
**Choice**: All-or-nothing
**Rationale**: Partial statistics can mislead query optimizer. Better to have none than incomplete data.

### 2. Single File vs Multiple Files
**Choice**: Single file with 7 rows
**Rationale**: Atomic writes, simpler management, scales to millions of columns

### 3. Type-Specific Storage vs String Serialization
**Choice**: Type-specific (native Arrow types)
**Rationale**: More efficient, no parsing overhead, better compression

### 4. Manual Type Dispatch vs Automatic
**Choice**: Automatic using dataset schema
**Rationale**: Cleaner API, less error-prone, schema already has type info

### 5. Global Offsets vs Fragment-Local
**Choice**: Global offsets in consolidated stats
**Rationale**: Simplifies query planning, avoids offset translation at query time

---

## Success Metrics

### Functional
- [ ] All fragments have consistent statistics policy
- [ ] Consolidation produces correct 7-row Lance file
- [ ] Automatic type dispatching works for all common types
- [ ] Selective column loading works via projection
- [ ] Global offsets calculated correctly
- [ ] All-or-nothing behavior enforced

### Performance
- [ ] Reading 10 columns from 1M-column dataset is fast (<100ms)
- [ ] Consolidation completes in reasonable time
- [ ] Encoding/decoding doesn't dominate query time

### Code Quality
- [ ] Well-documented public APIs
- [ ] Comprehensive test coverage (>80%)
- [ ] No compilation warnings
- [ ] Follows Lance code conventions

---

## Future Enhancements

1. **Additional Statistics**
   - Distinct count (HyperLogLog sketch)
   - Histogram/quantiles
   - Bloom filters for membership tests

2. **Incremental Consolidation**
   - Update consolidated stats without full rebuild
   - Useful for append-heavy workloads

3. **Statistics-Based Query Optimization**
   - Zone pruning during scan
   - Cardinality estimation for joins
   - Histogram-based selectivity

4. **Typed Stats Reader**
   - Generic API: `read_column_stats_typed::<i32>("age")?`
   - Returns `TypedColumnStats<i32>` with native types

5. **Statistics Versioning**
   - Support multiple stats formats
   - Graceful migration between versions

---

## References

- [Per-Fragment Statistics Implementation](../rust/lance-file/src/writer.rs)
- [Zone Processing Infrastructure](../rust/lance-core/src/utils/zone.rs)
- [Zone Map Index](../rust/lance-index/src/scalar/zonemap.rs)
- [Dataset Write Operations](../rust/lance/src/dataset/write.rs)

---

**Document Version**: 1.0  
**Last Updated**: December 17, 2024  
**Status**: Design Complete, Implementation Pending
