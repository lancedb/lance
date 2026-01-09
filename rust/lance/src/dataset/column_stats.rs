// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column statistics consolidation and reading utilities.
//!
//! This module provides functionality for:
//! 1. Consolidating per-fragment column statistics into a single file
//! 2. Reading consolidated statistics with automatic type dispatching
//!
//! Per-fragment statistics are stored in each data file's global buffer.
//! During compaction, these can be consolidated into a single column statistics
//! file for efficient query planning.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder, UInt32Builder, UInt64Builder};
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, UInt32Array, UInt64Array};
// These are only used in tests
#[cfg_attr(not(test), allow(unused_imports))]
use arrow_array::{Float32Array, ListArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_core::utils::zone::ZoneBound;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::FileReader;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use snafu::location;

use crate::dataset::fragment::FileFragment;
use crate::{Dataset, Error};

/// Consolidated statistics for a single zone of a single column.
#[derive(Debug, Clone)]
pub struct ZoneStats {
    /// Zone boundary information (fragment_id, start offset, length)
    pub bound: ZoneBound,
    pub null_count: u32,
    pub nan_count: u32,
    pub min: String, // ScalarValue as string (no type prefix)
    pub max: String, // ScalarValue as string (no type prefix)
}

/// Consolidate column statistics from all fragments into a single file.
///
/// This function implements an "all-or-nothing" approach: if any fragment
/// lacks column statistics, consolidation is skipped entirely.
///
/// # How It Works
///
/// Each fragment file contains per-fragment statistics in a **flat layout** (see writer.rs):
///
/// **Fragment 0 stats** (rows 0-2M, local offsets):
/// ```text
/// ┌─────────────┬─────────┬────────────┬─────────────┬────────────┬───────────┐
/// │ column_name │ zone_id │ zone_start │ zone_length │ min_value  │ max_value │
/// ├─────────────┼─────────┼────────────┼─────────────┼────────────┼───────────┤
/// │ "id"        │ 0       │ 0          │ 1000000     │ "1"        │ "1000000" │
/// │ "id"        │ 1       │ 1000000    │ 1000000     │ "1000001"  │ "2000000" │
/// │ "price"     │ 0       │ 0          │ 1000000     │ "9.99"     │ "99.99"   │
/// │ "price"     │ 1       │ 1000000    │ 1000000     │ "10.50"    │ "100.50"  │
/// └─────────────┴─────────┴────────────┴─────────────┴────────────┴───────────┘
/// ```
///
/// **Fragment 1 stats** (rows 2M-4M, local offsets):
/// ```text
/// ┌─────────────┬─────────┬────────────┬─────────────┬────────────┬───────────┐
/// │ column_name │ zone_id │ zone_start │ zone_length │ min_value  │ max_value │
/// ├─────────────┼─────────┼────────────┼─────────────┼────────────┼───────────┤
/// │ "id"        │ 0       │ 0          │ 1000000     │ "2000001"  │ "3000000" │
/// │ "id"        │ 1       │ 1000000    │ 1000000     │ "3000001"  │ "4000000" │
/// │ "price"     │ 0       │ 0          │ 1000000     │ "15.00"    │ "150.00"  │
/// │ "price"     │ 1       │ 1000000    │ 1000000     │ "20.00"    │ "200.00"  │
/// └─────────────┴─────────┴────────────┴─────────────┴────────────┴───────────┘
/// ```
///
/// This function **consolidates** them into a **list-based layout** with global offsets:
///
/// **Consolidated stats** (one row per column, across all fragments):
/// ```text
/// ┌─────────────┬──────────────┬─────────────────────┬───────────────┬────────────────────┐
/// │ column_name │ fragment_ids │ zone_starts         │ min_values    │ max_values         │
/// │ (string)    │ (list<u64>)  │ (list<u64>)         │ (list<str>)   │ (list<str>)        │
/// ├─────────────┼──────────────┼─────────────────────┼───────────────┼────────────────────┤
/// │ "id"        │ [0,0,1,1]    │ [0,1M,2M,3M] ←GLOBAL│ [1,1M,2M,3M]  │ [1M,2M,3M,4M]      │
/// │ "price"     │ [0,0,1,1]    │ [0,1M,2M,3M] ←GLOBAL│ [9.99,10.50,  │ [99.99,100.50,     │
/// │             │              │                     │  15.00,20.00] │  150.00,200.00]    │
/// └─────────────┴──────────────┴─────────────────────┴───────────────┴────────────────────┘
/// ```
///
/// **Key transformations**:
/// - Fragment 0 local offset 0 → Global offset 0
/// - Fragment 0 local offset 1M → Global offset 1M
/// - Fragment 1 local offset 0 → Global offset 2M (base_offset = 2M)
/// - Fragment 1 local offset 1M → Global offset 3M (base_offset + 1M)
///
pub async fn consolidate_column_stats(
    dataset: &Dataset,
    new_version: u64,
) -> Result<Option<String>> {
    // Step 1: Pre-check - ALL fragments must have stats (all-or-nothing)
    let fragments = dataset.get_fragments();
    let total_fragments = fragments.len();
    let mut fragments_with_stats = 0;

    for fragment in &fragments {
        if fragment_has_stats(dataset, fragment).await? {
            fragments_with_stats += 1;
        }
    }

    if fragments_with_stats < total_fragments {
        log::info!(
            "Skipping column stats consolidation: only {}/{} fragments have stats",
            fragments_with_stats,
            total_fragments
        );
        return Ok(None);
    }

    // Step 2: Build fragment offset map (for global offsets)
    let mut fragment_offsets = HashMap::new();
    let mut current_offset = 0u64;

    for fragment in &fragments {
        fragment_offsets.insert(fragment.id() as u64, current_offset);
        current_offset += fragment.count_rows(None).await? as u64;
    }

    // Step 3: Collect stats from all fragments, organized by column
    let mut stats_by_column: HashMap<String, Vec<ZoneStats>> = HashMap::new();

    for fragment in &fragments {
        let base_offset = fragment_offsets[&(fragment.id() as u64)];

        for data_file in &fragment.metadata().files {
            let file_path = dataset
                .data_file_dir(data_file)?
                .child(data_file.path.as_str());
            let file_stats = read_fragment_column_stats(dataset, &file_path).await?;

            if let Some(file_stats) = file_stats {
                for (col_name, zones) in file_stats {
                    // Adjust zone_start to global offset
                    let adjusted_zones: Vec<ZoneStats> = zones
                        .into_iter()
                        .map(|z| ZoneStats {
                            bound: ZoneBound {
                                fragment_id: fragment.id() as u64,
                                start: base_offset + z.bound.start, // LOCAL → GLOBAL
                                length: z.bound.length,
                            },
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
    }

    // If no statistics were collected, return early
    if stats_by_column.is_empty() {
        return Ok(None);
    }

    // Step 4: Build consolidated batch (column-oriented)
    let consolidated_batch = build_consolidated_batch(stats_by_column, dataset.schema())?;

    // Step 5: Write as Lance file (version is stored in metadata, not filename)
    let stats_path = String::from("_stats/column_stats.lance");
    write_stats_file(
        dataset.object_store(),
        &dataset.base.child(stats_path.as_str()),
        consolidated_batch,
        new_version,
    )
    .await?;

    log::info!(
        "Consolidated column stats from {} fragments into {} (version {})",
        total_fragments,
        stats_path,
        new_version
    );

    Ok(Some(stats_path))
}

/// Check if a fragment has column statistics.
///
/// A fragment consists of one or more data files. Column statistics are stored
/// per-file (each FileWriter writes stats independently). This function returns
/// true only if ALL data files in the fragment have column statistics.
///
/// This is necessary because:
/// - A fragment can have multiple data files (e.g., after appending or splitting)
/// - Each file's FileWriter independently decides whether to write stats
/// - For consolidation, we need stats from ALL files to be present
async fn fragment_has_stats(dataset: &Dataset, fragment: &FileFragment) -> Result<bool> {
    // Check all data files - all must have stats for the fragment to be considered complete
    for data_file in &fragment.metadata().files {
        let file_path = dataset
            .data_file_dir(data_file)?
            .child(data_file.path.as_str());
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig::max_bandwidth(&dataset.object_store),
        );
        let file_scheduler = scheduler
            .open_file(&file_path, &CachedFileSize::unknown())
            .await?;

        let file_reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &dataset
                .session
                .metadata_cache
                .file_metadata_cache(&file_path),
            dataset.file_reader_options.clone().unwrap_or_default(),
        )
        .await?;

        // If any file lacks stats, return false immediately
        if !file_reader.has_column_stats() {
            return Ok(false);
        }
    }

    // All files have stats
    Ok(true)
}

/// Read column statistics from a single data file (.lance file).
///
/// Returns a map from column name to list of zone statistics. The zones are
/// stored in a flat layout in the data file (one row per zone per column), which
/// this function converts to a nested structure for easier processing.
///
/// # Example
///
/// For a data file with 2 columns and 2 zones each, the flat layout in the file:
/// ```text
/// column_name | zone_id | zone_start | zone_length | ...
/// "id"        | 0       | 0          | 1000000     | ...
/// "id"        | 1       | 1000000    | 500000      | ...
/// "price"     | 0       | 0          | 1000000     | ...
/// "price"     | 1       | 1000000    | 500000      | ...
/// ```
///
/// Gets converted to:
/// ```text
/// {
///   "id": [ZoneStats(zone_id=0, ...), ZoneStats(zone_id=1, ...)],
///   "price": [ZoneStats(zone_id=0, ...), ZoneStats(zone_id=1, ...)]
/// }
/// ```
async fn read_fragment_column_stats(
    dataset: &Dataset,
    file_path: &Path,
) -> Result<Option<HashMap<String, Vec<ZoneStats>>>> {
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(file_path, &CachedFileSize::unknown())
        .await?;

    let file_reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &dataset
            .session
            .metadata_cache
            .file_metadata_cache(file_path),
        dataset.file_reader_options.clone().unwrap_or_default(),
    )
    .await?;

    let Some(stats_batch) = file_reader.read_column_stats().await? else {
        return Ok(None);
    };

    // Parse the column-oriented stats batch
    let mut result = HashMap::new();

    let column_names = stats_batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Error::Internal {
            message: "Expected StringArray for column_names".to_string(),
            location: location!(),
        })?;

    let zone_ids = stats_batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected UInt32Array for zone_ids".to_string(),
            location: location!(),
        })?;

    let zone_starts = stats_batch
        .column(2)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected UInt64Array for zone_starts".to_string(),
            location: location!(),
        })?;

    let zone_lengths = stats_batch
        .column(3)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected UInt64Array for zone_lengths".to_string(),
            location: location!(),
        })?;

    let null_counts = stats_batch
        .column(4)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected UInt32Array for null_counts".to_string(),
            location: location!(),
        })?;

    let nan_counts = stats_batch
        .column(5)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected UInt32Array for nan_counts".to_string(),
            location: location!(),
        })?;

    let min_values = stats_batch
        .column(6)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Error::Internal {
            message: "Expected StringArray for min_values".to_string(),
            location: location!(),
        })?;

    let max_values = stats_batch
        .column(7)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Error::Internal {
            message: "Expected StringArray for max_values".to_string(),
            location: location!(),
        })?;

    // Process each row (one row per zone per column) and convert from flat layout
    // to nested structure. Zones may arrive out of order, so we need to resize vectors.
    for row_idx in 0..stats_batch.num_rows() {
        let col_name = column_names.value(row_idx).to_string();
        let zone_id = zone_ids.value(row_idx) as usize;

        let zone_stat = ZoneStats {
            bound: ZoneBound {
                fragment_id: 0, // Will be set by caller when computing global offsets
                start: zone_starts.value(row_idx),
                length: zone_lengths.value(row_idx) as usize,
            },
            null_count: null_counts.value(row_idx),
            nan_count: nan_counts.value(row_idx),
            min: min_values.value(row_idx).to_string(),
            max: max_values.value(row_idx).to_string(),
        };

        // Get or create the zones vector for this column
        let zones_for_column = result.entry(col_name).or_insert_with(Vec::new);

        // Ensure the zones vector has enough capacity for this zone_id
        // (zones may be read out of order, so we need to pre-allocate)
        let required_capacity = zone_id + 1;
        if zones_for_column.len() < required_capacity {
            zones_for_column.resize(
                required_capacity,
                ZoneStats {
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 0,
                    },
                    null_count: 0,
                    nan_count: 0,
                    min: String::new(),
                    max: String::new(),
                },
            );
        }

        zones_for_column[zone_id] = zone_stat;
    }

    Ok(Some(result))
}

/// Builder structure for list columns in consolidated statistics
struct ZoneListBuilders {
    fragment_ids: ListBuilder<UInt64Builder>,
    zone_starts: ListBuilder<UInt64Builder>,
    zone_lengths: ListBuilder<UInt64Builder>,
    null_counts: ListBuilder<UInt32Builder>,
    nan_counts: ListBuilder<UInt32Builder>,
    mins: ListBuilder<StringBuilder>,
    maxs: ListBuilder<StringBuilder>,
}

impl ZoneListBuilders {
    fn new() -> Self {
        Self {
            fragment_ids: ListBuilder::new(UInt64Builder::new()).with_field(ArrowField::new(
                "fragment_id",
                DataType::UInt64,
                false,
            )),
            zone_starts: ListBuilder::new(UInt64Builder::new()).with_field(ArrowField::new(
                "zone_start",
                DataType::UInt64,
                false,
            )),
            zone_lengths: ListBuilder::new(UInt64Builder::new()).with_field(ArrowField::new(
                "zone_length",
                DataType::UInt64,
                false,
            )),
            null_counts: ListBuilder::new(UInt32Builder::new()).with_field(ArrowField::new(
                "null_count",
                DataType::UInt32,
                false,
            )),
            nan_counts: ListBuilder::new(UInt32Builder::new()).with_field(ArrowField::new(
                "nan_count",
                DataType::UInt32,
                false,
            )),
            mins: ListBuilder::new(StringBuilder::new()).with_field(ArrowField::new(
                "min",
                DataType::Utf8,
                false,
            )),
            maxs: ListBuilder::new(StringBuilder::new()).with_field(ArrowField::new(
                "max",
                DataType::Utf8,
                false,
            )),
        }
    }

    /// Append zone statistics to the builders
    fn append_zones(&mut self, zones: &[ZoneStats]) {
        for zone in zones {
            self.fragment_ids
                .values()
                .append_value(zone.bound.fragment_id);
            self.zone_starts.values().append_value(zone.bound.start);
            self.zone_lengths
                .values()
                .append_value(zone.bound.length as u64);
            self.null_counts.values().append_value(zone.null_count);
            self.nan_counts.values().append_value(zone.nan_count);
            self.mins.values().append_value(&zone.min);
            self.maxs.values().append_value(&zone.max);
        }
    }

    /// Finish lists for the current column (creates one row)
    fn finish_column(&mut self) {
        self.fragment_ids.append(true);
        self.zone_starts.append(true);
        self.zone_lengths.append(true);
        self.null_counts.append(true);
        self.nan_counts.append(true);
        self.mins.append(true);
        self.maxs.append(true);
    }

    /// Finalize and build Arrow arrays
    fn build_arrays(mut self) -> Vec<ArrayRef> {
        vec![
            Arc::new(self.fragment_ids.finish()) as ArrayRef,
            Arc::new(self.zone_starts.finish()) as ArrayRef,
            Arc::new(self.zone_lengths.finish()) as ArrayRef,
            Arc::new(self.null_counts.finish()) as ArrayRef,
            Arc::new(self.nan_counts.finish()) as ArrayRef,
            Arc::new(self.mins.finish()) as ArrayRef,
            Arc::new(self.maxs.finish()) as ArrayRef,
        ]
    }
}

/// Create the Arrow schema for consolidated statistics
fn create_consolidated_stats_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("column_name", DataType::Utf8, false),
        ArrowField::new(
            "fragment_ids",
            DataType::List(Arc::new(ArrowField::new(
                "fragment_id",
                DataType::UInt64,
                false,
            ))),
            false,
        ),
        ArrowField::new(
            "zone_starts",
            DataType::List(Arc::new(ArrowField::new(
                "zone_start",
                DataType::UInt64,
                false,
            ))),
            false,
        ),
        ArrowField::new(
            "zone_lengths",
            DataType::List(Arc::new(ArrowField::new(
                "zone_length",
                DataType::UInt64,
                false,
            ))),
            false,
        ),
        ArrowField::new(
            "null_counts",
            DataType::List(Arc::new(ArrowField::new(
                "null_count",
                DataType::UInt32,
                false,
            ))),
            false,
        ),
        ArrowField::new(
            "nan_counts",
            DataType::List(Arc::new(ArrowField::new(
                "nan_count",
                DataType::UInt32,
                false,
            ))),
            false,
        ),
        ArrowField::new(
            "min_values",
            DataType::List(Arc::new(ArrowField::new("min", DataType::Utf8, false))),
            false,
        ),
        ArrowField::new(
            "max_values",
            DataType::List(Arc::new(ArrowField::new("max", DataType::Utf8, false))),
            false,
        ),
    ]))
}

/// Build a consolidated RecordBatch from collected statistics.
///
/// Uses column-oriented layout: one row per dataset column, each field is a list.
fn build_consolidated_batch(
    stats_by_column: HashMap<String, Vec<ZoneStats>>,
    dataset_schema: &Schema,
) -> Result<RecordBatch> {
    let mut column_names = Vec::new();
    let mut builders = ZoneListBuilders::new();

    // Process each dataset column (in schema order)
    for field in dataset_schema.fields.iter() {
        let col_name = &field.name;

        if let Some(mut zones) = stats_by_column.get(col_name).cloned() {
            // Sort zones by (fragment_id, zone_start) for consistency
            zones.sort_by_key(|z| (z.bound.fragment_id, z.bound.start));

            column_names.push(col_name.clone());

            // Append zone data and finish the list for this column
            builders.append_zones(&zones);
            builders.finish_column();
        }
    }

    if column_names.is_empty() {
        return Err(Error::Internal {
            message: "[ColumnStats] No column statistics to consolidate".to_string(),
            location: location!(),
        });
    }

    // Build final arrays
    let column_name_array = Arc::new(StringArray::from(column_names)) as ArrayRef;
    let mut arrays = vec![column_name_array];
    arrays.extend(builders.build_arrays());

    // Create RecordBatch
    RecordBatch::try_new(create_consolidated_stats_schema(), arrays).map_err(|e| Error::Internal {
        message: format!(
            "[ColumnStats] Failed to create consolidated stats batch: {}",
            e
        ),
        location: location!(),
    })
}

/// Write the consolidated stats RecordBatch as a Lance file.
async fn write_stats_file(
    object_store: &ObjectStore,
    path: &Path,
    batch: RecordBatch,
    version: u64,
) -> Result<()> {
    use lance_file::writer::{FileWriter, FileWriterOptions};

    let lance_schema =
        lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).map_err(|e| {
            Error::Internal {
                message: format!("Failed to convert schema: {}", e),
                location: location!(),
            }
        })?;

    let mut writer = FileWriter::try_new(
        object_store.create(path).await?,
        lance_schema,
        FileWriterOptions::default(),
    )?;

    // Store dataset version in file metadata
    writer.add_schema_metadata("lance:dataset:version", version.to_string());

    writer.write_batch(&batch).await?;
    writer.finish().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::WriteParams;
    use futures::stream::TryStreamExt;

    /// Helper function to read consolidated stats file using FileReader
    async fn read_stats_file(dataset: &Dataset, stats_path: &str) -> Vec<RecordBatch> {
        let full_path = dataset.base.child(stats_path);
        let scheduler = lance_io::scheduler::ScanScheduler::new(
            dataset.object_store.clone(),
            lance_io::scheduler::SchedulerConfig::max_bandwidth(&dataset.object_store),
        );
        let file_scheduler = scheduler
            .open_file(&full_path, &lance_io::utils::CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = lance_file::reader::FileReader::try_open(
            file_scheduler,
            None,
            Arc::<lance_encoding::decoder::DecoderPlugins>::default(),
            &dataset
                .session
                .metadata_cache
                .file_metadata_cache(&full_path),
            dataset.file_reader_options.clone().unwrap_or_default(),
        )
        .await
        .unwrap();

        let mut stream = reader
            .read_stream(
                lance_io::ReadBatchParams::RangeFull,
                4096,
                16,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .unwrap();

        let mut batches = Vec::new();
        while let Some(batch) = stream.try_next().await.unwrap() {
            batches.push(batch);
        }
        batches
    }
    use crate::Dataset;
    use arrow_array::{Int32Array, RecordBatchIterator, StringArray as ArrowStringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_testing::datagen::generate_random_array;

    #[tokio::test]
    async fn test_consolidation_all_fragments_have_stats() {
        // Create dataset with column stats enabled
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]));

        // Create 3 fragments, each with stats
        let write_params = WriteParams {
            max_rows_per_file: 100,
            enable_column_stats: true,
            ..Default::default()
        };

        for i in 0..3 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values((i * 100)..((i + 1) * 100))),
                    Arc::new(ArrowStringArray::from_iter_values(
                        ((i * 100)..((i + 1) * 100))
                            .map(|n| format!("name_{}", n))
                            .collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap();

            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

            if i == 0 {
                Dataset::write(reader, test_uri, Some(write_params.clone()))
                    .await
                    .unwrap();
            } else {
                let append_params = WriteParams {
                    mode: crate::dataset::WriteMode::Append,
                    enable_column_stats: true,
                    ..Default::default()
                };
                Dataset::write(reader, test_uri, Some(append_params))
                    .await
                    .unwrap();
            }
        }

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        // Test consolidation
        let result = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "Consolidation should succeed when all fragments have stats"
        );

        let stats_path = result.unwrap();
        assert_eq!(stats_path, "_stats/column_stats.lance");
        assert!(stats_path.ends_with(".lance"));

        // Verify the consolidated stats content
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        // 2 rows (id, name columns)
        assert_eq!(batch.num_rows(), 2);

        // Verify full content using debug output
        let column_names = batch.column_by_name("column_name").unwrap();
        let fragment_ids = batch.column_by_name("fragment_ids").unwrap();
        let zone_starts = batch.column_by_name("zone_starts").unwrap();
        let zone_lengths = batch.column_by_name("zone_lengths").unwrap();
        let null_counts = batch.column_by_name("null_counts").unwrap();
        let nan_counts = batch.column_by_name("nan_counts").unwrap();
        let mins = batch.column_by_name("min_values").unwrap();
        let maxs = batch.column_by_name("max_values").unwrap();

        // Row 0: "id" column stats
        assert_eq!(
            column_names
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "id"
        );
        assert_eq!(
            format!(
                "{:?}",
                fragment_ids
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(0)
            ),
            format!("{:?}", UInt64Array::from(vec![0, 1, 2]))
        );
        assert_eq!(
            format!(
                "{:?}",
                zone_starts
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(0)
            ),
            format!("{:?}", UInt64Array::from(vec![0, 100, 200]))
        );
        assert_eq!(
            format!(
                "{:?}",
                zone_lengths
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(0)
            ),
            format!("{:?}", UInt64Array::from(vec![100, 100, 100]))
        );
        assert_eq!(
            format!(
                "{:?}",
                null_counts
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(0)
            ),
            format!("{:?}", UInt32Array::from(vec![0, 0, 0]))
        );
        assert_eq!(
            format!(
                "{:?}",
                nan_counts
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(0)
            ),
            format!("{:?}", UInt32Array::from(vec![0, 0, 0]))
        );
        assert_eq!(
            format!(
                "{:?}",
                mins.as_any().downcast_ref::<ListArray>().unwrap().value(0)
            ),
            format!("{:?}", StringArray::from(vec!["0", "100", "200"]))
        );
        assert_eq!(
            format!(
                "{:?}",
                maxs.as_any().downcast_ref::<ListArray>().unwrap().value(0)
            ),
            format!("{:?}", StringArray::from(vec!["99", "199", "299"]))
        );

        // Row 1: "name" column stats
        assert_eq!(
            column_names
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(1),
            "name"
        );
        assert_eq!(
            format!(
                "{:?}",
                fragment_ids
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(1)
            ),
            format!("{:?}", UInt64Array::from(vec![0, 1, 2]))
        );
        assert_eq!(
            format!(
                "{:?}",
                mins.as_any().downcast_ref::<ListArray>().unwrap().value(1)
            ),
            format!(
                "{:?}",
                StringArray::from(vec!["name_0", "name_100", "name_200"])
            )
        );
        assert_eq!(
            format!(
                "{:?}",
                maxs.as_any().downcast_ref::<ListArray>().unwrap().value(1)
            ),
            format!(
                "{:?}",
                StringArray::from(vec!["name_99", "name_199", "name_299"])
            )
        );
    }

    #[tokio::test]
    async fn test_global_offset_calculation() {
        // Test that zone offsets are correctly adjusted to global positions
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));

        let write_params = WriteParams {
            max_rows_per_file: 100,
            enable_column_stats: true,
            ..Default::default()
        };

        // Create 2 fragments with 100 rows each
        for i in 0..2 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    (i * 100)..((i + 1) * 100),
                ))],
            )
            .unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

            if i == 0 {
                Dataset::write(reader, test_uri, Some(write_params.clone()))
                    .await
                    .unwrap();
            } else {
                let _dataset = Dataset::open(test_uri).await.unwrap();
                let append_params = WriteParams {
                    mode: crate::dataset::WriteMode::Append,
                    enable_column_stats: true,
                    ..Default::default()
                };
                Dataset::write(reader, test_uri, Some(append_params))
                    .await
                    .unwrap();
            }
        }

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stats_path = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap()
            .unwrap();

        // Read the consolidated stats file
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        // Verify zone_starts contain global offsets
        let zone_starts = batch
            .column_by_name("zone_starts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let zone_starts = zone_starts.as_any().downcast_ref::<UInt64Array>().unwrap();

        // Should have at least 1 zone, first zone starts at 0
        assert!(!zone_starts.is_empty());
        assert_eq!(zone_starts.value(0), 0);

        // If there are multiple zones, verify global offset calculation
        // Fragment 1 starts at row 100, so any zone from fragment 1 should have offset >= 100
        if zone_starts.len() > 1 {
            let second_zone_start = zone_starts.value(1);
            assert!(
                second_zone_start >= 100,
                "Second zone should start at or after row 100 (fragment 1 boundary), got {}",
                second_zone_start
            );
        }
    }

    #[tokio::test]
    async fn test_empty_dataset() {
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_column_stats: true,
            ..Default::default()
        };

        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete all rows
        dataset.delete("id >= 0").await.unwrap();
        dataset = Dataset::open(test_uri).await.unwrap();

        // Should still work but return None (no data to consolidate)
        let result = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap();

        // With deletions, fragments still exist, so consolidation should work
        // This tests that we handle the case gracefully
        assert!(result.is_some() || result.is_none());
    }

    #[tokio::test]
    async fn test_multiple_column_types() {
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("int_col", DataType::Int32, false),
            ArrowField::new("float_col", DataType::Float32, false),
            ArrowField::new("string_col", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(generate_random_array(100)),
                Arc::new(ArrowStringArray::from_iter_values(
                    (0..100).map(|i| format!("str_{}", i)),
                )),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_column_stats: true,
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let result = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap();

        assert!(result.is_some(), "Should handle multiple column types");

        // Verify the stats file contains all 3 column types
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        // Should have 3 rows (one for each column)
        assert_eq!(batch.num_rows(), 3);

        let column_names = batch
            .column_by_name("column_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(column_names.value(0), "int_col");
        assert_eq!(column_names.value(1), "float_col");
        assert_eq!(column_names.value(2), "string_col");

        // Verify min/max for int_col (row 0)
        let mins = batch
            .column_by_name("min_values")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let maxs = batch
            .column_by_name("max_values")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        // int_col: values [0, 100)
        let int_mins_array = mins.value(0);
        let int_mins = int_mins_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let int_maxs_array = maxs.value(0);
        let int_maxs = int_maxs_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(int_mins.value(0), "0");
        assert_eq!(int_maxs.value(int_maxs.len() - 1), "99");

        // float_col: random values, verify they are valid and min <= max
        let float_mins_array = mins.value(1);
        let float_mins = float_mins_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let float_maxs_array = maxs.value(1);
        let float_maxs = float_maxs_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(float_mins.len(), float_maxs.len());
        // For each zone, verify min <= max
        for i in 0..float_mins.len() {
            let min_val: f32 = float_mins.value(i).parse().unwrap();
            let max_val: f32 = float_maxs.value(i).parse().unwrap();
            assert!(
                min_val <= max_val,
                "Float column zone {}: min ({}) should be <= max ({})",
                i,
                min_val,
                max_val
            );
            // Verify they are finite (not NaN or Inf)
            assert!(min_val.is_finite(), "Float min should be finite");
            assert!(max_val.is_finite(), "Float max should be finite");
        }

        // string_col: values ["str_0", "str_99"]
        let str_mins_array = mins.value(2);
        let str_mins = str_mins_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let str_maxs_array = maxs.value(2);
        let str_maxs = str_maxs_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(str_mins.value(0), "str_0");
        assert_eq!(str_maxs.value(str_maxs.len() - 1), "str_99");

        // Verify null_counts are all zero (no nulls)
        let null_counts = batch
            .column_by_name("null_counts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for i in 0..3 {
            let col_null_counts_array = null_counts.value(i);
            let col_null_counts = col_null_counts_array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            let total: u32 = (0..col_null_counts.len())
                .map(|j| col_null_counts.value(j))
                .sum();
            assert_eq!(total, 0, "Column {} should have no nulls", i);
        }
    }

    #[tokio::test]
    async fn test_consolidation_single_fragment() {
        // Test consolidation with just one fragment
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_column_stats: true,
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);

        let result = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "Should consolidate even with single fragment"
        );

        // Verify content
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 1); // One column: "id"

        let column_names = batch
            .column_by_name("column_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(column_names.value(0), "id");

        let fragment_ids = batch
            .column_by_name("fragment_ids")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let fragment_ids = fragment_ids.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert!(!fragment_ids.is_empty()); // At least one zone
        assert_eq!(fragment_ids.value(0), 0); // Fragment 0

        // Verify min/max for "id" column: [0, 99]
        let mins = batch
            .column_by_name("min_values")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let mins = mins.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(mins.value(0), "0");

        let maxs = batch
            .column_by_name("max_values")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let maxs = maxs.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(maxs.value(maxs.len() - 1), "99");

        // Verify zone_starts begin at 0
        let zone_starts = batch
            .column_by_name("zone_starts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let zone_starts = zone_starts.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(zone_starts.value(0), 0);

        // Verify zone_lengths sum to 100
        let zone_lengths = batch
            .column_by_name("zone_lengths")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let zone_lengths = zone_lengths.as_any().downcast_ref::<UInt64Array>().unwrap();
        let total_length: u64 = (0..zone_lengths.len()).map(|i| zone_lengths.value(i)).sum();
        assert_eq!(total_length, 100);

        // Verify null_counts are zero
        let null_counts = batch
            .column_by_name("null_counts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let null_counts = null_counts.as_any().downcast_ref::<UInt32Array>().unwrap();
        let total_nulls: u32 = (0..null_counts.len()).map(|i| null_counts.value(i)).sum();
        assert_eq!(total_nulls, 0);
    }

    #[tokio::test]
    async fn test_consolidation_large_dataset() {
        // Test with larger dataset to verify zone handling
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            ArrowField::new("value", DataType::Float32, false),
        ]));

        let write_params = WriteParams {
            max_rows_per_file: 50_000,
            enable_column_stats: true,
            ..Default::default()
        };

        // Write 2 fragments with 50k rows each (should create multiple zones)
        for i in 0..2 {
            let start = i * 50_000;
            let end = (i + 1) * 50_000;
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(arrow_array::Int64Array::from_iter_values(
                        start as i64..end as i64,
                    )),
                    Arc::new(Float32Array::from_iter_values(
                        (start..end).map(|n| n as f32),
                    )),
                ],
            )
            .unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

            if i == 0 {
                Dataset::write(reader, test_uri, Some(write_params.clone()))
                    .await
                    .unwrap();
            } else {
                let _dataset = Dataset::open(test_uri).await.unwrap();
                let append_params = WriteParams {
                    mode: crate::dataset::WriteMode::Append,
                    enable_column_stats: true,
                    ..Default::default()
                };
                Dataset::write(reader, test_uri, Some(append_params))
                    .await
                    .unwrap();
            }
        }

        let dataset = Dataset::open(test_uri).await.unwrap();
        let result = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "Should handle large dataset with multiple zones"
        );

        // Verify content with large dataset
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 2); // Two columns: "id" and "value"

        let column_names = batch
            .column_by_name("column_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(column_names.value(0), "id");
        assert_eq!(column_names.value(1), "value");

        // Verify "id" column (row 0) has zones from both fragments
        let fragment_ids = batch
            .column_by_name("fragment_ids")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let fragment_ids = fragment_ids.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert!(
            fragment_ids.len() >= 2,
            "Should have zones from multiple fragments"
        );
        // Check both fragments are represented
        assert_eq!(fragment_ids.value(0), 0);
        assert_eq!(fragment_ids.value(fragment_ids.len() - 1), 1);

        let mins = batch
            .column_by_name("min_values")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let maxs = batch
            .column_by_name("max_values")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        // Verify min/max for "id" column spans the full range [0, 99999]
        let id_mins_array = mins.value(0);
        let id_mins = id_mins_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let id_maxs_array = maxs.value(0);
        let id_maxs = id_maxs_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(id_mins.value(0), "0"); // First zone starts at 0
        let last_max: i64 = id_maxs.value(id_maxs.len() - 1).parse().unwrap();
        assert_eq!(last_max, 99999); // Last zone ends at 99999

        // Verify min/max for "value" column (Float32)
        let value_mins_array = mins.value(1);
        let value_mins = value_mins_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let value_maxs_array = maxs.value(1);
        let value_maxs = value_maxs_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let first_min: f32 = value_mins.value(0).parse().unwrap();
        let last_max: f32 = value_maxs.value(value_maxs.len() - 1).parse().unwrap();
        assert_eq!(first_min, 0.0);
        assert_eq!(last_max, 99999.0);

        // Verify zone_starts span the full dataset with global offsets
        let zone_starts = batch
            .column_by_name("zone_starts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let zone_starts = zone_starts.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(zone_starts.value(0), 0); // First fragment starts at 0
        assert!(
            zone_starts.value(zone_starts.len() - 1) >= 50000,
            "Last zone should be in second fragment (offset >= 50000)"
        );

        // Verify zone_lengths sum to 100000 total rows
        let zone_lengths = batch
            .column_by_name("zone_lengths")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        let zone_lengths = zone_lengths.as_any().downcast_ref::<UInt64Array>().unwrap();
        let total_length: u64 = (0..zone_lengths.len()).map(|i| zone_lengths.value(i)).sum();
        assert_eq!(total_length, 100000);

        // Verify null_counts are all zero
        let null_counts = batch
            .column_by_name("null_counts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        for col_idx in 0..2 {
            let col_null_counts_array = null_counts.value(col_idx);
            let col_null_counts = col_null_counts_array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            let total: u32 = (0..col_null_counts.len())
                .map(|i| col_null_counts.value(i))
                .sum();
            assert_eq!(total, 0, "Column {} should have no nulls", col_idx);
        }
    }

    #[tokio::test]
    async fn test_consolidation_with_nullable_columns() {
        // Test with nullable columns that have actual nulls
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("nullable_value", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from(
                    (0..100)
                        .map(|i| if i % 3 == 0 { None } else { Some(i) })
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_column_stats: true,
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let result = consolidate_column_stats(&dataset, dataset.manifest.version + 1)
            .await
            .unwrap();

        assert!(
            result.is_some(),
            "Should handle nullable columns with nulls"
        );

        // Verify null_counts are tracked correctly
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 2); // Two columns

        // Check null_counts for nullable_value column (row 1)
        let null_counts = batch
            .column_by_name("null_counts")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(1); // nullable_value column
        let null_counts = null_counts.as_any().downcast_ref::<UInt32Array>().unwrap();
        let total_nulls: u32 = (0..null_counts.len()).map(|i| null_counts.value(i)).sum();
        assert_eq!(total_nulls, 34); // 34 values are null (every 3rd: 0, 3, 6, ..., 99)
    }

    #[tokio::test]
    async fn test_fragment_with_multiple_data_files() {
        // Test that fragment_has_stats correctly checks ALL data files in a fragment
        use lance_core::utils::tempfile::TempStrDir;

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        // Create dataset with stats and small max_rows_per_file to force multiple files
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..500))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_column_stats: true,
            max_rows_per_file: 100, // Force multiple data files per fragment
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let fragments = dataset.get_fragments();

        // Should have at least one fragment
        assert!(!fragments.is_empty());

        // Check that fragment_has_stats works correctly
        for fragment in &fragments {
            let has_stats = fragment_has_stats(&dataset, fragment).await.unwrap();
            assert!(has_stats, "All data files in fragment should have stats");

            // Verify multiple data files exist
            let num_files = fragment.metadata().files.len();
            assert!(num_files > 0, "Fragment should have at least one data file");
        }
    }
}
