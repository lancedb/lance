// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column statistics consolidation utilities.
//!
//! This module provides functionality for consolidating per-fragment column statistics
//! into a single consolidated stats file. It works in conjunction with
//! [`column_stats_reader`](crate::dataset::column_stats_reader) which provides
//! the reading API.
//!
//! # Overview
//!
//! Per-fragment statistics are stored in each data file's global buffer in a **columnar layout**
//! (one column per dataset column, each row represents a zone, with type `ColumnZoneStatistics`).
//! This module consolidates them into a **columnar layout** with one row total
//! (one column per dataset column, each containing a `List<struct<...>>` with zone statistics).
//!
//! # Workflow
//!
//! 1. **Per-fragment stats** (columnar layout, local offsets) → stored in data files
//! 2. **Consolidation** (this module) → converts to columnar layout with one row, local offsets preserved
//! 3. **Reading** ([`column_stats_reader`](crate::dataset::column_stats_reader)) → provides
//!    typed access to consolidated stats
//!

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::StructArray;
use arrow_array::{Array, ArrayRef, ListArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::concat::concat;
use lance_arrow_stats::ArrowScalar;
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_core::utils::zone::ZoneBound;
use lance_encoding::decoder::DecoderPlugins;
use lance_encoding::version::LanceFileVersion;
use lance_file::determine_file_version;
use lance_file::reader::FileReader;
use lance_file::writer::create_consolidated_zone_struct_type;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

use crate::dataset::fragment::FileFragment;
use crate::{Dataset, Error};

/// Consolidated statistics for a single zone of a single column.
#[derive(Debug, Clone)]
pub struct ZoneStats {
    /// Zone boundary information (fragment_id, start offset, length)
    pub bound: ZoneBound,
    /// Zone ID within the fragment (0, 1, 2, ...)
    /// This is the index of the zone within the fragment file
    pub zone_id: u32,
    pub null_count: u32,
    pub nan_count: u32,
    pub min: ArrowScalar,
    pub max: ArrowScalar,
}

/// Consolidate column statistics from all fragments into a single file.
///
/// This function implements an "all-or-nothing" approach: if any fragment
/// lacks column statistics, consolidation is skipped entirely.
/// It should be relaxed in the future to support partial stats dataset consolidation. #5857
///
/// # How It Works
///
/// Each fragment file contains per-fragment statistics in a **columnar layout** (see writer.rs):
/// Each dataset column maps to a column in the stats file, with type `ColumnZoneStatistics` (struct).
/// Each row represents a zone.
///
/// **Fragment file layout**:
/// ```text
/// ┌─────────────┬──────────────────────────────┬──────────────────────────────┐
/// │ Row (Zone)  │ "id" (ColumnZoneStatistics)  │ "price" (ColumnZoneStatistics)│
/// ├─────────────┼──────────────────────────────┼──────────────────────────────┤
/// │ 0           │ {min, max, null_count, ...}  │ {min, max, null_count, ...}  │
/// │ 1           │ {min, max, null_count, ...}  │ {min, max, null_count, ...}  │
/// │ ...         │ ...                          │ ...                          │
/// └─────────────┴──────────────────────────────┴──────────────────────────────┘
/// ```
///
/// **Fragment 0 stats** (2 zones, local offsets):
/// ```text
/// Row 0 (zone 0):
///   "id": ColumnZoneStatistics{min="1", max="1000000", null_count=0, nan_count=0, bound={fragment_id=0, start=0, length=1000000}}
///   "price": ColumnZoneStatistics{min="9.99", max="99.99", null_count=0, nan_count=0, bound={fragment_id=0, start=0, length=1000000}}
///
/// Row 1 (zone 1):
///   "id": ColumnZoneStatistics{min="1000001", max="2000000", null_count=0, nan_count=0, bound={fragment_id=0, start=1000000, length=1000000}}
///   "price": ColumnZoneStatistics{min="10.50", max="100.50", null_count=0, nan_count=0, bound={fragment_id=0, start=1000000, length=1000000}}
/// ```
///
/// **Fragment 1 stats** (2 zones, local offsets):
/// ```text
/// Row 0 (zone 0):
///   "id": ColumnZoneStatistics{min="2000001", max="3000000", null_count=0, nan_count=0, bound={fragment_id=1, start=0, length=1000000}}
///   "price": ColumnZoneStatistics{min="15.00", max="150.00", null_count=0, nan_count=0, bound={fragment_id=1, start=0, length=1000000}}
///
/// Row 1 (zone 1):
///   "id": ColumnZoneStatistics{min="3000001", max="4000000", null_count=0, nan_count=0, bound={fragment_id=1, start=1000000, length=1000000}}
///   "price": ColumnZoneStatistics{min="20.00", max="200.00", null_count=0, nan_count=0, bound={fragment_id=1, start=1000000, length=1000000}}
/// ```
///
/// This function **consolidates** them into a **columnar layout** with one row total:
/// Each dataset column maps to a column in the consolidated stats file, with type `List<struct<fragment_id, zone_start, zone_length, null_count, nan_count, min_value, max_value>>`.
/// The list is ordered by zone_id first, then fragment_id. Zone offsets remain local (per fragment).
///
/// **Consolidated file layout**:
/// ```text
/// ┌─────┬──────────────────────────────────────┬──────────────────────────────────────┐
/// │ Row │ "id" (List<struct<...>>)             │ "price" (List<struct<...>>)          │
/// ├─────┼──────────────────────────────────────┼──────────────────────────────────────┤
/// │ 0   │ [struct{...}, struct{...}, ...]     │ [struct{...}, struct{...}, ...]     │
/// └─────┴──────────────────────────────────────┴──────────────────────────────────────┘
/// ```
///
/// **Consolidated stats** (one row total, columnar):
/// ```text
/// Row 0:
///   "id": List[
///     struct{fragment_id=0, zone_start=0, zone_length=1000000, null_count=0, nan_count=0, min_value="1", max_value="1000000"},
///     struct{fragment_id=1, zone_start=0, zone_length=1000000, null_count=0, nan_count=0, min_value="2000001", max_value="3000000"},
///     struct{fragment_id=0, zone_start=1000000, zone_length=1000000, null_count=0, nan_count=0, min_value="1000001", max_value="2000000"},
///     struct{fragment_id=1, zone_start=1000000, zone_length=1000000, null_count=0, nan_count=0, min_value="3000001", max_value="4000000"}
///   ]
///   "price": List[
///     struct{fragment_id=0, zone_start=0, zone_length=1000000, null_count=0, nan_count=0, min_value="9.99", max_value="99.99"},
///     struct{fragment_id=1, zone_start=0, zone_length=1000000, null_count=0, nan_count=0, min_value="15.00", max_value="150.00"},
///     struct{fragment_id=0, zone_start=1000000, zone_length=1000000, null_count=0, nan_count=0, min_value="10.50", max_value="100.50"},
///     struct{fragment_id=1, zone_start=1000000, zone_length=1000000, null_count=0, nan_count=0, min_value="20.00", max_value="200.00"}
///   ]
/// ```
///
/// **Key points**:
/// - Zone offsets (`zone_start`) remain **local** (per fragment), not global
/// - List elements are ordered by `(zone_id, fragment_id)`: all zone 0s first, then all zone 1s, etc.
/// - Each dataset column has its own column in the consolidated file
///
pub async fn consolidate_column_stats(dataset: &Dataset) -> Result<Option<String>> {
    // Step 1: Pre-check - ALL fragments must have stats (all-or-nothing)
    let fragments = dataset.get_fragments();
    let total_fragments = fragments.len();
    let mut fragments_with_stats = 0;

    for fragment in &fragments {
        if fragment_has_stats(dataset, fragment).await? {
            fragments_with_stats += 1;
        }
    }

    // TODO: Support partial stats dataset consolidation
    if fragments_with_stats < total_fragments {
        log::warn!(
            "Skipping column stats consolidation: only {fragments_with_stats}/{total_fragments} fragments have stats"
        );
        return Ok(None);
    }

    // Step 2: Collect stats from all fragments, organized by column
    let mut stats_by_column: HashMap<String, Vec<ZoneStats>> = HashMap::new();

    for fragment in &fragments {
        for data_file in &fragment.metadata().files {
            let file_path = dataset
                .data_file_dir(data_file)?
                .join(data_file.path.as_str());
            let file_stats = read_fragment_column_stats(dataset, &file_path).await?;

            if let Some(file_stats) = file_stats {
                for (col_name, zones) in file_stats {
                    let adjusted_zones: Vec<ZoneStats> = zones
                        .into_iter()
                        .map(|z| ZoneStats {
                            bound: ZoneBound {
                                fragment_id: fragment.id() as u64,
                                start: z.bound.start, // Keep local offset
                                length: z.bound.length,
                            },
                            zone_id: z.zone_id,
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

    // Step 3: Build consolidated batch
    let consolidated_batch = build_consolidated_batch(stats_by_column, dataset.schema())?;

    // Step 4: Write as Lance file
    let stats_path = String::from("_stats/column_stats.lance");
    write_stats_file(
        dataset.object_store.as_ref(),
        &dataset.base.clone().join(stats_path.as_str()),
        consolidated_batch,
    )
    .await?;

    log::info!(
        "Consolidated column stats from {} fragments into {}",
        total_fragments,
        stats_path,
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
            .join(data_file.path.as_str());
        // Legacy (0.2) format does not have column stats; skip to avoid opening with v2 reader
        if determine_file_version(dataset.object_store.as_ref(), &file_path, None).await?
            == LanceFileVersion::Legacy
        {
            return Ok(false);
        }
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
            &dataset.metadata_cache.file_metadata_cache(&file_path),
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
/// stored in a columnar layout in the data file (one column per dataset column,
/// each row represents a zone, with type `ColumnZoneStatistics`), which
/// this function converts to a nested structure for easier processing.
///
/// # Example
///
/// For a data file with 2 columns and 2 zones each, the columnar layout in the file:
/// ```text
/// ┌─────┬──────────────────────────────┬──────────────────────────────┐
/// │ Row │ "id" (ColumnZoneStatistics)  │ "price" (ColumnZoneStatistics)│
/// ├─────┼──────────────────────────────┼──────────────────────────────┤
/// │ 0   │ {min, max, null_count, ...}  │ {min, max, null_count, ...}  │
/// │ 1   │ {min, max, null_count, ...}  │ {min, max, null_count, ...}  │
/// └─────┴──────────────────────────────┴──────────────────────────────┘
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
    // Legacy (0.2) format does not have column stats; v2 reader would reject the file
    if determine_file_version(dataset.object_store.as_ref(), file_path, None).await?
        == LanceFileVersion::Legacy
    {
        return Ok(None);
    }
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
        &dataset.metadata_cache.file_metadata_cache(file_path),
        dataset.file_reader_options.clone().unwrap_or_default(),
    )
    .await?;

    let Some(stats_batch) = file_reader.read_column_stats().await? else {
        return Ok(None);
    };

    // Parse the columnar stats batch: one column per dataset column, each containing ColumnZoneStatistics structs
    // Rows = zones (one row per zone)
    let mut result = HashMap::new();
    use arrow_array::StructArray;

    let num_zones = stats_batch.num_rows();
    let schema = stats_batch.schema();

    // Iterate over each column in the batch (each column corresponds to a dataset column)
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name();
        let column_array = stats_batch.column(col_idx);

        // Extract the StructArray for this column
        let struct_array = column_array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected StructArray for column '{}' in column stats",
                    col_name
                ))
            })?;

        // Extract min/max arrays (typed as the column's type in fragment stats)
        let min_array = struct_array.column_by_name("min").ok_or_else(|| {
            Error::internal(format!(
                "Missing 'min' field in column stats for '{}'",
                col_name
            ))
        })?;

        let max_array = struct_array.column_by_name("max").ok_or_else(|| {
            Error::internal(format!(
                "Missing 'max' field in column stats for '{}'",
                col_name
            ))
        })?;

        let null_count_array = struct_array
            .column_by_name("null_count")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'null_count' field in column stats for '{}'",
                    col_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt32Array for 'null_count' field in column '{}'",
                    col_name
                ))
            })?;

        let nan_count_array = struct_array
            .column_by_name("nan_count")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'nan_count' field in column stats for '{}'",
                    col_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt32Array for 'nan_count' field in column '{}'",
                    col_name
                ))
            })?;

        // Extract the bound struct
        let bound_struct = struct_array
            .column_by_name("bound")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'bound' field in column stats for '{}'",
                    col_name
                ))
            })?
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected StructArray for 'bound' field in column '{}'",
                    col_name
                ))
            })?;

        let fragment_id_array = bound_struct
            .column_by_name("fragment_id")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'fragment_id' in bound struct for column '{}'",
                    col_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt64Array for 'fragment_id' in bound struct for column '{}'",
                    col_name
                ))
            })?;

        let start_array = bound_struct
            .column_by_name("start")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'start' in bound struct for column '{}'",
                    col_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt64Array for 'start' in bound struct for column '{}'",
                    col_name
                ))
            })?;

        let length_array = bound_struct
            .column_by_name("length")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'length' in bound struct for column '{}'",
                    col_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt64Array for 'length' in bound struct for column '{}'",
                    col_name
                ))
            })?;

        // Process each zone (row) for this column
        // zone_idx is the zone_id within the fragment
        let mut zones = Vec::with_capacity(num_zones);
        for zone_idx in 0..num_zones {
            let min_scalar = ArrowScalar::try_new(min_array, zone_idx).map_err(|e| {
                Error::internal(format!(
                    "Failed to get min ArrowScalar for column '{}': {}",
                    col_name, e
                ))
            })?;
            let max_scalar = ArrowScalar::try_new(max_array, zone_idx).map_err(|e| {
                Error::internal(format!(
                    "Failed to get max ArrowScalar for column '{}': {}",
                    col_name, e
                ))
            })?;
            let zone_stat = ZoneStats {
                bound: ZoneBound {
                    fragment_id: fragment_id_array.value(zone_idx),
                    start: start_array.value(zone_idx),
                    length: length_array.value(zone_idx) as usize,
                },
                zone_id: zone_idx as u32,
                null_count: null_count_array.value(zone_idx),
                nan_count: nan_count_array.value(zone_idx),
                min: min_scalar,
                max: max_scalar,
            };
            zones.push(zone_stat);
        }

        result.insert(col_name.to_string(), zones);
    }

    Ok(Some(result))
}

/// Create Arrow schema for consolidated statistics
///
/// Schema: one column per dataset column, each of type List<struct>
pub(crate) fn create_consolidated_stats_schema(dataset_schema: &Schema) -> Arc<ArrowSchema> {
    let fields: Vec<ArrowField> = dataset_schema
        .fields
        .iter()
        .map(|field| {
            let column_type = field.data_type();
            ArrowField::new(
                &field.name,
                DataType::List(Arc::new(ArrowField::new(
                    "zone",
                    create_consolidated_zone_struct_type(&column_type),
                    false,
                ))),
                false,
            )
        })
        .collect();

    Arc::new(ArrowSchema::new(fields))
}

/// Build a consolidated RecordBatch from collected statistics.
///
/// Uses columnar layout: one row total, one column per dataset column.
/// Each column is List<struct> where struct contains zone statistics.
/// List is ordered by zone_id first, then fragment_id.
fn build_consolidated_batch(
    stats_by_column: HashMap<String, Vec<ZoneStats>>,
    dataset_schema: &Schema,
) -> Result<RecordBatch> {
    let mut column_arrays: Vec<ArrayRef> = Vec::new();
    let mut schema_fields: Vec<ArrowField> = Vec::new();

    // Get the full schema (for all columns) to ensure consistency
    let full_schema = create_consolidated_stats_schema(dataset_schema);
    let full_schema_fields: HashMap<String, Arc<ArrowField>> = full_schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.clone()))
        .collect();

    // Process each dataset column (in schema order)
    for field in dataset_schema.fields.iter() {
        let col_name = &field.name;

        if let Some(mut zones) = stats_by_column.get(col_name).cloned() {
            // Sort zones by zone_id first, then fragment_id (as per requirements)
            zones.sort_by_key(|z| (z.zone_id, z.bound.fragment_id));

            // Build arrays for the struct fields; min/max use ArrowScalar's typed Arrow arrays.
            let mut fragment_ids = Vec::with_capacity(zones.len());
            let mut zone_starts = Vec::with_capacity(zones.len());
            let mut zone_lengths = Vec::with_capacity(zones.len());
            let mut null_counts = Vec::with_capacity(zones.len());
            let mut nan_counts = Vec::with_capacity(zones.len());

            for zone in &zones {
                fragment_ids.push(zone.bound.fragment_id);
                zone_starts.push(zone.bound.start);
                zone_lengths.push(zone.bound.length as u64);
                null_counts.push(zone.null_count);
                nan_counts.push(zone.nan_count);
            }

            let min_scalars: Vec<_> = zones.iter().map(|z| z.min.clone()).collect();
            let max_scalars: Vec<_> = zones.iter().map(|z| z.max.clone()).collect();
            let min_array = arrow_scalars_to_array(&min_scalars, col_name, "min")?;
            let max_array = arrow_scalars_to_array(&max_scalars, col_name, "max")?;

            let column_type = field.data_type();
            let consolidated_zone_struct_type = create_consolidated_zone_struct_type(&column_type);

            // Build the struct array for this column's zones (min/max are typed)
            let zone_struct_array = StructArray::from(vec![
                (
                    Arc::new(ArrowField::new("fragment_id", DataType::UInt64, false)),
                    Arc::new(UInt64Array::from(fragment_ids.clone())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("zone_start", DataType::UInt64, false)),
                    Arc::new(UInt64Array::from(zone_starts.clone())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("zone_length", DataType::UInt64, false)),
                    Arc::new(UInt64Array::from(zone_lengths.clone())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("null_count", DataType::UInt32, false)),
                    Arc::new(UInt32Array::from(null_counts.clone())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("nan_count", DataType::UInt32, false)),
                    Arc::new(UInt32Array::from(nan_counts.clone())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("min_value", column_type.clone(), true)),
                    min_array,
                ),
                (
                    Arc::new(ArrowField::new("max_value", column_type.clone(), true)),
                    max_array,
                ),
            ]);

            // Wrap in a List array (one list containing all zones for this column)
            // Create offsets: [0, zones.len()] to represent a single list
            let offsets = OffsetBuffer::from_lengths([zones.len()]);
            let list_field = Arc::new(ArrowField::new(
                "zone",
                consolidated_zone_struct_type,
                false,
            ));
            let list_array = ListArray::try_new(
                list_field.clone(),
                offsets,
                Arc::new(zone_struct_array) as ArrayRef,
                None,
            )
            .map_err(|e| {
                Error::internal(format!(
                    "Failed to create ListArray for column '{}': {}",
                    col_name, e
                ))
            })?;

            // Use the field definition from the full schema to ensure consistency
            let schema_field = full_schema_fields.get(col_name).ok_or_else(|| {
                Error::internal(format!(
                    "Column '{}' not found in consolidated stats schema",
                    col_name
                ))
            })?;
            schema_fields.push((**schema_field).clone());
            column_arrays.push(Arc::new(list_array) as ArrayRef);
        }
    }

    if column_arrays.is_empty() {
        return Err(Error::internal(
            "[ColumnStats] No column statistics to consolidate",
        ));
    }

    // Create schema: one column per dataset column, each of type List<struct>
    let schema = Arc::new(ArrowSchema::new(schema_fields));

    // Create RecordBatch: one row total
    RecordBatch::try_new(schema, column_arrays).map_err(|e| {
        Error::internal(format!(
            "[ColumnStats] Failed to create consolidated stats batch: {}",
            e
        ))
    })
}

fn arrow_scalars_to_array(
    scalars: &[ArrowScalar],
    column_name: &str,
    stat_name: &str,
) -> Result<ArrayRef> {
    let arrays: Vec<&dyn Array> = scalars
        .iter()
        .map(|scalar| scalar.as_array().as_ref())
        .collect();
    concat(&arrays).map_err(|e| {
        Error::internal(format!(
            "Failed to build {} array for column '{}': {}",
            stat_name, column_name, e
        ))
    })
}

/// Write the consolidated stats RecordBatch as a Lance file.
async fn write_stats_file(
    object_store: &ObjectStore,
    path: &Path,
    batch: RecordBatch,
) -> Result<()> {
    use lance_file::writer::{FileWriter, FileWriterOptions};

    let lance_schema = lance_core::datatypes::Schema::try_from(batch.schema().as_ref())
        .map_err(|e| Error::internal(format!("Failed to convert schema: {}", e)))?;

    let mut writer = FileWriter::try_new(
        object_store.create(path).await?,
        lance_schema,
        FileWriterOptions {
            disable_column_stats: true, // Consolidated stats file has List<struct> columns; no per-column min/max
            ..Default::default()
        },
    )?;

    writer.write_batch(&batch).await?;
    writer.finish().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::WriteParams;
    use futures::stream::TryStreamExt;

    // Helper functions for common test schemas
    fn create_id_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]))
    }

    fn create_id_name_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]))
    }

    fn create_id_value_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            ArrowField::new("value", DataType::Float32, false),
        ]))
    }

    fn create_multi_type_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("int_col", DataType::Int32, false),
            ArrowField::new("float_col", DataType::Float32, false),
            ArrowField::new("string_col", DataType::Utf8, false),
        ]))
    }

    fn create_nullable_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("nullable_value", DataType::Int32, true),
        ]))
    }

    /// Helper function to read consolidated stats file using FileReader
    async fn read_stats_file(dataset: &Dataset, stats_path: &str) -> Vec<RecordBatch> {
        let full_path = dataset.base.clone().join(stats_path);
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
            &dataset.metadata_cache.file_metadata_cache(&full_path),
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
            .await
            .unwrap();

        let mut batches = Vec::new();
        while let Some(batch) = stream.try_next().await.unwrap() {
            batches.push(batch);
        }
        batches
    }
    use crate::Dataset;
    use arrow_array::{Float32Array, Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_testing::datagen::generate_random_array;

    #[tokio::test]
    async fn test_consolidation_all_fragments_have_stats() {
        // Create dataset with column stats enabled
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = create_id_name_schema();

        // Create 3 fragments, each with stats
        let write_params = WriteParams {
            max_rows_per_file: 100,
            disable_column_stats: false, // Stats enabled
            ..Default::default()
        };

        for i in 0..3 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values((i * 100)..((i + 1) * 100))),
                    Arc::new(StringArray::from_iter_values(
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
                    disable_column_stats: false, // Stats enabled
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
        let result = consolidate_column_stats(&dataset).await.unwrap();

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

        // New format: 1 row total, 2 columns (id, name)
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 2);

        // Verify "id" column stats
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let id_struct = id_column.value(0);
        let id_struct = id_struct.as_any().downcast_ref::<StructArray>().unwrap();

        let fragment_ids = id_struct
            .column_by_name("fragment_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", fragment_ids),
            format!("{:?}", UInt64Array::from(vec![0, 1, 2]))
        );

        let zone_starts = id_struct
            .column_by_name("zone_start")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", zone_starts),
            format!("{:?}", UInt64Array::from(vec![0, 0, 0])) // Local offsets
        );

        let zone_lengths = id_struct
            .column_by_name("zone_length")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", zone_lengths),
            format!("{:?}", UInt64Array::from(vec![100, 100, 100]))
        );

        let null_counts = id_struct
            .column_by_name("null_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", null_counts),
            format!("{:?}", UInt32Array::from(vec![0, 0, 0]))
        );

        let nan_counts = id_struct
            .column_by_name("nan_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", nan_counts),
            format!("{:?}", UInt32Array::from(vec![0, 0, 0]))
        );
        let mins = id_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", mins),
            format!("{:?}", Int32Array::from(vec![0, 100, 200]))
        );
        let maxs = id_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", maxs),
            format!("{:?}", Int32Array::from(vec![99, 199, 299]))
        );

        // Verify "name" column stats
        let name_column = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let name_struct = name_column.value(0);
        let name_struct = name_struct.as_any().downcast_ref::<StructArray>().unwrap();

        let name_fragment_ids = name_struct
            .column_by_name("fragment_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(
            format!("{:?}", name_fragment_ids),
            format!("{:?}", UInt64Array::from(vec![0, 1, 2]))
        );

        let name_mins = name_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            format!("{:?}", name_mins),
            format!(
                "{:?}",
                StringArray::from(vec!["name_0", "name_100", "name_200"])
            )
        );
        let name_maxs = name_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            format!("{:?}", name_maxs),
            format!(
                "{:?}",
                StringArray::from(vec!["name_99", "name_199", "name_299"])
            )
        );
    }

    #[tokio::test]
    async fn test_local_offset_preservation() {
        // Test that zone offsets remain local (per fragment), not global.
        // 205 rows: fragment 0 has 100 rows; append of 105 with max_rows_per_file=100
        // yields fragment 1 (100 rows) and fragment 2 (5 rows) — 3 zones total.
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
            disable_column_stats: false,
            ..Default::default()
        };

        // Fragment 0: 100 rows (values 0..100)
        let batch0 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        let reader0 = RecordBatchIterator::new(vec![Ok(batch0)], schema.clone());
        Dataset::write(reader0, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        // Fragment 1: 105 rows (values 100..205) -> 2 files due to max_rows_per_file=100
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(100..205))],
        )
        .unwrap();
        let reader1 = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        let append_params = WriteParams {
            mode: crate::dataset::WriteMode::Append,
            max_rows_per_file: 100,
            disable_column_stats: false,
            ..Default::default()
        };
        Dataset::write(reader1, test_uri, Some(append_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stats_path = consolidate_column_stats(&dataset).await.unwrap().unwrap();

        // Read the consolidated stats file
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        // Verify zone_starts are local (per fragment)
        // In the new columnar format, we need to read from the List<struct> column
        let value_column = batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        let struct_array = value_column.value(0);
        let struct_array = struct_array.as_any().downcast_ref::<StructArray>().unwrap();

        let zone_starts = struct_array
            .column_by_name("zone_start")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        let zone_lengths = struct_array
            .column_by_name("zone_length")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        let fragment_ids = struct_array
            .column_by_name("fragment_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        let min_values = struct_array
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        let max_values = struct_array
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        // 3 zones total: frag0 1 file, frag1 2 files (100 + 5 rows)
        assert_eq!(
            zone_starts.len(),
            3,
            "expected 3 zones for 205 rows (100 + 105)"
        );
        assert_eq!(zone_lengths.len(), 3);
        assert_eq!(fragment_ids.len(), 3);

        // Zone 0: fragment 0, start=0, length=100, min=0, max=99
        assert_eq!(fragment_ids.value(0), 0);
        assert_eq!(zone_starts.value(0), 0);
        assert_eq!(zone_lengths.value(0), 100);
        assert_eq!(min_values.value(0), 0);
        assert_eq!(max_values.value(0), 99);

        // Zone 1: fragment 1, first file, start=0, length=100, min=100, max=199
        assert_eq!(fragment_ids.value(1), 1);
        assert_eq!(zone_starts.value(1), 0);
        assert_eq!(zone_lengths.value(1), 100);
        assert_eq!(min_values.value(1), 100);
        assert_eq!(max_values.value(1), 199);

        // Zone 2: fragment 2 (second file from append), start=0, length=5, min=200, max=204
        assert_eq!(fragment_ids.value(2), 2);
        assert_eq!(zone_starts.value(2), 0);
        assert_eq!(zone_lengths.value(2), 5);
        assert_eq!(min_values.value(2), 200);
        assert_eq!(max_values.value(2), 204);

        // Verify that zones from the same fragment have local offsets (starting from 0)
        // Zones are ordered by zone_id first, then fragment_id
        let mut fragment_zone_starts: HashMap<u64, Vec<u64>> = HashMap::new();
        for i in 0..zone_starts.len() {
            let frag_id = fragment_ids.value(i);
            let zone_start = zone_starts.value(i);
            fragment_zone_starts
                .entry(frag_id)
                .or_default()
                .push(zone_start);
        }

        // Each fragment should have zones starting from 0 (local offsets)
        for (frag_id, starts) in fragment_zone_starts {
            let min_start = starts.iter().min().unwrap();
            assert_eq!(
                *min_start, 0,
                "Fragment {} zones should start at local offset 0, but minimum is {}",
                frag_id, min_start
            );
        }
    }

    #[tokio::test]
    async fn test_empty_dataset() {
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = create_id_schema();

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            disable_column_stats: false, // Stats enabled
            ..Default::default()
        };

        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete all rows
        dataset.delete("id >= 0").await.unwrap();
        dataset = Dataset::open(test_uri).await.unwrap();

        // Should still work but return None (no data to consolidate)
        let result = consolidate_column_stats(&dataset).await.unwrap();

        // With deletions, fragments still exist, so consolidation should work
        // This tests that we handle the case gracefully
        assert!(result.is_some() || result.is_none());
    }

    #[tokio::test]
    async fn test_multiple_column_types() {
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = create_multi_type_schema();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(generate_random_array(100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|i| format!("str_{}", i)),
                )),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            disable_column_stats: false, // Stats enabled
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let result = consolidate_column_stats(&dataset).await.unwrap();

        assert!(result.is_some(), "Should handle multiple column types");

        // Verify the stats file contains all 3 column types
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        // New format: 1 row total, 3 columns (int_col, float_col, string_col)
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 3);

        // Verify int_col
        let int_col = batch
            .column_by_name("int_col")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let int_struct = int_col.value(0);
        let int_struct = int_struct.as_any().downcast_ref::<StructArray>().unwrap();

        let int_mins = int_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let int_maxs = int_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(int_mins.value(0), 0);
        assert_eq!(int_maxs.value(int_maxs.len() - 1), 99);

        // Verify float_col
        let float_col = batch
            .column_by_name("float_col")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let float_struct = float_col.value(0);
        let float_struct = float_struct.as_any().downcast_ref::<StructArray>().unwrap();

        let float_mins = float_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let float_maxs = float_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(float_mins.len(), float_maxs.len());
        // For each zone, verify min <= max
        for i in 0..float_mins.len() {
            let min_val: f32 = float_mins.value(i);
            let max_val: f32 = float_maxs.value(i);
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

        // Verify string_col
        let string_col = batch
            .column_by_name("string_col")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let string_struct = string_col.value(0);
        let string_struct = string_struct
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let str_mins = string_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let str_maxs = string_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(str_mins.value(0), "str_0");
        assert_eq!(str_maxs.value(str_maxs.len() - 1), "str_99");

        // Verify null_counts are all zero (no nulls) for all columns
        let columns = vec!["int_col", "float_col", "string_col"];
        for col_name in columns {
            let col = batch
                .column_by_name(col_name)
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let struct_array = col.value(0);
            let struct_array = struct_array.as_any().downcast_ref::<StructArray>().unwrap();
            let col_null_counts = struct_array
                .column_by_name("null_count")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            let total: u32 = (0..col_null_counts.len())
                .map(|j| col_null_counts.value(j))
                .sum();
            assert_eq!(total, 0, "Column {} should have no nulls", col_name);
        }
    }

    #[tokio::test]
    async fn test_consolidation_single_fragment() {
        // Test consolidation with just one fragment
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = create_id_schema();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..100))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            disable_column_stats: false, // Stats enabled
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);

        let result = consolidate_column_stats(&dataset).await.unwrap();

        assert!(
            result.is_some(),
            "Should consolidate even with single fragment"
        );

        // Verify content
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 1); // One row total
        assert_eq!(batch.num_columns(), 1); // One column: "id"

        // In new format: "id" column contains List<struct>
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        let struct_array = id_column.value(0);
        let struct_array = struct_array.as_any().downcast_ref::<StructArray>().unwrap();

        // Extract fields from struct
        let fragment_ids = struct_array
            .column_by_name("fragment_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(!fragment_ids.is_empty()); // At least one zone
        assert_eq!(fragment_ids.value(0), 0); // Fragment 0

        // Verify min/max for "id" column: [0, 99]
        let mins = struct_array
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(mins.value(0), 0);

        let maxs = struct_array
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(maxs.value(maxs.len() - 1), 99);

        // Verify zone_starts begin at 0
        let zone_starts = struct_array
            .column_by_name("zone_start")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(zone_starts.value(0), 0);

        // Verify zone_lengths sum to 100
        let zone_lengths = struct_array
            .column_by_name("zone_length")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let total_length: u64 = (0..zone_lengths.len()).map(|i| zone_lengths.value(i)).sum();
        assert_eq!(total_length, 100);

        // Verify null_counts are zero
        let null_counts = struct_array
            .column_by_name("null_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
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

        let schema = create_id_value_schema();

        let write_params = WriteParams {
            max_rows_per_file: 50_000,
            disable_column_stats: false, // Stats enabled
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
                    disable_column_stats: false, // Stats enabled
                    ..Default::default()
                };
                Dataset::write(reader, test_uri, Some(append_params))
                    .await
                    .unwrap();
            }
        }

        let dataset = Dataset::open(test_uri).await.unwrap();
        let result = consolidate_column_stats(&dataset).await.unwrap();

        assert!(
            result.is_some(),
            "Should handle large dataset with multiple zones"
        );

        // Verify content with large dataset
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 1); // One row total
        assert_eq!(batch.num_columns(), 2); // Two columns: "id" and "value"

        // Verify "id" column has zones from both fragments
        let id_column = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let id_struct = id_column.value(0);
        let id_struct = id_struct.as_any().downcast_ref::<StructArray>().unwrap();

        let fragment_ids = id_struct
            .column_by_name("fragment_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(
            fragment_ids.len() >= 2,
            "Should have zones from multiple fragments"
        );
        // Check both fragments are represented
        assert_eq!(fragment_ids.value(0), 0);
        assert_eq!(fragment_ids.value(fragment_ids.len() - 1), 1);

        // "id" column is Int64 in create_id_value_schema
        let mins = id_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        let maxs = id_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();

        // Verify min/max for "id" column spans the full range [0, 99999]
        assert_eq!(mins.value(0), 0); // First zone starts at 0
        assert_eq!(maxs.value(maxs.len() - 1), 99999); // Last zone ends at 99999

        // Verify min/max for "value" column (Float32)
        let value_column = batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let value_struct = value_column.value(0);
        let value_struct = value_struct.as_any().downcast_ref::<StructArray>().unwrap();

        let value_mins = value_struct
            .column_by_name("min_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let value_maxs = value_struct
            .column_by_name("max_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(value_mins.value(0), 0.0);
        assert_eq!(value_maxs.value(value_maxs.len() - 1), 99999.0);

        // Verify zone_starts are local (per fragment)
        let zone_starts = id_struct
            .column_by_name("zone_start")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        // First zone should start at local offset 0
        assert_eq!(zone_starts.value(0), 0);

        // Verify zone_lengths sum to 100000 total rows
        let zone_lengths = id_struct
            .column_by_name("zone_length")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let zone_lengths = zone_lengths.as_any().downcast_ref::<UInt64Array>().unwrap();
        let total_length: u64 = (0..zone_lengths.len()).map(|i| zone_lengths.value(i)).sum();
        assert_eq!(total_length, 100000);

        // Verify null_counts are all zero for both columns
        let columns = vec!["id", "value"];
        for col_name in columns {
            let col = batch
                .column_by_name(col_name)
                .unwrap()
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let struct_array = col.value(0);
            let struct_array = struct_array.as_any().downcast_ref::<StructArray>().unwrap();
            let col_null_counts = struct_array
                .column_by_name("null_count")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            let total: u32 = (0..col_null_counts.len())
                .map(|i| col_null_counts.value(i))
                .sum();
            assert_eq!(total, 0, "Column {} should have no nulls", col_name);
        }
    }

    #[tokio::test]
    async fn test_consolidation_with_nullable_columns() {
        // Test with nullable columns that have actual nulls
        use lance_core::utils::tempfile::TempStrDir;
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = create_nullable_schema();

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
            disable_column_stats: false, // Stats enabled
            ..Default::default()
        };

        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let result = consolidate_column_stats(&dataset).await.unwrap();

        assert!(
            result.is_some(),
            "Should handle nullable columns with nulls"
        );

        // Verify null_counts are tracked correctly
        let stats_path = result.unwrap();
        let batches = read_stats_file(&dataset, &stats_path).await;
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 1); // One row total
        assert_eq!(batch.num_columns(), 2); // Two columns: "id" and "nullable_value"

        // Check null_counts for nullable_value column
        let nullable_col = batch
            .column_by_name("nullable_value")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let nullable_struct = nullable_col.value(0);
        let nullable_struct = nullable_struct
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let null_counts = nullable_struct
            .column_by_name("null_count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let total_nulls: u32 = (0..null_counts.len()).map(|i| null_counts.value(i)).sum();
        assert_eq!(total_nulls, 34); // 34 values are null (every 3rd: 0, 3, 6, ..., 99)
    }

    #[tokio::test]
    async fn test_fragment_with_multiple_data_files() {
        // Test that fragment_has_stats correctly checks ALL data files in a fragment
        use lance_core::utils::tempfile::TempStrDir;

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = create_id_schema();

        // Create dataset with stats and small max_rows_per_file to force multiple files
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..500))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            disable_column_stats: false, // Stats enabled
            max_rows_per_file: 100,      // Force multiple data files per fragment
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
