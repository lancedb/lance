// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::datatypes::DataType;
use arrow_array::{cast::AsArray, Array, ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_buffer::Buffer;
use futures::StreamExt;
use lance_arrow::DataTypeExt;
use lance_core::datatypes::Schema;
use lance_linalg::distance::DistanceType;
use log::{info, warn};
use rand::rngs::SmallRng;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::SeedableRng;
use snafu::location;
use tokio::sync::Mutex;

use crate::dataset::Dataset;
use crate::{Error, Result};

/// Helper function to extract a column from a RecordBatch, supporting nested field paths.
///
/// This function handles:
/// - Simple column names: "column"
/// - Nested paths: "parent.child" or "parent.child.grandchild"
/// - Backtick-escaped field names: "parent.`field.with.dots`"
fn get_column_from_batch(batch: &RecordBatch, column: &str) -> Result<ArrayRef> {
    // Try to get the column directly first (fast path for simple columns)
    if let Some(col) = batch.column_by_name(column) {
        return Ok(col.clone());
    }

    // Parse the field path using Lance's field path parsing logic
    // This properly handles backtick-escaped field names
    let parts = lance_core::datatypes::parse_field_path(column).map_err(|e| Error::Index {
        message: format!("Failed to parse field path '{}': {}", column, e),
        location: location!(),
    })?;

    if parts.is_empty() {
        return Err(Error::Index {
            message: format!("Invalid empty field path: {}", column),
            location: location!(),
        });
    }

    // Get the root column
    let mut current_array: ArrayRef = batch
        .column_by_name(&parts[0])
        .ok_or_else(|| Error::Index {
            message: format!(
                "Column '{}' does not exist in batch (looking for root field '{}')",
                column, parts[0]
            ),
            location: location!(),
        })?
        .clone();

    // Navigate through nested struct fields
    for part in &parts[1..] {
        let struct_array = current_array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| Error::Index {
                message: format!(
                    "Cannot access nested field '{}' in column '{}': parent is not a struct",
                    part, column
                ),
                location: location!(),
            })?;

        current_array = struct_array
            .column_by_name(part)
            .ok_or_else(|| Error::Index {
                message: format!(
                    "Nested field '{}' does not exist in column '{}'",
                    part, column
                ),
                location: location!(),
            })?
            .clone();
    }

    Ok(current_array)
}

async fn estimate_multivector_vectors_per_row(
    dataset: &Dataset,
    column: &str,
    num_rows: usize,
) -> Result<usize> {
    if num_rows == 0 {
        return Ok(1030);
    }

    let projection = dataset.schema().project(&[column])?;

    // Try a few random samples first (fast path).
    let sample_batch_size = std::cmp::min(64, num_rows);
    for _ in 0..8 {
        let batch = dataset.sample(sample_batch_size, &projection).await?;
        let array = get_column_from_batch(&batch, column)?;
        let list_array = array.as_list::<i32>();
        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                continue;
            }
            let len = list_array.value_length(i) as usize;
            if len > 0 {
                return Ok(len);
            }
        }
    }

    // Fallback: scan a small prefix to find a non-null example. This avoids rare
    // flakiness when values are extremely sparse.
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    let column_expr = lance_datafusion::logical_expr::field_path_to_expr(column)?;
    scanner.filter_expr(column_expr.is_not_null());
    scanner.limit(Some(std::cmp::min(num_rows, 1024) as i64), None)?;
    let batch = scanner.try_into_batch().await?;
    let array = get_column_from_batch(&batch, column)?;
    let list_array = array.as_list::<i32>();
    for i in 0..list_array.len() {
        let len = list_array.value_length(i) as usize;
        if len > 0 {
            return Ok(len);
        }
    }

    warn!(
        "Could not find a non-empty multivector value for column {}, falling back to n=1030",
        column
    );
    Ok(1030)
}

/// Get the vector dimension of the given column in the schema.
pub fn get_vector_dim(schema: &Schema, column: &str) -> Result<usize> {
    let field = schema.field(column).ok_or(Error::Index {
        message: format!("Column {} does not exist in schema {}", column, schema),
        location: location!(),
    })?;
    infer_vector_dim(&field.data_type())
}

/// Infer the vector dimension from the given data type.
pub fn infer_vector_dim(data_type: &arrow::datatypes::DataType) -> Result<usize> {
    infer_vector_dim_impl(data_type, false)
}

fn infer_vector_dim_impl(data_type: &arrow::datatypes::DataType, in_list: bool) -> Result<usize> {
    match (data_type,in_list) {
        (arrow::datatypes::DataType::FixedSizeList(_, dim),_) => Ok(*dim as usize),
        (arrow::datatypes::DataType::List(inner), false) => infer_vector_dim_impl(inner.data_type(),true),
        _ => Err(Error::invalid_input(format!("Data type is not a vector (FixedSizeListArray or List<FixedSizeListArray>), but {:?}", data_type), location!()))
    }
}

/// Checks whether the given column is with a valid vector type
/// returns the vector type (FixedSizeList for vectors, or List for multivectors),
/// and element type (Float16/Float32/Float64 or UInt8 for binary vectors).
pub fn get_vector_type(
    schema: &Schema,
    column: &str,
) -> Result<(arrow_schema::DataType, arrow_schema::DataType)> {
    let field = schema.field(column).ok_or(Error::Index {
        message: format!("column {} does not exist in schema {}", column, schema),
        location: location!(),
    })?;
    Ok((
        field.data_type(),
        infer_vector_element_type(&field.data_type())?,
    ))
}

/// Returns the default distance type for the given vector element type.
pub fn default_distance_type_for(element_type: &arrow_schema::DataType) -> DistanceType {
    match element_type {
        arrow_schema::DataType::UInt8 => DistanceType::Hamming,
        _ => DistanceType::L2,
    }
}

/// Validate that the distance type is supported by the vector element type.
pub fn validate_distance_type_for(
    distance_type: DistanceType,
    element_type: &arrow_schema::DataType,
) -> Result<()> {
    let supported = match element_type {
        arrow_schema::DataType::UInt8 => matches!(distance_type, DistanceType::Hamming),
        arrow_schema::DataType::Int8
        | arrow_schema::DataType::Float16
        | arrow_schema::DataType::Float32
        | arrow_schema::DataType::Float64 => {
            matches!(
                distance_type,
                DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
            )
        }
        _ => false,
    };

    if supported {
        Ok(())
    } else {
        Err(Error::invalid_input(
            format!(
                "Distance type {} does not support {} vectors",
                distance_type, element_type
            ),
            location!(),
        ))
    }
}

/// If the data type is a fixed size list or list of fixed size list return the inner element type
/// and verify it is a type we can create a vector index on.
///
/// Return an error if the data type is any other type
pub fn infer_vector_element_type(
    data_type: &arrow::datatypes::DataType,
) -> Result<arrow_schema::DataType> {
    infer_vector_element_type_impl(data_type, false)
}

fn infer_vector_element_type_impl(
    data_type: &arrow::datatypes::DataType,
    in_list: bool,
) -> Result<arrow_schema::DataType> {
    match (data_type, in_list) {
        (arrow::datatypes::DataType::FixedSizeList(element_field, _), _) => {
            match element_field.data_type() {
                arrow::datatypes::DataType::Float16
                | arrow::datatypes::DataType::Float32
                | arrow::datatypes::DataType::Float64
                | arrow::datatypes::DataType::UInt8
                | arrow::datatypes::DataType::Int8 => Ok(element_field.data_type().clone()),
                _ => Err(Error::Index {
                    message: format!(
                        "vector element is not expected type (Float16/Float32/Float64 or UInt8): {:?}",
                        element_field.data_type()
                    ),
                    location: location!(),
                }),
            }
        }
        (arrow::datatypes::DataType::List(inner), false) => {
            infer_vector_element_type_impl(inner.data_type(), true)
        }
        _ => Err(Error::invalid_input(
            format!(
            "Data type is not a vector (FixedSizeListArray or List<FixedSizeListArray>), but {:?}",
            data_type
        ),
            location!(),
        )),
    }
}

/// Maybe sample training data from dataset, specified by column name.
///
/// Returns a [FixedSizeListArray], containing the training dataset.
///
pub async fn maybe_sample_training_data(
    dataset: &Dataset,
    column: &str,
    sample_size_hint: usize,
) -> Result<FixedSizeListArray> {
    let num_rows = dataset.count_rows(None).await?;

    let vector_field = dataset.schema().field(column).ok_or(Error::Index {
        message: format!(
            "Sample training data: column {} does not exist in schema",
            column
        ),
        location: location!(),
    })?;
    let is_nullable = vector_field.nullable;

    let sample_size_hint = match vector_field.data_type() {
        arrow::datatypes::DataType::List(_) => {
            // for multivector, we need `sample_size_hint` vectors for training,
            // but each multivector is a list of vectors, but we don't know how many
            // vectors are in each multivector. Estimate this by looking at a non-null row.
            // Set a minimum sample size of 128 to avoid too small samples,
            // it's not a problem because 128 multivectors is just about 64 MiB
            let vectors_per_row =
                estimate_multivector_vectors_per_row(dataset, column, num_rows).await?;
            sample_size_hint.div_ceil(vectors_per_row).max(128)
        }
        _ => sample_size_hint,
    };

    let should_sample = num_rows > sample_size_hint;
    if should_sample && !is_nullable {
        let projection = dataset.schema().project(&[column])?;
        let batch = dataset.sample(sample_size_hint, &projection).await?;
        info!(
            "Sample training data: retrieved {} rows by sampling",
            batch.num_rows()
        );
        vector_column_to_fsl(&batch, column)
    } else if should_sample && is_nullable {
        sample_nullable_training_data(dataset, column, sample_size_hint, num_rows, vector_field)
            .await
    } else {
        // too small to require sampling
        let batch = scan_all_training_data(dataset, column, is_nullable).await?;
        vector_column_to_fsl(&batch, column)
    }
}

#[derive(Debug)]
pub struct PartitionLoadLock {
    partition_locks: Vec<Arc<Mutex<()>>>,
}

impl PartitionLoadLock {
    pub fn new(num_partitions: usize) -> Self {
        Self {
            partition_locks: (0..num_partitions)
                .map(|_| Arc::new(Mutex::new(())))
                .collect(),
        }
    }

    pub fn get_partition_mutex(&self, partition_id: usize) -> Arc<Mutex<()>> {
        let mtx = &self.partition_locks[partition_id];

        mtx.clone()
    }
}

/// Extract a vector column from a batch as a flat [`FixedSizeListArray`].
///
/// Handles both regular vector columns (FixedSizeList) and multivector columns
/// (List\<FixedSizeList\>), flattening the latter.
fn vector_column_to_fsl(batch: &RecordBatch, column: &str) -> Result<FixedSizeListArray> {
    let array = get_column_from_batch(batch, column)?;
    match array.data_type() {
        arrow::datatypes::DataType::FixedSizeList(_, _) => Ok(array.as_fixed_size_list().clone()),
        arrow::datatypes::DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            let vectors = list_array.values().as_fixed_size_list();
            Ok(vectors.clone())
        }
        _ => Err(Error::Index {
            message: format!(
                "Sample training data: column {} is not a vector column",
                column
            ),
            location: location!(),
        }),
    }
}

/// Scan the entire dataset to collect training data, optionally filtering nulls.
///
/// Used when the dataset is small enough that random sampling is unnecessary.
async fn scan_all_training_data(
    dataset: &Dataset,
    column: &str,
    is_nullable: bool,
) -> Result<RecordBatch> {
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    if is_nullable {
        let column_expr = lance_datafusion::logical_expr::field_path_to_expr(column)?;
        scanner.filter_expr(column_expr.is_not_null());
    }
    let batch = scanner.try_into_batch().await?;
    info!(
        "Sample training data: retrieved {} rows scanning full dataset",
        batch.num_rows()
    );
    Ok(batch)
}

/// Sample training data from a nullable column, filtering out null rows.
///
/// For FixedSizeList columns, non-null vector bytes are accumulated directly
/// into a flat buffer (avoiding holding all source batches in memory). For
/// other types (e.g. multivector), falls back to [`sample_nullable_fallback`].
async fn sample_nullable_training_data(
    dataset: &Dataset,
    column: &str,
    sample_size_hint: usize,
    num_rows: usize,
    vector_field: &lance_core::datatypes::Field,
) -> Result<FixedSizeListArray> {
    // Use min block size + vector size to determine sample granularity.
    // For example, on object storage, block size is 64 KB. A 768-dim 32-bit
    // vector is 3 KB. So we can sample every 64 KB / 3 KB = 21 vectors.
    let block_size = dataset.object_store().block_size();
    // We provide a fallback in case of multi-vector, which will have
    // a variable size. We use 4 KB as a fallback.
    let byte_width = vector_field
        .data_type()
        .byte_width_opt()
        .unwrap_or(4 * 1024);

    let ranges = random_ranges(num_rows, sample_size_hint, block_size, byte_width);

    let mut scan = dataset.take_scan(
        Box::pin(futures::stream::iter(ranges).map(Ok)),
        Arc::new(dataset.schema().project(&[column])?),
        dataset.object_store().io_parallelism(),
    );

    // Peek at the first non-empty batch to determine the column type, then
    // dispatch to the remainder of the scan, along with the first batch, to the
    // appropriate streaming strategy.
    loop {
        let Some(batch) = scan.next().await else {
            // No data at all — return an empty FSL array.
            return fsl_values_to_array(vector_field, &[], 0);
        };
        let batch = batch?;
        let array = get_column_from_batch(&batch, column)?;
        if array.logical_null_count() >= array.len() {
            continue;
        }

        return match array.data_type() {
            arrow::datatypes::DataType::FixedSizeList(_, _) => {
                sample_nullable_fsl(
                    column,
                    sample_size_hint,
                    byte_width,
                    vector_field,
                    array,
                    scan,
                )
                .await
            }
            _ => sample_nullable_fallback(column, sample_size_hint, array, batch, scan).await,
        };
    }
}

/// Build a FixedSizeListArray from raw flat value bytes.
fn fsl_values_to_array(
    field: &lance_core::datatypes::Field,
    values_buf: &[u8],
    num_rows: usize,
) -> Result<FixedSizeListArray> {
    let (inner_field, dim) = match field.data_type() {
        DataType::FixedSizeList(f, d) => (f, d as usize),
        other => {
            return Err(Error::Index {
                message: format!("Expected FixedSizeList, got {:?}", other),
                location: location!(),
            })
        }
    };

    let elem_size = inner_field
        .data_type()
        .primitive_width()
        .ok_or_else(|| Error::Index {
            message: format!(
                "FixedSizeList inner type {:?} has no fixed width",
                inner_field.data_type()
            ),
            location: location!(),
        })?;

    let expected_bytes = num_rows * dim * elem_size;
    let buf = Buffer::from(&values_buf[..expected_bytes]);
    let values_array = arrow_array::make_array(ArrayData::try_new(
        inner_field.data_type().clone(),
        num_rows * dim,
        None,
        0,
        vec![buf],
        vec![],
    )?);

    Ok(FixedSizeListArray::try_new(
        inner_field,
        dim as i32,
        values_array,
        None,
    )?)
}

/// Stream-and-compact nullable sampling for FixedSizeList vector columns.
///
/// Unlike [`sample_nullable_fallback`], which must collect all source batches in
/// memory for interleaving, this exploits the fixed-width layout of FSL columns
/// to accumulate non-null vector bytes directly into a flat buffer, dropping
/// each source batch immediately. This keeps peak memory proportional to the
/// output sample rather than the input scan.
async fn sample_nullable_fsl(
    column: &str,
    sample_size_hint: usize,
    byte_width: usize,
    vector_field: &lance_core::datatypes::Field,
    first_array: ArrayRef,
    mut scan: crate::dataset::scanner::DatasetRecordBatchStream,
) -> Result<FixedSizeListArray> {
    let mut values_buf: Vec<u8> = Vec::with_capacity(sample_size_hint * byte_width);
    let mut num_non_null: usize = 0;

    // Process the already-read first batch.
    accumulate_fsl_non_nulls(&mut values_buf, &mut num_non_null, &first_array, byte_width)?;

    // Continue streaming remaining batches.
    while num_non_null < sample_size_hint {
        let Some(batch) = scan.next().await else {
            break;
        };
        let batch = batch?;
        let array = get_column_from_batch(&batch, column)?;
        if array.logical_null_count() >= array.len() {
            continue;
        }
        accumulate_fsl_non_nulls(&mut values_buf, &mut num_non_null, &array, byte_width)?;
    }

    let num_rows_out = num_non_null.min(sample_size_hint);
    values_buf.truncate(num_rows_out * byte_width);

    info!(
        "Sample training data: retrieved {} rows by sampling after filtering out nulls",
        num_rows_out
    );

    fsl_values_to_array(vector_field, &values_buf, num_rows_out)
}

/// Append non-null values from a FixedSizeList array into a flat byte buffer.
///
/// Uses Arrow's `filter` kernel to handle null removal and offset arithmetic,
/// then copies the resulting contiguous bytes into the output buffer.
fn accumulate_fsl_non_nulls(
    values_buf: &mut Vec<u8>,
    num_non_null: &mut usize,
    array: &ArrayRef,
    byte_width: usize,
) -> Result<()> {
    // Always filter to both remove nulls and produce a zero-offset array.
    // When there are no nulls this is just a copy, which is cheap relative
    // to the I/O cost of reading each batch.
    let mask = match array.nulls() {
        Some(nulls) => arrow_array::BooleanArray::from(nulls.inner().clone()),
        None => arrow_array::BooleanArray::from(vec![true; array.len()]),
    };
    let filtered = arrow::compute::filter(array, &mask)?;
    let fsl = filtered.as_fixed_size_list();
    let values_data = fsl.values().to_data();
    let value_bytes = &values_data.buffers()[0].as_slice()[..fsl.len() * byte_width];
    values_buf.extend_from_slice(value_bytes);
    *num_non_null += fsl.len();
    Ok(())
}

/// Fallback for nullable sampling when the column type is not FixedSizeList
/// (e.g. multivector List columns). Filters nulls from each batch as it
/// arrives, then concatenates the filtered batches.
async fn sample_nullable_fallback(
    column: &str,
    sample_size_hint: usize,
    first_array: ArrayRef,
    first_batch: RecordBatch,
    mut scan: crate::dataset::scanner::DatasetRecordBatchStream,
) -> Result<FixedSizeListArray> {
    let schema = first_batch.schema();
    let mut filtered = Vec::new();
    let mut num_non_null: usize = 0;

    // Filter and collect the already-read first batch.
    let batch = filter_non_null_rows(first_array, first_batch)?;
    num_non_null += batch.num_rows();
    filtered.push(batch);

    while num_non_null < sample_size_hint {
        let Some(batch) = scan.next().await else {
            break;
        };
        let batch = batch?;
        let array = get_column_from_batch(&batch, column)?;
        if array.logical_null_count() >= array.len() {
            continue;
        }
        let batch = filter_non_null_rows(array, batch)?;
        num_non_null += batch.num_rows();
        filtered.push(batch);
    }

    let batch = arrow::compute::concat_batches(&schema, &filtered)?;
    let num_rows_out = batch.num_rows().min(sample_size_hint);
    let batch = batch.slice(0, num_rows_out);

    info!(
        "Sample training data (fallback): retrieved {} rows by sampling after filtering out nulls",
        num_rows_out
    );

    vector_column_to_fsl(&batch, column)
}

/// Filter a batch to only include rows where `array` is non-null.
fn filter_non_null_rows(array: ArrayRef, batch: RecordBatch) -> Result<RecordBatch> {
    if let Some(nulls) = array.nulls() {
        let mask = arrow_array::BooleanArray::from(nulls.inner().clone());
        Ok(arrow::compute::filter_record_batch(&batch, &mask)?)
    } else {
        Ok(batch)
    }
}

/// Generate random ranges to sample from a dataset.
///
/// This will return an iterator of ranges that cover the whole dataset. It
/// provides an unbound iterator so that the caller can decide when to stop.
/// This is useful when the caller wants to sample a fixed number of rows, but
/// has an additional filter that must be applied.
///
/// Parameters:
/// * `num_rows`: number of rows in the dataset
/// * `sample_size_hint`: the target number of rows to be sampled in the end.
///   This is a hint for the minimum number of rows that will be consumed, but
///   the caller may consume more than this.
/// * `block_size`: the byte size of ranges that should be used.
/// * `byte_width`: the byte width of the vectors that will be sampled.
fn random_ranges(
    num_rows: usize,
    sample_size_hint: usize,
    block_size: usize,
    byte_width: usize,
) -> impl Iterator<Item = std::ops::Range<u64>> + Send {
    let rows_per_batch = 1.max(block_size / byte_width);
    let mut rng = SmallRng::from_os_rng();
    let num_bins = num_rows.div_ceil(rows_per_batch);

    let bins_iter: Box<dyn Iterator<Item = usize> + Send> = if sample_size_hint * 5 >= num_rows {
        // It's faster to just allocate and shuffle
        let mut indices = (0..num_bins).collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        Box::new(indices.into_iter())
    } else {
        // If the sample is a small proportion, then we can instead use a set
        // to track which bins we have seen. We start by using the sample_size_hint
        // to provide an efficient start, and from there we randomly choose bins
        // one by one.
        let num_bins = num_rows.div_ceil(rows_per_batch);
        // Start with the minimum number we will need.
        let min_sample_size = sample_size_hint / rows_per_batch;
        let starting_bins = (0..num_bins).choose_multiple(&mut rng, min_sample_size);
        let mut seen = starting_bins
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();

        let additional = std::iter::from_fn(move || loop {
            if seen.len() >= num_bins {
                break None;
            }
            let next = (0..num_bins).choose(&mut rng).unwrap();
            if seen.contains(&next) {
                continue;
            } else {
                seen.insert(next);
                return Some(next);
            }
        });

        Box::new(starting_bins.into_iter().chain(additional))
    };

    bins_iter.map(move |i| {
        let start = (i * rows_per_batch) as u64;
        let end = ((i + 1) * rows_per_batch) as u64;
        let end = std::cmp::min(end, num_rows as u64);
        start..end
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::Float32Type;
    use lance_datagen::{array, gen_batch, ArrayGeneratorExt, Dimension, RowCount};

    use crate::dataset::InsertBuilder;

    #[rstest::rstest]
    #[test]
    fn test_random_ranges(
        #[values(99, 100, 102)] num_rows: usize,
        #[values(10, 100)] sample_size: usize,
    ) {
        // We can just assert that the output when sorted is the same as the input
        let block_size = 100;
        let byte_width = 10;

        let bin_size = block_size / byte_width;
        assert_eq!(bin_size, 10);

        let mut ranges =
            random_ranges(num_rows, sample_size, block_size, byte_width).collect::<Vec<_>>();
        ranges.sort_by_key(|r| r.start);
        let expected = (0..num_rows as u64).step_by(bin_size).map(|start| {
            let end = std::cmp::min(start + bin_size as u64, num_rows as u64);
            start..end
        });
        assert_eq!(ranges, expected.collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn test_maybe_sample_training_data_multivector_infers_vectors_per_row() {
        let nrows: usize = 2000;
        let dims: u32 = 8;
        let vectors_per_row: u32 = 2;

        let mv = array::cycle_vec_var(
            array::rand_vec::<Float32Type>(Dimension::from(dims)),
            Dimension::from(vectors_per_row),
            Dimension::from(vectors_per_row + 1),
        );

        let data = gen_batch()
            .col("mv", mv)
            .into_batch_rows(RowCount::from(nrows as u64))
            .unwrap();

        let dataset = InsertBuilder::new("memory://")
            .execute(vec![data])
            .await
            .unwrap();

        let training_data = maybe_sample_training_data(&dataset, "mv", 1000)
            .await
            .unwrap();
        assert_eq!(training_data.len(), 1000);
    }

    #[rstest::rstest]
    #[case::f16(arrow::datatypes::DataType::Float16, 2)]
    #[case::f32(arrow::datatypes::DataType::Float32, 4)]
    #[case::f64(arrow::datatypes::DataType::Float64, 8)]
    #[test]
    fn test_fsl_values_to_array_roundtrip(
        #[case] elem_type: arrow::datatypes::DataType,
        #[case] elem_size: usize,
    ) {
        let dim = 4;
        let num_rows = 3;
        // Fill with recognizable byte patterns: each element gets its index as bytes.
        let num_elems = num_rows * dim;
        let values_buf: Vec<u8> = (0..num_elems)
            .flat_map(|i| {
                let mut bytes = vec![0u8; elem_size];
                // Write index into the first bytes (little-endian).
                let i_bytes = (i as u32).to_le_bytes();
                bytes[..i_bytes.len().min(elem_size)]
                    .copy_from_slice(&i_bytes[..i_bytes.len().min(elem_size)]);
                bytes
            })
            .collect();

        let dt = DataType::FixedSizeList(
            Arc::new(arrow::datatypes::Field::new("item", elem_type, true)),
            dim as i32,
        );
        let field = lance_core::datatypes::Field::new_arrow("vec", dt, true).unwrap();

        let fsl = fsl_values_to_array(&field, &values_buf, num_rows).unwrap();
        assert_eq!(fsl.len(), num_rows);
        assert_eq!(fsl.value_length(), dim as i32);

        // Verify the raw bytes round-tripped correctly.
        let out_data = fsl.values().to_data();
        let out_bytes = out_data.buffers()[0].as_slice();
        assert_eq!(&out_bytes[..values_buf.len()], &values_buf[..]);
    }

    #[rstest::rstest]
    #[case::f32(array::rand_vec::<Float32Type>(Dimension::from(8)))]
    #[case::f64(array::rand_vec::<arrow_array::types::Float64Type>(Dimension::from(8)))]
    #[tokio::test]
    async fn test_maybe_sample_training_data_nullable_fsl(
        #[case] vec_gen: Box<dyn lance_datagen::ArrayGenerator>,
    ) {
        let nrows: usize = 2000;
        let dims: u32 = 8;
        let sample_size: usize = 500;

        let data = gen_batch()
            .col("vec", vec_gen.with_random_nulls(0.5))
            .into_batch_rows(RowCount::from(nrows as u64))
            .unwrap();

        let col = data.column_by_name("vec").unwrap();
        assert!(col.null_count() > 0, "test data should have nulls");

        let dataset = InsertBuilder::new("memory://nullable_fsl_test")
            .execute(vec![data])
            .await
            .unwrap();

        let training_data = maybe_sample_training_data(&dataset, "vec", sample_size)
            .await
            .unwrap();

        assert!(training_data.len() <= sample_size);
        assert!(training_data.len() > 0);
        assert_eq!(training_data.null_count(), 0);
        assert_eq!(training_data.value_length(), dims as i32);
    }

    #[tokio::test]
    async fn test_estimate_multivector_vectors_per_row_fallback_1030() {
        let nrows: usize = 256;
        let dims: u32 = 8;

        let mv = array::cycle_vec_var(
            array::rand_vec::<Float32Type>(Dimension::from(dims)),
            Dimension::from(2),
            Dimension::from(3),
        )
        .with_random_nulls(1.0);

        let data = gen_batch()
            .col("mv", mv)
            .into_batch_rows(RowCount::from(nrows as u64))
            .unwrap();

        let dataset = InsertBuilder::new("memory://")
            .execute(vec![data])
            .await
            .unwrap();

        let n = estimate_multivector_vectors_per_row(&dataset, "mv", nrows)
            .await
            .unwrap();
        assert_eq!(n, 1030);
    }
}
