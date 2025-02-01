// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, FixedSizeListArray};
use futures::StreamExt;
use lance_arrow::{interleave_batches, DataTypeExt};
use lance_core::datatypes::Schema;
use log::info;
use rand::rngs::SmallRng;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::SeedableRng;
use snafu::location;
use tokio::sync::Mutex;

use crate::dataset::Dataset;
use crate::{Error, Result};

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
        _ => Err(Error::Index {
            message: format!("Data type is not a vector (FixedSizeListArray or List<FixedSizeListArray>), but {:?}", data_type),
            location: location!(),
        }),
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
                | arrow::datatypes::DataType::UInt8 => Ok(element_field.data_type().clone()),
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
        _ => Err(Error::Index {
            message: format!(
                "Data type is not a vector (FixedSizeListArray or List<FixedSizeListArray>), but {:?}",
                data_type
            ),
            location: location!(),
        }),
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

    let batch = if num_rows > sample_size_hint && !is_nullable {
        let projection = dataset.schema().project(&[column])?;
        let batch = dataset.sample(sample_size_hint, &projection).await?;
        info!(
            "Sample training data: retrieved {} rows by sampling",
            batch.num_rows()
        );
        batch
    } else if num_rows > sample_size_hint && is_nullable {
        // Use min block size + vector size to determine sample granularity
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

        let mut collected = Vec::with_capacity(ranges.size_hint().0);
        let mut indices = Vec::with_capacity(sample_size_hint);
        let mut num_non_null = 0;

        let mut scan = dataset.take_scan(
            Box::pin(futures::stream::iter(ranges).map(Ok)),
            Arc::new(dataset.schema().project(&[column])?),
            dataset.object_store().io_parallelism(),
        );

        while let Some(batch) = scan.next().await {
            let batch = batch?;

            let array = batch.column_by_name(column).ok_or(Error::Index {
                message: format!(
                    "Sample training data: column {} does not exist in return",
                    column
                ),
                location: location!(),
            })?;
            let null_count = array.logical_null_count();
            if null_count < array.len() {
                num_non_null += array.len() - null_count;

                let batch_i = collected.len();
                if let Some(null_buffer) = array.nulls() {
                    for i in null_buffer.valid_indices() {
                        indices.push((batch_i, i));
                    }
                } else {
                    indices.extend((0..array.len()).map(|i| (batch_i, i)));
                }

                collected.push(batch);
            }
            if num_non_null >= sample_size_hint {
                break;
            }
        }

        let batch = interleave_batches(&collected, &indices).map_err(|err| Error::Index {
            message: format!("Sample training data: {}", err),
            location: location!(),
        })?;
        info!(
            "Sample training data: retrieved {} rows by sampling after filtering out nulls",
            batch.num_rows()
        );
        batch
    } else {
        let mut scanner = dataset.scan();
        scanner.project(&[column])?;
        if is_nullable {
            scanner.filter_expr(datafusion_expr::col(column).is_not_null());
        }
        let batch = scanner.try_into_batch().await?;
        info!(
            "Sample training data: retrieved {} rows scanning full datasets",
            batch.num_rows()
        );
        batch
    };

    let array = batch.column_by_name(column).ok_or(Error::Index {
        message: format!(
            "Sample training data: column {} does not exist in return",
            column
        ),
        location: location!(),
    })?;

    match array.data_type() {
        arrow::datatypes::DataType::FixedSizeList(_, _) => Ok(array.as_fixed_size_list().clone()),
        // for multivector, flatten the vectors into a FixedSizeListArray
        arrow::datatypes::DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            let vectors = list_array.values().as_fixed_size_list();
            Ok(vectors.clone())
        }
        _ => Err(Error::Index {
            message: format!(
                "Sample training data: column {} is not a FixedSizeListArray",
                column
            ),
            location: location!(),
        }),
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
    let mut rng = SmallRng::from_entropy();
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
}
