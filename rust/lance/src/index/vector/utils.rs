// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, FixedSizeListArray};
use lance_core::datatypes::Schema;
use log::info;
use snafu::{location, Location};
use tokio::sync::Mutex;

use crate::dataset::{Dataset, ProjectionRequest, TakeBuilder};
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

    let is_nullable = dataset
        .schema()
        .field(column)
        .ok_or(Error::Index {
            message: format!(
                "Sample training data: column {} does not exist in schema",
                column
            ),
            location: location!(),
        })?
        .nullable;

    let batch = if num_rows > sample_size_hint && !is_nullable {
        let projection = dataset.schema().project(&[column])?;
        let batch = dataset.sample(sample_size_hint, &projection).await?;
        info!(
            "Sample training data: retrieved {} rows by sampling",
            batch.num_rows()
        );
        batch
    } else if num_rows > sample_size_hint && is_nullable {
        // Need to filter out null values
        // Use a scan to collect row ids. Then sample from the row ids. Then do take.
        let row_addrs = dataset
            .scan()
            .filter_expr(datafusion_expr::col(column).is_not_null())
            .with_row_address()
            .project::<&str>(&[])?
            .try_into_batch()
            .await?;
        debug_assert_eq!(row_addrs.num_columns(), 1);
        debug_assert_eq!(row_addrs["_rowaddr"].logical_null_count(), 0);
        let row_addrs = row_addrs
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .ok_or(Error::Index {
                message: format!(
                    "Sample training data: column {} is not a UInt64Array",
                    column
                ),
                location: location!(),
            })?;

        let batch = TakeBuilder::try_new_from_addresses(
            Arc::new(dataset.clone()),
            row_addrs.values().to_vec(),
            Arc::new(
                ProjectionRequest::from_columns([column], dataset.schema())
                    .into_projection_plan(dataset.schema())?,
            ),
        )?
        .execute()
        .await?;
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
