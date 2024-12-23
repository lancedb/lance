// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, FixedSizeListArray};
use lance_core::datatypes::Schema;
use snafu::{location, Location};
use tokio::sync::Mutex;

use crate::dataset::Dataset;
use crate::{Error, Result};

pub fn get_vector_dim(schema: &Schema, column: &str) -> Result<usize> {
    let field = schema.field(column).ok_or(Error::Index {
        message: format!("Column {} does not exist in schema {}", column, schema),
        location: location!(),
    })?;
    infer_vector_dim(&field.data_type())
}

fn infer_vector_dim(data_type: &arrow::datatypes::DataType) -> Result<usize> {
    match data_type {
        arrow::datatypes::DataType::FixedSizeList(_, dim) => Ok(*dim as usize),
        arrow::datatypes::DataType::List(inner) => infer_vector_dim(inner.data_type()),
        _ => Err(Error::Index {
            message: format!("Data type is not a FixedSizeListArray, but {:?}", data_type),
            location: location!(),
        }),
    }
}

// this checks whether the given column is with a valid vector type
// returns the vector type (FixedSizeList for vectors, or List for multivectors),
// and element type (Float16/Float32/Float64 or UInt8 for binary vectors).
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

fn infer_vector_element_type(
    data_type: &arrow::datatypes::DataType,
) -> Result<arrow_schema::DataType> {
    match data_type {
        arrow::datatypes::DataType::FixedSizeList(element_field, _) => {
            match element_field.data_type() {
                arrow::datatypes::DataType::Float16
                | arrow::datatypes::DataType::Float32
                | arrow::datatypes::DataType::Float64
                | arrow::datatypes::DataType::UInt8 => Ok(element_field.data_type().clone()),
                _ => Err(Error::Index {
                    message: format!(
                        "vector element is not expected type (Float16/Float32/Float64 or UInt8) {:?}",
                        element_field.data_type()
                    ),
                    location: location!(),
                }),
            }
        }
        arrow::datatypes::DataType::List(inner) => infer_vector_element_type(inner.data_type()),
        _ => Err(Error::Index {
            message: format!("vector is not with valid data type: {:?}", data_type),
            location: location!(),
        }),
    }
}

pub fn get_vector_element_type(dataset: &Dataset, column: &str) -> Result<arrow_schema::DataType> {
    let schema = dataset.schema();
    let field = schema.field(column).ok_or(Error::Index {
        message: format!("column {} does not exist in schema {}", column, schema),
        location: location!(),
    })?;
    let data_type = field.data_type();
    if let arrow_schema::DataType::FixedSizeList(element_field, _) = data_type {
        Ok(element_field.data_type().clone())
    } else {
        Err(Error::Index {
            message: format!("column {} is not a vector type: {:?}", column, data_type),
            location: location!(),
        })
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
    let batch = if num_rows > sample_size_hint {
        let projection = dataset.schema().project(&[column])?;
        dataset.sample(sample_size_hint, &projection).await?
    } else {
        let mut scanner = dataset.scan();
        scanner.project(&[column])?;
        scanner.try_into_batch().await?
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
