// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, FixedSizeListArray};
use arrow_schema::Schema as ArrowSchema;
use arrow_select::concat::concat_batches;
use futures::stream::TryStreamExt;
use snafu::{location, Location};
use tokio::sync::Mutex;

use crate::dataset::Dataset;
use crate::{Error, Result};

pub fn get_vector_dim(dataset: &Dataset, column: &str) -> Result<usize> {
    let schema = dataset.schema();
    let field = schema.field(column).ok_or(Error::Index {
        message: format!("Column {} does not exist in schema {}", column, schema),
        location: location!(),
    })?;
    infer_vector_dim(&field.data_type())
}

fn infer_vector_dim(data_type: &arrow_schema::DataType) -> Result<usize> {
    match data_type {
        arrow_schema::DataType::FixedSizeList(_, dim) => Ok(*dim as usize),
        arrow_schema::DataType::List(field) => infer_vector_dim(field.data_type()),
        _ => Err(Error::Index {
            message: format!("Column is not a FixedSizeListArray, but {:?}", data_type),
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
    modal_index: Option<usize>,
    sample_size_hint: usize,
) -> Result<FixedSizeListArray> {
    let num_rows = dataset.count_rows(None).await?;
    let vector_column = match modal_index {
        Some(index) => format!("{}[{}]", column, index),
        None => column.to_string(),
    };
    let projection = dataset.schema().project(&[&vector_column])?;
    let batch = if num_rows > sample_size_hint {
        dataset.sample(sample_size_hint, &projection).await?
    } else {
        let mut scanner = dataset.scan();
        scanner.project(&[vector_column])?;
        let batches = scanner
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&Arc::new(ArrowSchema::from(&projection)), &batches)?
    };

    let array = batch.column(0);
    Ok(array.as_fixed_size_list().clone())
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
