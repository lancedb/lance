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
    let data_type = field.data_type();
    if let arrow_schema::DataType::FixedSizeList(_, dim) = data_type {
        Ok(dim as usize)
    } else {
        Err(Error::Index {
            message: format!(
                "Column {} is not a FixedSizeListArray, but {:?}",
                column, data_type
            ),
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
    let projection = dataset.schema().project(&[column])?;
    let batch = if num_rows > sample_size_hint {
        dataset.sample(sample_size_hint, &projection).await?
    } else {
        let mut scanner = dataset.scan();
        scanner.project(&[column])?;
        let batches = scanner
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&Arc::new(ArrowSchema::from(&projection)), &batches)?
    };

    let array = batch.column_by_name(column).ok_or(Error::Index {
        message: format!(
            "Sample training data: column {} does not exist in return",
            column
        ),
        location: location!(),
    })?;
    Ok(array.as_fixed_size_list().clone())
}

#[derive(Debug)]
pub (crate) struct PartitionLoadLock {
    partition_locks: Vec<Arc<Mutex<()>>>,
}

impl PartitionLoadLock {
    pub fn new(num_partitions: usize) -> Self {
        Self {
            partition_locks: (0..num_partitions).map(|_| Arc::new(Mutex::new(()))).collect(),
        }
    }

    pub fn get_partition_mutex(&self, partition_id: usize) -> Arc<Mutex<()>> {
        let mtx = &self.partition_locks[partition_id];
        
        mtx.clone()
    }
}
