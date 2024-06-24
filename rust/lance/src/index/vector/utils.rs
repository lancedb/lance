// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, FixedSizeListArray};
use arrow_schema::{Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use futures::stream::TryStreamExt;
use snafu::{location, Location};
use lance_arrow::{self, FixedSizeListArrayExt};
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

    batch
        .column_by_name(column)
        .ok_or(Error::Index {
            message: format!(
                "Sample training data: column {} does not exist in return",
                column
            ),
            location: location!()
        })?
        .as_fixed_size_list().clone()
        .convert_to_floating_point()
        .map_err(|e| lance_core::Error::Arrow { 
            message: e.to_string(), 
            location: location!() })
}
