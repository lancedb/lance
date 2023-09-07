// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use arrow_select::concat::concat_batches;
use futures::stream::TryStreamExt;

use crate::arrow::{linalg::matrix::MatrixView, *};
use crate::dataset::Dataset;
use crate::{Error, Result};

/// Maybe sample training data from dataset, specified by column name.
pub async fn maybe_sample_training_data(
    dataset: &Dataset,
    column: &str,
    sample_size_hint: usize,
) -> Result<MatrixView> {
    let num_rows = dataset.count_rows().await?;
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
    })?;
    let fixed_size_array = as_fixed_size_list_array(array);
    fixed_size_array.try_into()
}
