// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::datatypes::Int32Type;
use arrow_array::RecordBatch;
use lance::Dataset;

use lance_datagen::{array, gen_batch, ArrayGeneratorExt, RowCount};
use lance_index::IndexType;

use super::{test_filter, test_scan, test_take};
use crate::utils::DatasetTestCases;

#[tokio::test]
async fn test_query_bool() {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col(
            "value",
            array::cycle_bool(vec![true, false]).with_random_nulls(0.1),
        )
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [None, Some(IndexType::Bitmap), Some(IndexType::BTree)],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value").await;
            test_filter(&original, &ds, "!value").await;
        })
        .await
}

#[tokio::test]
async fn test_query_integers() {
    todo!()
}

#[tokio::test]
async fn test_query_floats() {
    todo!()
}

#[tokio::test]
async fn test_query_decimals() {
    todo!()
}

#[tokio::test]
async fn test_query_strings() {
    todo!()
}

#[tokio::test]
async fn test_query_timestamps() {
    todo!()
}
