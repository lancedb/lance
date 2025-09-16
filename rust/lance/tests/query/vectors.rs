// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{test_ann, test_scan, test_take};
use crate::utils::DatasetTestCases;
use arrow::datatypes::{Date32Type, Float32Type, Int32Type};
use arrow_array::RecordBatch;
use lance::Dataset;
use lance_datagen::{array, gen_batch, ArrayGeneratorExt, Dimension, RowCount};
use lance_index::IndexType;

#[tokio::test]
async fn test_query_vector() {
    todo!()
}

#[tokio::test]
async fn test_query_prefilter_date() {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::step::<Date32Type>().with_random_nulls(0.1))
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types("value", [None, Some(IndexType::BTree)])
        .with_index_types(
            "vec",
            [
                None,
                Some(IndexType::IvfPq),
                Some(IndexType::IvfFlat),
                Some(IndexType::IvfHnswFlat),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_ann(&original, &ds, Some("value is not null")).await;
            test_ann(
                &original,
                &ds,
                Some("value >= DATE '2020-06-01' AND value <= DATE '2021-06-01'"),
            )
            .await;
        })
        .await
}
