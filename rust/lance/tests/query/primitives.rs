// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::datatypes::*;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
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
            // TODO: fix bug with bitmap and btree https://github.com/lancedb/lance/issues/4756
            // TODO: fix bug with zone map https://github.com/lancedb/lance/issues/4758
            // TODO: Add boolean to bloom filter supported types https://github.com/lancedb/lance/issues/4757
            // [None, Some(IndexType::Bitmap), Some(IndexType::BTree), Some(IndexType::BloomFilter), Some(IndexType::ZoneMap)],
            [None],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value").await;
            test_filter(&original, &ds, "NOT value").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::int8(DataType::Int8)]
#[case::int16(DataType::Int16)]
#[case::int32(DataType::Int32)]
#[case::int64(DataType::Int64)]
#[case::uint8(DataType::UInt8)]
#[case::uint16(DataType::UInt16)]
#[case::uint32(DataType::UInt32)]
#[case::uint64(DataType::UInt64)]
async fn test_query_integer(#[case] data_type: DataType) {
    let value_generator = match data_type {
        DataType::Int8 => array::rand_primitive::<Int8Type>(data_type),
        DataType::Int16 => array::rand_primitive::<Int16Type>(data_type),
        DataType::Int32 => array::rand_primitive::<Int32Type>(data_type),
        DataType::Int64 => array::rand_primitive::<Int64Type>(data_type),
        DataType::UInt8 => array::rand_primitive::<UInt8Type>(data_type),
        DataType::UInt16 => array::rand_primitive::<UInt16Type>(data_type),
        DataType::UInt32 => array::rand_primitive::<UInt32Type>(data_type),
        DataType::UInt64 => array::rand_primitive::<UInt64Type>(data_type),
        _ => unreachable!(),
    };

    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", value_generator.with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            // TODO: add zone map and bloom filter once we fix https://github.com/lancedb/lance/issues/4758
            [None, Some(IndexType::Bitmap), Some(IndexType::BTree)],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 20").await;
            test_filter(&original, &ds, "NOT (value > 20)").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

// TODO: floats (including NaN, +/-Inf, +/-0)
// TODO: decimals
// TODO: binary
// TODO: strings (including largestrings and view)
// TODO: timestamps
