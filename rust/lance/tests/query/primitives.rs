// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::*;
use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, Int32Array,
    LargeBinaryArray, LargeStringArray, RecordBatch, StringArray, StringViewArray,
};
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
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 20").await;
            test_filter(&original, &ds, "NOT (value > 20)").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
            test_filter(&original, &ds, "(value != 0) OR (value < 20)").await;
            test_filter(&original, &ds, "NOT ((value != 0) OR (value < 20))").await;
            test_filter(
                &original,
                &ds,
                "(value != 5) OR ((value != 52) OR (value IS NULL))",
            )
            .await;
            test_filter(
                &original,
                &ds,
                "NOT ((value != 5) OR ((value != 52) OR (value IS NULL)))",
            )
            .await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::float32(DataType::Float32)]
#[case::float64(DataType::Float64)]
async fn test_query_float(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();
    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::BTree),
                Some(IndexType::Bitmap),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 0.5").await;
            test_filter(&original, &ds, "NOT (value > 0.5)").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
            test_filter(&original, &ds, "isnan(value)").await;
            test_filter(&original, &ds, "not isnan(value)").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::float32(DataType::Float32)]
#[case::float64(DataType::Float64)]
async fn test_query_float_special_values(#[case] data_type: DataType) {
    let value_array: Arc<dyn arrow_array::Array> = match data_type {
        DataType::Float32 => Arc::new(Float32Array::from(vec![
            Some(0.0_f32),
            Some(-0.0_f32),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(f32::NAN),
            Some(1.0_f32),
            Some(-1.0_f32),
            Some(f32::MIN),
            Some(f32::MAX),
            None,
        ])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![
            Some(0.0_f64),
            Some(-0.0_f64),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(f64::NAN),
            Some(1.0_f64),
            Some(-1.0_f64),
            Some(f64::MIN),
            Some(f64::MAX),
            None,
        ])),
        _ => unreachable!(),
    };

    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));

    let batch =
        RecordBatch::try_from_iter(vec![("id", id_array as ArrayRef), ("value", value_array)])
            .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::BTree),
                Some(IndexType::Bitmap),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 0.0").await;
            test_filter(&original, &ds, "value < 0.0").await;
            test_filter(&original, &ds, "value = 0.0").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
            test_filter(&original, &ds, "isnan(value)").await;
            test_filter(&original, &ds, "not isnan(value)").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::date32(DataType::Date32)]
#[case::date64(DataType::Date64)]
async fn test_query_date(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value < current_date()").await;
            test_filter(&original, &ds, "value > DATE '2024-01-01'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::timestamp_second(DataType::Timestamp(TimeUnit::Second, None))]
#[case::timestamp_millisecond(DataType::Timestamp(TimeUnit::Millisecond, None))]
#[case::timestamp_microsecond(DataType::Timestamp(TimeUnit::Microsecond, None))]
#[case::timestamp_nanosecond(DataType::Timestamp(TimeUnit::Nanosecond, None))]
async fn test_query_timestamp(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::BTree),
                Some(IndexType::Bitmap),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value < current_timestamp()").await;
            test_filter(&original, &ds, "value > TIMESTAMP '2024-01-01 00:00:00'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::utf8(DataType::Utf8)]
#[case::large_utf8(DataType::LargeUtf8)]
// #[case::string_view(DataType::Utf8View)] // TODO: https://github.com/lancedb/lance/issues/5172
async fn test_query_string(#[case] data_type: DataType) {
    // Create arrays that include empty strings
    let string_values = vec![
        Some("hello"),
        Some("world"),
        Some(""),
        Some("test"),
        Some("data"),
        Some(""),
        None,
        Some("apple"),
        Some("zebra"),
        Some(""),
    ];

    let value_array: ArrayRef = match data_type {
        DataType::Utf8 => Arc::new(StringArray::from(string_values.clone())),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(string_values.clone())),
        DataType::Utf8View => Arc::new(StringViewArray::from(string_values.clone())),
        _ => unreachable!(),
    };

    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));

    let batch =
        RecordBatch::try_from_iter(vec![("id", id_array as ArrayRef), ("value", value_array)])
            .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value = 'hello'").await;
            test_filter(&original, &ds, "value != 'hello'").await;
            test_filter(&original, &ds, "value = ''").await;
            test_filter(&original, &ds, "value > 'hello'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
#[case::binary(DataType::Binary)]
#[case::large_binary(DataType::LargeBinary)]
// #[case::binary_view(DataType::BinaryView)] // TODO: https://github.com/lancedb/lance/issues/5172
async fn test_query_binary(#[case] data_type: DataType) {
    // Create arrays that include empty binary
    let binary_values = vec![
        Some(b"hello".as_slice()),
        Some(b"world".as_slice()),
        Some(b"".as_slice()),
        Some(b"test".as_slice()),
        Some(b"data".as_slice()),
        Some(b"".as_slice()),
        None,
        Some(b"apple".as_slice()),
        Some(b"zebra".as_slice()),
        Some(b"".as_slice()),
    ];

    let value_array: ArrayRef = match data_type {
        DataType::Binary => Arc::new(BinaryArray::from(binary_values.clone())),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from(binary_values.clone())),
        DataType::BinaryView => Arc::new(BinaryViewArray::from(binary_values.clone())),
        _ => unreachable!(),
    };

    let id_array = Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>()));

    let batch =
        RecordBatch::try_from_iter(vec![("id", id_array as ArrayRef), ("value", value_array)])
            .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            [
                None,
                Some(IndexType::Bitmap),
                Some(IndexType::BTree),
                Some(IndexType::BloomFilter),
                Some(IndexType::ZoneMap),
            ],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value = X'68656C6C6F'").await; // 'hello' in hex
            test_filter(&original, &ds, "value != X'68656C6C6F'").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}

#[tokio::test]
#[rstest::rstest]
// TODO: Add Decimal32 and Decimal64 https://github.com/lancedb/lance/issues/5174
#[case::decimal128(DataType::Decimal128(38, 10))]
#[case::decimal256(DataType::Decimal256(76, 20))]
async fn test_query_decimal(#[case] data_type: DataType) {
    let batch = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("value", array::rand_type(&data_type).with_random_nulls(0.1))
        .into_batch_rows(RowCount::from(60))
        .unwrap();

    DatasetTestCases::from_data(batch)
        .with_index_types(
            "value",
            // NOTE: BloomFilter not supported for decimals
            [None, Some(IndexType::Bitmap), Some(IndexType::BTree)],
        )
        .run(|ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value > 0").await;
            test_filter(&original, &ds, "value < 0").await;
            test_filter(&original, &ds, "value is null").await;
            test_filter(&original, &ds, "value is not null").await;
        })
        .await
}
