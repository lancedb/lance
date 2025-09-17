// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, UInt32Array};
use arrow_select::concat::concat_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use lance::dataset::scanner::ColumnOrdering;
use lance::Dataset;

mod primitives;
mod vectors;

async fn test_scan(original: &RecordBatch, ds: &Dataset) {
    let mut scanner = ds.scan();
    scanner
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "id".to_string(),
        )]))
        .unwrap();
    let scanned = scanner.try_into_batch().await.unwrap();

    assert_eq!(original, &scanned);
}

async fn test_take(original: &RecordBatch, ds: &Dataset) {
    let num_rows = original.num_rows();
    let cases: Vec<Vec<usize>> = vec![
        vec![0, 1, 2],                   // First few rows
        vec![5, 3, 1],                   // Out of order
        vec![0],                         // Single row
        vec![],                          // Empty
        (0..num_rows.min(10)).collect(), // Sequential
        vec![num_rows - 1, 0],           // Last and first
    ];

    for indices in cases {
        // Skip cases with invalid indices
        if indices.iter().any(|&i| i >= num_rows) {
            continue;
        }

        // Convert to u64 for Lance take
        let indices_u64: Vec<u64> = indices.iter().map(|&i| i as u64).collect();

        if indices_u64.is_empty() {
            // Skip empty case as Lance may not handle it the same way
            continue;
        }

        let taken_ds = ds.take(&indices_u64, ds.schema().clone()).await.unwrap();

        // Take from RecordBatch using arrow::compute
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indices_array = UInt32Array::from(indices_u32);
        let taken_rb = arrow::compute::take_record_batch(original, &indices_array).unwrap();

        assert_eq!(
            taken_rb, taken_ds,
            "Take results don't match for indices: {:?}",
            indices
        );
    }
}

async fn test_filter(original: &RecordBatch, ds: &Dataset, predicate: &str) {
    // Scan with filter and order
    let mut scanner = ds.scan();
    scanner
        .filter(predicate)
        .unwrap()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "id".to_string(),
        )]))
        .unwrap();
    let scanned = scanner.try_into_batch().await.unwrap();

    let ctx = SessionContext::new();
    let table = MemTable::try_new(original.schema(), vec![vec![original.clone()]]).unwrap();
    ctx.register_table("t", Arc::new(table)).unwrap();

    let sql = format!("SELECT * FROM t WHERE {} ORDER BY id", predicate);
    let df = ctx.sql(&sql).await.unwrap();
    let expected_batches = df.collect().await.unwrap();
    let expected = concat_batches(&original.schema(), &expected_batches).unwrap();

    assert_eq!(&expected, &scanned);
}

async fn test_ann(_original: &RecordBatch, _ds: &Dataset, _predicate: Option<&str>) {
    todo!("Scan ds with the ANN predicate");
}
