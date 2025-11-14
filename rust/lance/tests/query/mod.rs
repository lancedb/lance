// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, RecordBatch, UInt32Array};
use arrow_select::concat::concat_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use lance::dataset::scanner::ColumnOrdering;
use lance::Dataset;
use lance_datafusion::udf::register_functions;

/// Creates a fresh SessionContext with Lance UDFs registered
fn create_datafusion_context() -> SessionContext {
    let ctx = SessionContext::new();
    register_functions(&ctx);
    ctx
}

mod primitives;
mod vectors;

/// Scanning and ordering by id should give same result as original.
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

/// Taking specific rows should give the same result as taking from the original.
async fn test_take(original: &RecordBatch, ds: &Dataset) {
    let num_rows = original.num_rows();
    let cases: Vec<Vec<usize>> = vec![
        vec![0, 1, 2],                    // First few rows
        vec![5, 3, 1],                    // Out of order
        vec![0],                          // Single row
        vec![],                           // Empty
        (0..num_rows.min(10)).collect(),  // Sequential
        vec![num_rows - 1, 0],            // Last and first
        vec![1, 1, 2],                    // Duplicate indices
        vec![0, 0, 0],                    // All same index
        vec![num_rows - 1, num_rows - 1], // Duplicate of last row
    ];

    for indices in cases {
        // Convert to u64 for Lance take
        let indices_u64: Vec<u64> = indices.iter().map(|&i| i as u64).collect();

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

/// Querying with filter should give same result as filtering original
/// record batch in DataFusion.
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

    let ctx = create_datafusion_context();
    let table = MemTable::try_new(original.schema(), vec![vec![original.clone()]]).unwrap();
    ctx.register_table("t", Arc::new(table)).unwrap();

    let sql = format!("SELECT * FROM t WHERE {} ORDER BY id", predicate);
    let df = ctx.sql(&sql).await.unwrap();
    let expected_batches = df.collect().await.unwrap();
    let expected = concat_batches(&original.schema(), &expected_batches).unwrap();

    assert_eq!(&expected, &scanned);
}

/// Test that an exhaustive ANN query gives the same results as brute force
/// KNN against the original batch.
///
/// By exhaustive ANN, I mean we search all the partitions so we get perfect recall.
async fn test_ann(original: &RecordBatch, ds: &Dataset, column: &str, predicate: Option<&str>) {
    // Extract first vector from the column as query vector
    let vector_column = original.column_by_name(column).unwrap();
    let fixed_size_list = vector_column.as_fixed_size_list();

    // Extract the first vector's values as a new array
    let vector_values = fixed_size_list
        .values()
        .slice(0, fixed_size_list.value_length() as usize);
    let query_vector = vector_values;

    let mut scanner = ds.scan();
    scanner
        .nearest(column, query_vector.as_ref(), 10)
        .unwrap()
        .prefilter(true)
        .refine(2);
    if let Some(pred) = predicate {
        scanner.filter(pred).unwrap();
    }
    let result = scanner.try_into_batch().await.unwrap();

    // Use DataFusion to apply same vector search using SQL
    let ctx = create_datafusion_context();
    let table = MemTable::try_new(original.schema(), vec![vec![original.clone()]]).unwrap();
    ctx.register_table("t", Arc::new(table)).unwrap();

    // Convert query vector to SQL array literal
    let float_array = query_vector.as_primitive::<arrow::datatypes::Float32Type>();
    let vector_values_str = float_array
        .values()
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    // DataFusion's built-in `array_distance` function uses L2 distance.
    let sql = format!(
        "SELECT * FROM t {} ORDER BY array_distance(t.{}, [{}]) LIMIT 10",
        if let Some(pred) = predicate {
            format!("WHERE {}", pred)
        } else {
            String::new()
        },
        column,
        vector_values_str
    );

    let df = ctx.sql(&sql).await.unwrap();
    let expected_batches = df.collect().await.unwrap();
    let expected = concat_batches(&original.schema(), &expected_batches).unwrap();

    // Compare only the main data (excluding _distance column which Lance adds).
    // We validate that both return the same number of rows and same row ordering.
    // Note: We don't validate the _distance column values because:
    // 1. ANN indices provide approximate distances, not exact values
    // 2. Some distance functions return ordering values (e.g., squared euclidean
    //    without the final sqrt step) rather than true distances
    assert_eq!(
        expected.num_rows(),
        result.num_rows(),
        "Different number of results"
    );

    // Compare the first few columns (excluding _distance)
    for (col_idx, field) in original.schema().fields().iter().enumerate() {
        let expected_col = expected.column(col_idx);
        let result_col = result.column(col_idx);
        assert_eq!(
            expected_col,
            result_col,
            "Column '{}' differs between DataFusion and Lance results",
            field.name()
        );
    }
}
