// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A generic test suite.
//!
//! This suite aims to get wide coverage of cases in terms of data types, index
//! types, and dataset states.
//!
//! Data type test should include edge cases for each data type. For example:
//! * Null values
//! * For floats: NaN, Infinity, +/-0
//! * For integers: min / max
//! * For lists: nullability and empty values at all levels
//!
//! Data state includes:
//! * Deletion
//! * Fragmentation of data files
//! * Fragmentation of indices.

use std::{panic::AssertUnwindSafe, sync::Arc};

use arrow_array::{ArrayRef, BooleanArray, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::FutureExt;
use lance_datagen::{array, gen_batch};
use lance_index::{DatasetIndexExt, IndexType};

use crate::{
    dataset::{InsertBuilder, WriteParams},
    Dataset,
};

/// Represents the various test states we want to cover
#[derive(Clone, Copy, Debug)]
struct TableState {
    fragmentation: Fragmentation,
    deletion: DeletionState,
}

/// All combinations of test states to cover
const TABLE_STATES: &[TableState] = &[
    TableState {
        fragmentation: Fragmentation::SingleFragment,
        deletion: DeletionState::NoDeletions,
    },
    TableState {
        fragmentation: Fragmentation::SingleFragment,
        deletion: DeletionState::DeleteOdd,
    },
    TableState {
        fragmentation: Fragmentation::SingleFragment,
        deletion: DeletionState::DeleteEven,
    },
    TableState {
        fragmentation: Fragmentation::MultiFragment,
        deletion: DeletionState::NoDeletions,
    },
    TableState {
        fragmentation: Fragmentation::MultiFragment,
        deletion: DeletionState::DeleteOdd,
    },
    TableState {
        fragmentation: Fragmentation::MultiFragment,
        deletion: DeletionState::DeleteEven,
    },
];

/// Helper function to run tests across all table states and index types with panic catching
async fn test_states<F, Fut>(
    states: &[TableState],
    index_types: impl IntoIterator<Item = Option<IndexType>> + Clone,
    test_fn: F,
) where
    F: Fn(Dataset, RecordBatch) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    for &state in states {
        for index_type in index_types.clone() {
            let (original, ds) = create_test_dataset(state, index_type).await;

            let context = format!(
                "fragmentation: {:?}, deletion: {:?}, index: {:?}",
                state.fragmentation, state.deletion, index_type
            );

            AssertUnwindSafe(test_fn(ds, original.clone()))
                .catch_unwind()
                .await
                .unwrap_or_else(|_| panic!("Test failed for {}", context));
        }
    }
}

/// Create a test dataset with the given state and index configuration
async fn create_test_dataset(
    state: TableState,
    index_type: Option<IndexType>,
) -> (RecordBatch, Dataset) {
    let data_type = DataType::Boolean; // For now, start with Boolean
    let original = test_batch(&data_type);

    let ds = create_dataset_with_config(
        original.clone(),
        state.fragmentation,
        state.deletion,
        index_type,
    )
    .await;

    (original, ds)
}

#[derive(Clone, Copy, Debug)]
enum Fragmentation {
    SingleFragment,
    MultiFragment,
}

#[derive(Clone, Copy, Debug)]
enum DeletionState {
    NoDeletions,
    DeleteOdd,
    DeleteEven,
}

/// Helper function to create a dataset with specific configuration
async fn create_dataset_with_config(
    data: RecordBatch,
    fragmentation: Fragmentation,
    deletion_state: DeletionState,
    index_type: Option<IndexType>,
) -> Dataset {
    // Configure fragmentation
    let mut ds = match fragmentation {
        Fragmentation::SingleFragment => InsertBuilder::new("memory://")
            .execute(vec![data])
            .await
            .unwrap(),
        Fragmentation::MultiFragment => {
            let params = WriteParams {
                max_rows_per_file: 10,
                ..Default::default()
            };
            InsertBuilder::new("memory://")
                .with_params(&params)
                .execute(vec![data])
                .await
                .unwrap()
        }
    };

    // Apply deletions
    apply_deletions(&mut ds, deletion_state).await;

    // Create index if specified
    if let Some(idx_type) = index_type {
        use lance_index::scalar::ScalarIndexParams;
        let params = ScalarIndexParams::default();
        ds.create_index(&["value"], idx_type, None, &params, false)
            .await
            .unwrap();
    }

    ds
}

/// Create a record batch that has 60 rows and two columns: id (incremental int32)
/// and value (which should match DataType).
///
/// The values in `value` should include edge cases for the specific data type:
/// * Nulls for all types
/// * For floats: NaN, Infinity, +/-0
/// * For integers: min / max
/// * For lists: nullability and empty values at all levels
fn test_batch(data_type: &DataType) -> RecordBatch {
    let num_rows = 60;
    let id: ArrayRef = Arc::new(Int32Array::from_iter(0..num_rows as i32));

    // For now, create a simple value column based on data type
    // TODO: Add edge cases for each type
    let value: ArrayRef = match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from_iter((0..num_rows).map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some(i % 2 == 0)
            }
        }))),
        DataType::Int32 => Arc::new(Int32Array::from_iter((0..num_rows).map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some(i as i32)
            }
        }))),
        _ => todo!("Implement test data generation for {:?}", data_type),
    };

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", data_type.clone(), true),
    ]);

    RecordBatch::try_new(Arc::new(schema), vec![id, value]).unwrap()
}

async fn apply_deletions(dataset: &mut Dataset, state: DeletionState) {
    match state {
        DeletionState::NoDeletions => {}
        DeletionState::DeleteOdd => {
            dataset.delete("id % 2 = 1").await.unwrap();
        }
        DeletionState::DeleteEven => {
            dataset.delete("id % 2 = 0").await.unwrap();
        }
    }
}

#[tokio::test]
async fn test_query_bool() {
    // TODO: pull out data generator (so it can be re-used an inspected).
    let original = gen_batch()
        .col("id", array::sequence_i32(0, 60))
        .col("value", array::cycle_bool(vec![true, false]).with_nulls(2))
        .into_batch_rows(RowCount::new(60))
        .unwrap();
    test_states(
        // Rename: for_states_and_indices()
        TABLE_STATES,
        [None, Some(IndexType::Bitmap), Some(IndexType::BTree)],
        |ds: Dataset, original: RecordBatch| async move {
            test_scan(&original, &ds).await;
            test_take(&original, &ds).await;
            test_filter(&original, &ds, "value").await;
            test_filter(&original, &ds, "!value").await;
        },
    )
    .await
}

async fn test_scan(_original: &RecordBatch, _ds: &Dataset) {
    todo!("validate that if you scan ds, then sort by id, you get original back.")
}

async fn test_take(_original: &RecordBatch, _ds: &Dataset) {
    todo!("generate a few sets of ids and validate we can call take against the RB and the DS and get the same result.");
}

async fn test_filter(_original: &RecordBatch, _ds: &Dataset, _predicate: &str) {
    todo!("Scan ds with the predicate");
}
