// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::panic::AssertUnwindSafe;

use arrow_array::RecordBatch;
use futures::FutureExt;
use lance::Dataset;
use lance_index::IndexType;

#[derive(Clone, Copy, Debug)]
pub enum Fragmentation {
    /// All data in a single file.
    SingleFragment,
    /// Data is spread across multiple fragments, one file per fragment.
    MultiFragment,
}

#[derive(Clone, Copy, Debug)]
pub enum DeletionState {
    /// No deletions are applied.
    NoDeletions,
    /// Delete odd rows.
    DeleteOdd,
    /// Delete even rows.
    DeleteEven,
}

pub struct DatasetTestCases {
    original: RecordBatch,
    index_options: Vec<(String, Vec<Option<IndexType>>)>,
}

impl DatasetTestCases {
    pub fn from_data(original: RecordBatch) -> Self {
        Self {
            original,
            index_options: Vec::new(),
        }
    }

    pub fn with_index_types(
        mut self,
        column: impl Into<String>,
        index_types: impl IntoIterator<Item = Option<IndexType>>,
    ) -> Self {
        self.index_options
            .push((column.into(), index_types.into_iter().collect()));
        self
    }

    fn generate_index_combinations(&self) -> Vec<Vec<(&str, IndexType)>> {
        let mut combinations = Vec::new();
        for (column, index_types) in &self.index_options {
            for index_type in index_types {
                if let Some(index_type) = index_type {
                    combinations.push(vec![(column.as_str(), index_type.clone())]);
                }
            }
        }
        combinations
    }

    pub async fn run<F, Fut>(self, test_fn: F) -> Fut::Output
    where
        F: Fn(Dataset, RecordBatch) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        for fragmentation in [Fragmentation::SingleFragment, Fragmentation::MultiFragment] {
            for deletion in [
                DeletionState::NoDeletions,
                DeletionState::DeleteOdd,
                DeletionState::DeleteEven,
            ] {
                let index_combinations = self.generate_index_combinations();
                for indices in index_combinations {
                    let ds =
                        build_dataset(self.original.clone(), fragmentation, deletion, &indices);
                    let context = format!(
                        "fragmentation: {:?}, deletion: {:?}, index: {:?}",
                        fragmentation, deletion, indices
                    );
                    // Catch unwind so we can add test context to the panic.
                    AssertUnwindSafe(test_fn(ds, self.original.clone()))
                        .catch_unwind()
                        .await
                        .unwrap_or_else(|_| panic!("Test failed for {}", context));
                }
            }
        }
    }
}

/// Create an in-memory dataset with the given state and data.
///
/// The data in dataset will exactly match the `original` batch. (Extra rows are
/// created for the deleted rows created by `DeletionState`.)
fn build_dataset(
    original: RecordBatch,
    fragmentation: Fragmentation,
    deletion: DeletionState,
    indices: &[(&str, IndexType)],
) -> Dataset {
    todo!()
}
