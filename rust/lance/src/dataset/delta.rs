// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::transaction::Transaction;
use crate::Dataset;
use crate::Result;

/// The delta dataset between two versions of a dataset.
pub struct DatasetDelta {
    /// The base version number for comparison.
    pub(crate) begin_version: u64,
    /// The current version number.
    pub(crate) end_version: u64,
    /// Base path of the dataset.
    pub(crate) base_dataset: Dataset,
}

impl DatasetDelta {
    /// Listing the transactions between two versions.
    pub async fn diff_meta(&self) -> Result<Vec<Transaction>> {
        let mut transactions = Vec::new();
        for version in (self.begin_version + 1)..=self.end_version {
            let current_ds = self.base_dataset.checkout_version(version).await?;
            if let Some(tx) = current_ds.read_transaction().await? {
                transactions.push(tx);
            }
        }
        Ok(transactions)
    }
}

#[cfg(test)]
mod tests {

    use crate::dataset::transaction::Operation;
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::types::Int32Type;
    use lance_datagen::{array, BatchCount, RowCount};

    async fn create_test_dataset() -> Dataset {
        let data = lance_datagen::gen()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let write_params = WriteParams {
            ..Default::default()
        };
        Dataset::write(data, "memory://", Some(write_params.clone()))
            .await
            .unwrap()
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_diff_meta_no_transaction() {
        let ds = create_test_dataset().await;
        let result = ds.diff_meta(1).await;
        assert!(result.is_err());
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_diff_meta_single_transaction() {
        let mut ds = create_test_dataset().await;
        ds.delete("key = 5").await.unwrap();

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version: ds.version().version,
            base_dataset: ds.clone(),
        };
        let txs = delta_struct.diff_meta().await.unwrap();
        assert_eq!(txs.len(), 1);
        assert!(matches!(txs[0].operation, Operation::Delete { .. }));
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_diff_meta_multiple_transactions() {
        let mut ds = create_test_dataset().await;
        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version: ds.version().version,
            base_dataset: ds.clone(),
        };
        let txs = delta_struct.diff_meta().await.unwrap();
        assert_eq!(txs.len(), 2);
    }
}
