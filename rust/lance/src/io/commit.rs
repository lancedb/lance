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

//! Trait for commit implementations.
//!
//! In Lance, a transaction is committed by writing the next manifest file.
//! However, care should be taken to ensure that the manifest file is written
//! only once, even if there are concurrent writers. Different stores have
//! different abilities to handle concurrent writes, so a trait is provided
//! to allow for different implementations.
//!
//! The trait [CommitHandler] can be implemented to provide different commit
//! strategies. The default implementation for most object stores is
//! [RenameCommitHandler], which writes the manifest to a temporary path, then
//! renames the temporary path to the final path if no object already exists
//! at the final path. This is an atomic operation in most object stores, but
//! not in AWS S3. So for AWS S3, the default commit handler is
//! [UnsafeCommitHandler], which writes the manifest to the final path without
//! any checks.
//!
//! When providing your own commit handler, most often you are implementing in
//! terms of a lock. The trait [CommitLock] can be implemented as a simpler
//! alternative to [CommitHandler].

use std::fmt::Debug;
use std::future;
use std::sync::Arc;

#[cfg(feature = "dynamodb")]
pub(crate) mod dynamodb;
pub(crate) mod external_manifest;

use futures::future::Either;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::{
    format::{pb, Index, Manifest},
    io::commit::{CommitError, CommitHandler, ManifestWriter},
    Error, Result,
};
use object_store::path::Path;
use prost::Message;
use snafu::{location, Location};

use super::deletion::read_deletion_file;
use super::ObjectStore;
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{write_manifest_file, ManifestWriteConfig};
use crate::format::{DeletionFile, Fragment};
use crate::Dataset;

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const MANIFEST_EXTENSION: &str = "manifest";

/// Get the manifest file path for a version.
fn manifest_path(base: &Path, version: u64) -> Path {
    base.child(VERSIONS_DIR)
        .child(format!("{version}.{MANIFEST_EXTENSION}"))
}

fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
}

/// Get the latest manifest path
async fn current_manifest_path(object_store: &ObjectStore, base: &Path) -> Result<Path> {
    // TODO: list gives us the size, so we could also return the size of the manifest.
    // That avoids a HEAD request later.

    // We use `list_with_delimiter` to avoid listing the contents of child directories.
    let manifest_files = object_store
        .inner
        .list_with_delimiter(Some(&base.child(VERSIONS_DIR)))
        .await?;

    let current = manifest_files
        .objects
        .into_iter()
        .map(|meta| meta.location)
        .filter(|path| {
            path.filename().is_some() && path.filename().unwrap().ends_with(MANIFEST_EXTENSION)
        })
        .filter_map(|path| {
            let version = path
                .filename()
                .unwrap()
                .split_once('.')
                .and_then(|(version_str, _)| version_str.parse::<u64>().ok())?;
            Some((version, path))
        })
        .max_by_key(|(version, _)| *version)
        .map(|(_, path)| path);

    if let Some(path) = current {
        Ok(path)
    } else {
        Err(Error::NotFound {
            uri: manifest_path(base, 1).to_string(),
            location: location!(),
        })
    }
}

fn make_staging_manifest_path(base: &Path) -> Result<Path> {
    let id = uuid::Uuid::new_v4().to_string();
    Path::parse(format!("{base}-{id}")).map_err(|e| crate::Error::IO {
        message: format!("failed to parse path: {}", e),
        location: location!(),
    })
}

async fn list_manifests<'a>(
    base_path: &Path,
    object_store: &'a ObjectStore,
) -> Result<BoxStream<'a, Result<Path>>> {
    let base_path = base_path.clone();
    Ok(object_store
        .read_dir_all(&base_path.child(VERSIONS_DIR), None)
        .await?
        .try_filter_map(|obj_meta| {
            if obj_meta.location.extension() == Some("manifest") {
                future::ready(Ok(Some(obj_meta.location)))
            } else {
                future::ready(Ok(None))
            }
        })
        .boxed())
}

async fn write_latest_manifest(
    from_path: &Path,
    base_path: &Path,
    object_store: &ObjectStore,
) -> Result<()> {
    let latest_path = latest_manifest_path(base_path);
    let staging_path = make_staging_manifest_path(from_path)?;
    object_store
        .inner
        .copy(from_path, &staging_path)
        .await
        .map_err(|err| CommitError::OtherError(err.into()))?;
    object_store
        .inner
        .rename(&staging_path, &latest_path)
        .await?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct CommitConfig {
    pub num_retries: u32,
    // TODO: add isolation_level
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self { num_retries: 5 }
    }
}

/// Read the transaction data from a transaction file.
async fn read_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) -> Result<Transaction> {
    let path = base_path.child("_transactions").child(transaction_file);
    let result = object_store.inner.get(&path).await?;
    let data = result.bytes().await?;
    let transaction = pb::Transaction::decode(data)?;
    (&transaction).try_into()
}

/// Write a transaction to a file and return the relative path.
async fn write_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction: &Transaction,
) -> Result<String> {
    let file_name = format!("{}-{}.txn", transaction.read_version, transaction.uuid);
    let path = base_path.child("_transactions").child(file_name.as_str());

    let message = pb::Transaction::from(transaction);
    let buf = message.encode_to_vec();
    object_store.inner.put(&path, buf.into()).await?;

    Ok(file_name)
}

fn check_transaction(
    transaction: &Transaction,
    other_version: u64,
    other_transaction: &Option<Transaction>,
) -> Result<()> {
    if other_transaction.is_none() {
        return Err(crate::Error::Internal {
            message: format!(
                "There was a conflicting transaction at version {}, \
                and it was missing transaction metadata.",
                other_version
            ),
        });
    }

    if transaction.conflicts_with(other_transaction.as_ref().unwrap()) {
        return Err(crate::Error::CommitConflict {
            version: other_version,
            source: format!(
                "There was a concurrent commit that conflicts with this one and it \
                cannot be automatically resolved. Please rerun the operation off the latest version \
                of the table.\n Transaction: {:?}\n Conflicting Transaction: {:?}",
                transaction, other_transaction
            )
            .into(),
        });
    }

    Ok(())
}

pub(crate) async fn commit_new_dataset(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
) -> Result<Manifest> {
    let transaction_file = write_transaction_file(object_store, base_path, transaction).await?;

    let (mut manifest, indices) =
        transaction.build_manifest(None, vec![], &transaction_file, write_config)?;

    write_manifest_file(
        object_store,
        base_path,
        &mut manifest,
        if indices.is_empty() {
            None
        } else {
            Some(indices.clone())
        },
        write_config,
    )
    .await?;

    Ok(manifest)
}

/// Update manifest with new metadata fields.
///
/// Fields such as `physical_rows` and `num_deleted_rows` may not have been
/// in older datasets. To bring these old manifests up-to-date, we add them here.
async fn migrate_manifest(dataset: &Dataset, manifest: &mut Manifest) -> Result<()> {
    if manifest.fragments.iter().all(|f| {
        f.physical_rows > 0
            && (f.deletion_file.is_none()
                || f.deletion_file
                    .as_ref()
                    .map(|d| d.num_deleted_rows)
                    .unwrap_or_default()
                    > 0)
    }) {
        return Ok(());
    }

    let dataset = Arc::new(dataset.clone());

    let new_fragments = futures::stream::iter(manifest.fragments.as_ref())
        .map(|fragment| async {
            let physical_rows = if fragment.physical_rows == 0 {
                let file_fragment = FileFragment::new(dataset.clone(), fragment.clone());
                Either::Left(async move { file_fragment.count_rows().await })
            } else {
                Either::Right(futures::future::ready(Ok(fragment.physical_rows)))
            };
            let num_deleted_rows = match &fragment.deletion_file {
                None => Either::Left(futures::future::ready(Ok(0))),
                Some(deletion_file) if deletion_file.num_deleted_rows > 0 => {
                    Either::Left(futures::future::ready(Ok(deletion_file.num_deleted_rows)))
                }
                Some(_) => Either::Right(async {
                    let deletion_vector =
                        read_deletion_file(&dataset.base, fragment, dataset.object_store()).await?;
                    if let Some(deletion_vector) = deletion_vector {
                        Ok(deletion_vector.len())
                    } else {
                        Ok(0)
                    }
                }),
            };

            let (physical_rows, num_deleted_rows) =
                futures::future::try_join(physical_rows, num_deleted_rows).await?;

            let deletion_file = fragment
                .deletion_file
                .as_ref()
                .map(|deletion_file| DeletionFile {
                    num_deleted_rows,
                    ..deletion_file.clone()
                });

            Ok::<_, Error>(Fragment {
                physical_rows,
                deletion_file,
                ..fragment.clone()
            })
        })
        .buffered(num_cpus::get() * 2)
        .boxed();

    manifest.fragments = Arc::new(new_fragments.try_collect().await?);

    Ok(())
}

/// Update indices with new fields.
///
/// Indices might be missing `fragment_bitmap`, so this function will add it.
async fn migrate_indices(dataset: &Dataset, indices: &mut [Index]) -> Result<()> {
    // Early return so we have a fast path if they are already migrated.
    if indices.iter().all(|i| i.fragment_bitmap.is_some()) {
        return Ok(());
    }

    for index in indices {
        if index.fragment_bitmap.is_none() {
            // Load the read version of the index
            let old_version = match dataset.checkout_version(index.dataset_version).await {
                Ok(dataset) => dataset,
                // If the version doesn't exist anymore, skip it.
                Err(crate::Error::DatasetNotFound { .. }) => continue,
                // Any other error we return.
                Err(e) => return Err(e),
            };
            index.fragment_bitmap = Some(
                old_version
                    .get_fragments()
                    .iter()
                    .map(|f| f.id() as u32)
                    .collect(),
            );
        }
    }

    Ok(())
}

/// Attempt to commit a transaction, with retries and conflict resolution.
pub(crate) async fn commit_transaction(
    dataset: &Dataset,
    object_store: &ObjectStore,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    commit_config: &CommitConfig,
) -> Result<Manifest> {
    // Note: object_store has been configured with WriteParams, but dataset.object_store()
    // has not necessarily. So for anything involving writing, use `object_store`.
    let transaction_file = write_transaction_file(object_store, &dataset.base, transaction).await?;

    let mut dataset = dataset.clone();
    // First, get all transactions since read_version
    let mut other_transactions = Vec::new();
    let mut version = transaction.read_version;
    loop {
        version += 1;
        match dataset.checkout_version(version).await {
            Ok(next_dataset) => {
                let other_txn = if let Some(txn_file) = &next_dataset.manifest.transaction_file {
                    Some(read_transaction_file(object_store, &next_dataset.base, txn_file).await?)
                } else {
                    None
                };
                other_transactions.push(other_txn);
                dataset = next_dataset;
            }
            Err(crate::Error::NotFound { .. }) | Err(crate::Error::DatasetNotFound { .. }) => {
                break;
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    let mut target_version = version;

    // If any of them conflict with the transaction, return an error
    for (version_offset, other_transaction) in other_transactions.iter().enumerate() {
        let other_version = transaction.read_version + version_offset as u64 + 1;
        check_transaction(transaction, other_version, other_transaction)?;
    }

    for _ in 0..commit_config.num_retries {
        // Build an up-to-date manifest from the transaction and current manifest
        let (mut manifest, mut indices) = match transaction.operation {
            Operation::Restore { version } => {
                Transaction::restore_old_manifest(
                    object_store,
                    &dataset.base,
                    version,
                    write_config,
                    &transaction_file,
                )
                .await?
            }
            _ => transaction.build_manifest(
                Some(dataset.manifest.as_ref()),
                dataset.load_indices().await?,
                &transaction_file,
                write_config,
            )?,
        };

        manifest.version = target_version;

        migrate_manifest(&dataset, &mut manifest).await?;

        migrate_indices(&dataset, &mut indices).await?;

        // Try to commit the manifest
        let result = write_manifest_file(
            object_store,
            &dataset.base,
            &mut manifest,
            if indices.is_empty() {
                None
            } else {
                Some(indices.clone())
            },
            write_config,
        )
        .await;

        match result {
            Ok(()) => {
                return Ok(manifest);
            }
            Err(CommitError::CommitConflict) => {
                // See if we can retry the commit
                dataset = dataset.checkout_version(target_version).await?;

                let other_transaction =
                    if let Some(txn_file) = dataset.manifest.transaction_file.as_ref() {
                        Some(read_transaction_file(object_store, &dataset.base, txn_file).await?)
                    } else {
                        None
                    };
                check_transaction(transaction, target_version, &other_transaction)?;
                target_version += 1;
            }
            Err(CommitError::OtherError(err)) => {
                // If other error, return
                return Err(err);
            }
        }
    }

    Err(crate::Error::CommitConflict {
        version: target_version,
        source: format!(
            "Failed to commit the transaction after {} retries.",
            commit_config.num_retries
        )
        .into(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    use arrow_array::{Int32Array, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::future::join_all;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;

    use super::*;

    use crate::dataset::{transaction::Operation, WriteMode, WriteParams};
    use crate::index::{vector::VectorIndexParams, DatasetIndexExt, IndexType};
    use crate::io::object_store::ObjectStoreParams;
    use crate::Dataset;

    async fn test_commit_handler(handler: Arc<dyn CommitHandler>, should_succeed: bool) {
        // Create a dataset, passing handler as commit handler
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            DataType::Int64,
            false,
        )]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data)], schema);

        let options = WriteParams {
            store_params: Some(ObjectStoreParams {
                commit_handler: Some(handler),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://test", Some(options))
            .await
            .unwrap();

        // Create 10 concurrent tasks to write into the table
        // Record how many succeed and how many fail
        let tasks = (0..10).map(|_| {
            let mut dataset = dataset.clone();
            tokio::task::spawn(async move {
                dataset
                    .delete("x = 2")
                    .await
                    .map(|_| dataset.manifest.version)
            })
        });

        let task_results: Vec<Option<u64>> = join_all(tasks)
            .await
            .iter()
            .map(|res| match res {
                Ok(Ok(version)) => Some(*version),
                _ => None,
            })
            .collect();

        let num_successes = task_results.iter().filter(|x| x.is_some()).count();
        let distinct_results: HashSet<_> = task_results.iter().filter_map(|x| x.as_ref()).collect();

        if should_succeed {
            assert_eq!(
                num_successes,
                distinct_results.len(),
                "Expected no two tasks to succeed for the same version. Got {:?}",
                task_results
            );
        } else {
            // All we can promise here is at least one tasks succeeds, but multiple
            // could in theory.
            assert!(num_successes >= distinct_results.len(),);
        }
    }

    #[tokio::test]
    async fn test_rename_commit_handler() {
        // Rename is default for memory
        let handler = Arc::new(RenameCommitHandler);
        test_commit_handler(handler, true).await;
    }

    #[tokio::test]
    async fn test_custom_commit() {
        #[derive(Debug)]
        struct CustomCommitHandler {
            locked_version: Arc<Mutex<Option<u64>>>,
        }

        struct CustomCommitLease {
            version: u64,
            locked_version: Arc<Mutex<Option<u64>>>,
        }

        #[async_trait::async_trait]
        impl CommitLock for CustomCommitHandler {
            type Lease = CustomCommitLease;

            async fn lock(&self, version: u64) -> std::result::Result<Self::Lease, CommitError> {
                let mut locked_version = self.locked_version.lock().unwrap();
                if locked_version.is_some() {
                    // Already locked
                    return Err(CommitError::CommitConflict);
                }

                // Lock the version
                *locked_version = Some(version);

                Ok(CustomCommitLease {
                    version,
                    locked_version: self.locked_version.clone(),
                })
            }
        }

        #[async_trait::async_trait]
        impl CommitLease for CustomCommitLease {
            async fn release(&self, _success: bool) -> std::result::Result<(), CommitError> {
                let mut locked_version = self.locked_version.lock().unwrap();
                if *locked_version != Some(self.version) {
                    // Already released
                    return Err(CommitError::CommitConflict);
                }

                // Release the version
                *locked_version = None;

                Ok(())
            }
        }

        let locked_version = Arc::new(Mutex::new(None));
        let handler = Arc::new(CustomCommitHandler { locked_version });
        test_commit_handler(handler, true).await;
    }

    #[tokio::test]
    async fn test_unsafe_commit_handler() {
        let handler = Arc::new(UnsafeCommitHandler);
        test_commit_handler(handler, false).await;
    }

    #[tokio::test]
    async fn test_roundtrip_transaction_file() {
        let object_store = ObjectStore::memory();
        let base_path = Path::from("test");
        let transaction = Transaction::new(
            42,
            Operation::Append { fragments: vec![] },
            Some("hello world".to_string()),
        );

        let file_name = write_transaction_file(&object_store, &base_path, &transaction)
            .await
            .unwrap();
        let read_transaction = read_transaction_file(&object_store, &base_path, &file_name)
            .await
            .unwrap();

        assert_eq!(transaction.read_version, read_transaction.read_version);
        assert_eq!(transaction.uuid, read_transaction.uuid);
        assert!(matches!(
            read_transaction.operation,
            Operation::Append { .. }
        ));
        assert_eq!(transaction.tag, read_transaction.tag);
    }

    #[tokio::test]
    async fn test_concurrent_create_index() {
        // Create a table with two vector columns
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let dimension = 16;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(
                "vector1",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
            Field::new(
                "vector2",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));
        let float_arr = generate_random_array(512 * dimension as usize);
        let vectors = Arc::new(
            <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                float_arr, dimension,
            )
            .unwrap(),
        );
        let batches =
            vec![
                RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()])
                    .unwrap(),
            ];

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        // From initial version, concurrently call create index 3 times,
        // two of which will be for the same column.
        let params = VectorIndexParams::ivf_pq(10, 8, 2, false, MetricType::L2, 50);
        let futures: Vec<_> = ["vector1", "vector1", "vector2"]
            .iter()
            .map(|col_name| {
                let mut dataset = dataset.clone();
                let params = params.clone();
                tokio::spawn(async move {
                    dataset
                        .create_index(&[col_name], IndexType::Vector, None, &params, true)
                        .await
                })
            })
            .collect();

        let results = join_all(futures).await;
        for result in results {
            assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
        }

        // Validate that each version has the anticipated number of indexes
        let dataset = dataset.checkout_version(1).await.unwrap();
        assert!(dataset.load_indices().await.unwrap().is_empty());

        let dataset = dataset.checkout_version(2).await.unwrap();
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

        let dataset = dataset.checkout_version(3).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty() && indices.len() <= 2);

        // At this point, we have created two indices. If they are both for the same column,
        // it must be vector1 and not vector2.
        if indices.len() == 2 {
            let mut fields: Vec<i32> = indices.iter().flat_map(|i| i.fields.clone()).collect();
            fields.sort();
            assert_eq!(fields, vec![0, 1]);
        } else {
            assert_eq!(indices[0].fields, vec![0]);
        }

        let dataset = dataset.checkout_version(4).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
        let mut fields: Vec<i32> = indices.iter().flat_map(|i| i.fields.clone()).collect();
        fields.sort();
        assert_eq!(fields, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        for write_mode in [WriteMode::Append, WriteMode::Overwrite] {
            // Create an empty table
            let test_dir = tempfile::tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "i",
                DataType::Int32,
                false,
            )]));

            let dataset = Dataset::write(
                RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone()),
                test_uri,
                None,
            )
            .await
            .unwrap();

            // Make some sample data
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .unwrap();

            // Write data concurrently in 5 tasks
            let futures: Vec<_> = (0..5)
                .map(|_| {
                    let batch = batch.clone();
                    let schema = schema.clone();
                    let uri = test_uri.to_string();
                    tokio::spawn(async move {
                        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
                        Dataset::write(
                            reader,
                            &uri,
                            Some(WriteParams {
                                mode: write_mode,
                                ..Default::default()
                            }),
                        )
                        .await
                    })
                })
                .collect();
            let results = join_all(futures).await;

            // Assert all succeeded
            for result in results {
                assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
            }

            // Assert final fragments and versions expected
            let dataset = dataset.checkout_version(6).await.unwrap();

            match write_mode {
                WriteMode::Append => {
                    assert_eq!(dataset.get_fragments().len(), 5);
                }
                WriteMode::Overwrite => {
                    assert_eq!(dataset.get_fragments().len(), 1);
                }
                _ => unreachable!(),
            }

            dataset.validate().await.unwrap()
        }
    }
}
