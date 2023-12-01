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

//! A task to clean up a lance dataset, removing files that are no longer
//! needed.
//!
//! Currently we try and be rather conservative about what we delete.
//!
//! The following types of files may be deleted by the cleanup function:
//!
//! * Old manifest files - If a manifest file is older than the threshold
//!   and is not the latest manifest then it will be deleted.
//! * Unreferenced data files - If a data file is not referenced by any
//!   fragment in a valid manifest file then it will be deleted.
//! * Unreferenced delete files - If a delete file is not referenced by
//!   any fragment in a valid manifest file then it will be deleted.
//! * Unreferenced index files - If an index file is not referenced by
//!   any valid manifest file then it will be deleted.
//!
//! It is also difficult to distinguish between a data/tx/idx file which was
//! leftover from an abandoned transaction and a data file which is part
//! of an ongoing operation (both will look like unreferenced data files).
//!
//! If the file is referenced by at least one manifest (even if that manifest
//! is old and being deleted) then we assume it is not part of an ongoing
//! operation and can be safely deleted.
//!
//! If the data is not referenced by any manifest then we look at the age of
//! the file.  If the file is at least 7 days old then we assume it is probably
//! not part of any ongoing operation and we will delete it.
//!
//! Otherwise we will leave the file unless delete_unverified is set to true.
//! (which should only be done if the caller can guarantee there are no updates
//! happening at the same time)

use chrono::{DateTime, Duration, Utc};
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::{
    format::{Index, Manifest},
    io::{
        deletion::deletion_file_path,
        object_store::ObjectStore,
        reader::{read_manifest, read_manifest_indexes},
    },
    Error, Result,
};
use object_store::path::Path;
use std::{
    collections::HashSet,
    future,
    sync::{Mutex, MutexGuard},
};

use crate::{utils::temporal::utc_now, Dataset};

#[derive(Clone, Debug, Default)]
struct ReferencedFiles {
    data_paths: HashSet<Path>,
    delete_paths: HashSet<Path>,
    tx_paths: HashSet<Path>,
    index_uuids: HashSet<String>,
}

#[derive(Clone, Debug, Default)]
pub struct RemovalStats {
    pub bytes_removed: u64,
    pub old_versions: u64,
}

fn remove_prefix(path: &Path, prefix: &Path) -> Path {
    let relative_parts = path.prefix_match(prefix);
    if relative_parts.is_none() {
        return path.clone();
    }
    Path::from_iter(relative_parts.unwrap())
}

#[derive(Clone, Debug)]
struct CleanupTask<'a> {
    dataset: &'a Dataset,
    /// Cleanup all versions before this time
    before: DateTime<Utc>,
    /// If true, delete unverified data files even if they are recent
    delete_unverified: bool,
}

/// Information about the dataset that we learn by inspecting all of the manifests
#[derive(Clone, Debug, Default)]
struct CleanupInspection {
    old_manifests: Vec<Path>,
    /// Referenced files are part of our working set
    referenced_files: ReferencedFiles,
    /// Verified files may or may not be part of the working set but they are
    /// referenced by at least one manifest file (potentially an old one) and
    /// so we know that they are not part of an ongoing operation.
    verified_files: ReferencedFiles,
}

/// If a file cannot be verified then it will only be deleted if it is at least
/// this many days old.
const UNVERIFIED_THRESHOLD_DAYS: i64 = 7;

impl<'a> CleanupTask<'a> {
    fn new(dataset: &'a Dataset, before: DateTime<Utc>, delete_unverified: bool) -> Self {
        Self {
            dataset,
            before,
            delete_unverified,
        }
    }

    async fn run(self) -> Result<RemovalStats> {
        // First we process all manifest files in parallel to figure
        // out which files are referenced by valid manifests
        let inspection = self.process_manifests().await?;
        self.delete_unreferenced_files(inspection).await
    }

    async fn process_manifests(&'a self) -> Result<CleanupInspection> {
        let inspection = Mutex::new(CleanupInspection::default());
        self.dataset
            .object_store
            .commit_handler
            .list_manifests(&self.dataset.base, &self.dataset.object_store.inner)
            .await?
            .try_for_each_concurrent(num_cpus::get(), |path| {
                self.process_manifest_file(path, &inspection)
            })
            .await?;
        Ok(inspection.into_inner().unwrap())
    }

    async fn process_manifest_file(
        &self,
        path: Path,
        inspection: &Mutex<CleanupInspection>,
    ) -> Result<()> {
        // TODO: We can't cleanup invalid manifests.  There is no way to distinguish
        // between an invalid manifest and a temporary I/O error.  It's also not safe
        // to ignore a manifest error because if it is a temporary I/O error and we
        // ignore it then we might delete valid data files thinking they are not
        // referenced.

        let manifest = read_manifest(&self.dataset.object_store, &path).await?;
        let dataset_version = self.dataset.version().version;
        // Don't delete the latest version, even if it is old.  Also don't delete manifests
        // if their version is newer than the dataset version.  These are either in-progress
        // or newly added since we started.
        let is_latest = dataset_version <= manifest.version;
        let in_working_set = is_latest || manifest.timestamp() >= self.before;
        let indexes = read_manifest_indexes(&self.dataset.object_store, &path, &manifest).await?;

        let mut inspection = inspection.lock().unwrap();

        self.process_manifest(&manifest, &indexes, in_working_set, &mut inspection)?;
        if !in_working_set {
            inspection.old_manifests.push(path.clone());
        }
        Ok(())
    }

    fn process_manifest(
        &self,
        manifest: &Manifest,
        indexes: &Vec<Index>,
        in_working_set: bool,
        inspection: &mut MutexGuard<CleanupInspection>,
    ) -> Result<()> {
        // If this part of our working set then update referenced_files.  Otherwise, just mark the
        // file as verified.
        let referenced_files = if in_working_set {
            &mut inspection.referenced_files
        } else {
            &mut inspection.verified_files
        };

        for fragment in manifest.fragments.iter() {
            for file in fragment.files.iter() {
                let full_data_path = self.dataset.data_dir().child(file.path.as_str());
                let relative_data_path = remove_prefix(&full_data_path, &self.dataset.base);
                referenced_files.data_paths.insert(relative_data_path);
            }
            let delpath = fragment
                .deletion_file
                .as_ref()
                .map(|delfile| deletion_file_path(&self.dataset.base, fragment.id, delfile));
            if let Some(delpath) = delpath {
                let relative_path = remove_prefix(&delpath, &self.dataset.base);
                referenced_files.delete_paths.insert(relative_path);
            }
        }
        if let Some(relative_tx_path) = &manifest.transaction_file {
            referenced_files
                .tx_paths
                .insert(Path::parse("_transactions")?.child(relative_tx_path.as_str()));
        }

        for index in indexes {
            let uuid_str = index.uuid.to_string();
            referenced_files.index_uuids.insert(uuid_str);
        }
        Ok(())
    }

    async fn delete_unreferenced_files(
        &self,
        inspection: CleanupInspection,
    ) -> Result<RemovalStats> {
        let removal_stats = Mutex::new(RemovalStats::default());
        let verification_threshold = utc_now() - Duration::days(UNVERIFIED_THRESHOLD_DAYS);
        let unreferenced_paths = self
            .dataset
            .object_store
            .read_dir_all(&self.dataset.base, Some(self.before))
            .await?
            .try_filter_map(|obj_meta| {
                // If a file is new-ish then it might be part of an ongoing operation and so we only
                // delete it if we can verify it is part of an old version.
                let maybe_in_progress =
                    !self.delete_unverified && obj_meta.last_modified >= verification_threshold;
                let path_to_remove =
                    self.path_if_not_referenced(obj_meta.location, maybe_in_progress, &inspection);
                if matches!(path_to_remove, Ok(Some(..))) {
                    removal_stats.lock().unwrap().bytes_removed += obj_meta.size as u64;
                }
                future::ready(path_to_remove)
            })
            .boxed();

        let old_manifests = inspection.old_manifests.clone();
        let num_old_manifests = old_manifests.len();

        // Ideally this collect shouldn't be needed here but it seems necessary
        // to avoid https://github.com/rust-lang/rust/issues/102211
        let manifest_bytes_removed = stream::iter(&old_manifests)
            .map(|path| self.dataset.object_store.size(path))
            .collect::<Vec<_>>()
            .await;
        let manifest_bytes_removed = stream::iter(manifest_bytes_removed)
            .buffer_unordered(num_cpus::get())
            .try_fold(0, |acc, size| async move { Ok(acc + (size as u64)) })
            .await;

        let old_manifests_stream = stream::iter(old_manifests).map(Result::<Path>::Ok).boxed();
        let all_paths_to_remove =
            stream::iter(vec![unreferenced_paths, old_manifests_stream]).flatten();

        let delete_fut = self
            .dataset
            .object_store
            .remove_stream(all_paths_to_remove.boxed())
            .try_for_each(|_| future::ready(Ok(())));

        delete_fut.await?;

        let mut removal_stats = removal_stats.into_inner().unwrap();
        removal_stats.old_versions = num_old_manifests as u64;
        removal_stats.bytes_removed += manifest_bytes_removed?;
        Ok(removal_stats)
    }

    fn path_if_not_referenced(
        &self,
        path: Path,
        maybe_in_progress: bool,
        inspection: &CleanupInspection,
    ) -> Result<Option<Path>> {
        let relative_path = remove_prefix(&path, &self.dataset.base);
        if relative_path.as_ref().starts_with("_versions/.tmp") {
            // This is a temporary manifest file.
            //
            // If the file is old (or the user has verified there are no writes in progress) then
            // it must be leftover from a failed tx.
            if maybe_in_progress {
                return Ok(None);
            } else {
                return Ok(Some(path));
            }
        }
        if relative_path.as_ref().starts_with("_indices") {
            // Indices are referenced by UUID so we need to examine the UUID
            // portion of the path.
            if let Some(uuid) = relative_path.parts().nth(1) {
                if inspection
                    .referenced_files
                    .index_uuids
                    .contains(uuid.as_ref())
                {
                    return Ok(None);
                } else if !maybe_in_progress
                    || inspection
                        .verified_files
                        .index_uuids
                        .contains(uuid.as_ref())
                {
                    return Ok(Some(path));
                }
            } else {
                return Ok(None);
            }
        }
        match path.extension() {
            Some("lance") => {
                if relative_path.as_ref().starts_with("data") {
                    if inspection
                        .referenced_files
                        .data_paths
                        .contains(&relative_path)
                    {
                        Ok(None)
                    } else if !maybe_in_progress
                        || inspection
                            .verified_files
                            .data_paths
                            .contains(&relative_path)
                    {
                        Ok(Some(path))
                    } else {
                        Ok(None)
                    }
                } else {
                    // If a .lance file isn't in the data directory we err on the side of leaving it alone
                    Ok(None)
                }
            }
            Some("manifest") => {
                // We already scanned the manifest files
                Ok(None)
            }
            Some("arrow") | Some("bin") => {
                if relative_path.as_ref().starts_with("_deletions") {
                    if inspection
                        .referenced_files
                        .delete_paths
                        .contains(&relative_path)
                    {
                        Ok(None)
                    } else if !maybe_in_progress
                        || inspection
                            .verified_files
                            .delete_paths
                            .contains(&relative_path)
                    {
                        Ok(Some(path))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            Some("txn") => {
                if relative_path.as_ref().starts_with("_transactions") {
                    if inspection
                        .referenced_files
                        .tx_paths
                        .contains(&relative_path)
                    {
                        Ok(None)
                    } else if !maybe_in_progress
                        || inspection.verified_files.tx_paths.contains(&relative_path)
                    {
                        Ok(Some(path))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

/// Deletes old versions of a dataset, removing files that are no longer
/// needed.
///
/// This function will remove old manifest files, data files, indexes,
/// delete files, and transaction files.
///
/// It will only remove files that are not referenced by any valid manifest.
///
/// The latest manifest is always considered valid and will not be removed
/// even if it is older than the `before` parameter.
///
/// The `before` parameter must be at least 7 days before the current date.
pub async fn cleanup_old_versions(
    dataset: &Dataset,
    before: DateTime<Utc>,
    delete_unverified: Option<bool>,
) -> Result<RemovalStats> {
    let cleanup = CleanupTask::new(dataset, before, delete_unverified.unwrap_or(false));
    cleanup.run().await
}

/// Force cleanup of specific partial writes.
///
/// These files can be cleaned up easily with [cleanup_old_versions()] after 7 days,
/// but if you know specific partial writes have been made, you can call this
/// function to clean them up immediately.
///
/// To find partial writes, you can use the
/// [crate::dataset::progress::WriteFragmentProgress] trait to track which files
/// have been started but never finished.
pub async fn cleanup_partial_writes(
    store: &ObjectStore,
    objects: impl IntoIterator<Item = (&Path, &String)>,
) -> Result<()> {
    futures::stream::iter(objects)
        .map(Ok)
        .try_for_each_concurrent(num_cpus::get() * 2, |(path, multipart_id)| async move {
            let path: Path = store
                .base_path()
                .child("data")
                .parts()
                .chain(path.parts())
                .collect();
            match store.inner.abort_multipart(&path, multipart_id).await {
                Ok(_) => Ok(()),
                // We don't care if it's not there.
                // TODO: once this issue is addressed, we should just use the error
                // variant. https://github.com/apache/arrow-rs/issues/4749
                // Err(object_store::Error::NotFound { .. }) => {
                Err(e)
                    if e.to_string().contains("No such file or directory")
                        || e.to_string().contains("cannot find the file") =>
                {
                    log::warn!("Partial write not found: {} {}", path, multipart_id);
                    Ok(())
                }
                Err(e) => Err(Error::from(e)),
            }
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow_array::{RecordBatchIterator, RecordBatchReader};
    use chrono::Duration;
    use lance_core::{
        utils::testing::{MockClock, ProxyObjectStore, ProxyObjectStorePolicy},
        Error, Result,
    };
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::{some_batch, BatchGenerator, IncrementingInt32};
    use snafu::{location, Location};
    use tokio::io::AsyncWriteExt;

    use crate::{
        dataset::{builder::DatasetBuilder, ReadParams, WriteMode, WriteParams},
        index::{
            vector::{StageParams, VectorIndexParams},
            DatasetIndexExt,
        },
        io::{
            object_store::{ObjectStoreParams, WrappingObjectStore},
            ObjectStore,
        },
    };
    use all_asserts::{assert_gt, assert_lt};
    use tempfile::{tempdir, TempDir};

    use super::*;

    #[derive(Debug)]
    struct MockObjectStore {
        policy: Arc<Mutex<ProxyObjectStorePolicy>>,
        last_modified_times: Arc<Mutex<HashMap<Path, DateTime<Utc>>>>,
    }

    impl WrappingObjectStore for MockObjectStore {
        fn wrap(
            &self,
            original: Arc<dyn object_store::ObjectStore>,
        ) -> Arc<dyn object_store::ObjectStore> {
            Arc::new(ProxyObjectStore::new(original, self.policy.clone()))
        }
    }

    impl MockObjectStore {
        pub(crate) fn new() -> Self {
            let instance = Self {
                policy: Arc::new(Mutex::new(ProxyObjectStorePolicy::new())),
                last_modified_times: Arc::new(Mutex::new(HashMap::new())),
            };
            instance.add_timestamp_policy();
            instance
        }

        fn add_timestamp_policy(&self) {
            let mut policy = self.policy.lock().unwrap();
            let times_map = self.last_modified_times.clone();
            policy.set_before_policy(
                "record_file_time",
                Arc::new(move |_, path| {
                    let mut times_map = times_map.lock().unwrap();
                    times_map.insert(path.clone(), utc_now());
                    Ok(())
                }),
            );
            let times_map = self.last_modified_times.clone();
            policy.set_obj_meta_policy(
                "add_recorded_file_time",
                Arc::new(move |_, meta| {
                    let mut meta = meta;
                    if let Some(recorded) = times_map.lock().unwrap().get(&meta.location) {
                        meta.last_modified = *recorded;
                    }
                    Ok(meta)
                }),
            );
        }
    }

    #[derive(Debug, PartialEq)]
    struct FileCounts {
        num_data_files: usize,
        num_manifest_files: usize,
        num_index_files: usize,
        num_delete_files: usize,
        num_tx_files: usize,
        num_bytes: u64,
    }

    struct MockDatasetFixture<'a> {
        // This is a temporary directory that will be deleted when the fixture
        // is dropped
        _tmpdir: TempDir,
        dataset_path: String,
        mock_store: Arc<MockObjectStore>,
        pub clock: MockClock<'a>,
    }

    impl<'a> MockDatasetFixture<'a> {
        fn try_new() -> Result<Self> {
            let tmpdir = tempdir()?;
            // let tmpdir_uri = to_obj_store_uri(tmpdir.path())?;
            let tmpdir_path = tmpdir.path().as_os_str().to_str().unwrap().to_owned();
            Ok(Self {
                _tmpdir: tmpdir,
                dataset_path: format!("{}/my_db", tmpdir_path),
                mock_store: Arc::new(MockObjectStore::new()),
                clock: MockClock::new(),
            })
        }

        fn os_params(&self) -> ObjectStoreParams {
            ObjectStoreParams {
                object_store_wrapper: Some(self.mock_store.clone()),
                ..Default::default()
            }
        }

        async fn write_data_impl(
            &self,
            data: impl RecordBatchReader + Send + 'static,
            mode: WriteMode,
        ) -> Result<()> {
            Dataset::write(
                data,
                &self.dataset_path,
                Some(WriteParams {
                    store_params: Some(self.os_params()),
                    mode,
                    ..Default::default()
                }),
            )
            .await?;
            Ok(())
        }

        async fn write_some_data_impl(&self, mode: WriteMode) -> Result<()> {
            self.write_data_impl(some_batch(), mode).await?;
            Ok(())
        }

        async fn create_some_data(&self) -> Result<()> {
            self.write_some_data_impl(WriteMode::Create).await
        }

        async fn overwrite_some_data(&self) -> Result<()> {
            self.write_some_data_impl(WriteMode::Overwrite).await
        }

        async fn append_some_data(&self) -> Result<()> {
            self.write_some_data_impl(WriteMode::Append).await
        }

        async fn create_with_data(
            &self,
            data: impl RecordBatchReader + Send + 'static,
        ) -> Result<()> {
            self.write_data_impl(data, WriteMode::Create).await
        }

        async fn append_data(&self, data: impl RecordBatchReader + Send + 'static) -> Result<()> {
            self.write_data_impl(data, WriteMode::Append).await
        }

        async fn overwrite_data(
            &self,
            data: impl RecordBatchReader + Send + 'static,
        ) -> Result<()> {
            self.write_data_impl(data, WriteMode::Overwrite).await
        }

        async fn delete_data(&self, predicate: &str) -> Result<()> {
            let mut db = self.open().await?;
            db.delete(predicate).await?;
            Ok(())
        }

        async fn create_some_index(&self) -> Result<()> {
            let mut db = self.open().await?;
            let index_params = Box::new(VectorIndexParams {
                stages: vec![StageParams::DiskANN(Default::default())],
                metric_type: MetricType::L2,
            });
            db.create_index(
                &["indexable"],
                IndexType::Vector,
                Some("some_index".to_owned()),
                &*index_params,
                false,
            )
            .await?;
            Ok(())
        }

        fn block_commits(&mut self) {
            let mut policy = self.mock_store.policy.lock().unwrap();
            policy.set_before_policy(
                "block_commit",
                Arc::new(|op, _| -> Result<()> {
                    if op.contains("copy") {
                        return Err(Error::Internal {
                            message: "Copy blocked".to_string(),
                            location: location!(),
                        });
                    }
                    Ok(())
                }),
            );
        }

        fn block_delete_manifest(&mut self) {
            let mut policy = self.mock_store.policy.lock().unwrap();
            policy.set_before_policy(
                "block_delete_manifest",
                Arc::new(|op, path| -> Result<()> {
                    if op.contains("delete") && path.extension() == Some("manifest") {
                        Err(Error::Internal {
                            message: "Delete manifest blocked".to_string(),
                            location: location!(),
                        })
                    } else {
                        Ok(())
                    }
                }),
            );
        }

        fn unblock_delete_manifest(&mut self) {
            let mut policy = self.mock_store.policy.lock().unwrap();
            policy.clear_before_policy("block_delete_manifest");
        }

        async fn run_cleanup(&self, before: DateTime<Utc>) -> Result<RemovalStats> {
            let db = self.open().await?;
            cleanup_old_versions(&db, before, None).await
        }

        async fn run_cleanup_with_override(
            &self,
            before: DateTime<Utc>,
            delete_unverified: Option<bool>,
        ) -> Result<RemovalStats> {
            let db = self.open().await?;
            cleanup_old_versions(&db, before, delete_unverified).await
        }

        async fn open(&self) -> Result<Box<Dataset>> {
            let ds = DatasetBuilder::from_uri(&self.dataset_path)
                .with_read_params(ReadParams {
                    store_options: Some(self.os_params()),
                    ..Default::default()
                })
                .load()
                .await?;
            Ok(Box::new(ds))
        }

        async fn count_files(&self) -> Result<FileCounts> {
            let (os, path) =
                ObjectStore::from_uri_and_params(&self.dataset_path, &self.os_params()).await?;
            let mut file_stream = os.read_dir_all(&path, None).await?;
            let mut file_count = FileCounts {
                num_data_files: 0,
                num_delete_files: 0,
                num_index_files: 0,
                num_manifest_files: 0,
                num_tx_files: 0,
                num_bytes: 0,
            };
            while let Some(path) = file_stream.try_next().await? {
                file_count.num_bytes += path.size as u64;
                match path.location.extension() {
                    Some("lance") => file_count.num_data_files += 1,
                    Some("manifest") => file_count.num_manifest_files += 1,
                    Some("arrow") | Some("bin") => file_count.num_delete_files += 1,
                    Some("idx") => file_count.num_index_files += 1,
                    Some("txn") => file_count.num_tx_files += 1,
                    _ => (),
                }
            }
            Ok(file_count)
        }

        async fn count_rows(&self) -> Result<usize> {
            let db = self.open().await?;
            let count = db.count_rows().await?;
            Ok(count)
        }
    }

    #[tokio::test]
    async fn cleanup_unreferenced_data_files() {
        // We should clean up data files that are only referenced
        // by old versions.  This can happen, for example, due to
        // an overwrite
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.overwrite_some_data().await.unwrap();

        fixture.clock.set_system_time(Duration::days(10));

        let before_count = fixture.count_files().await.unwrap();

        let removed = fixture
            .run_cleanup(utc_now() - Duration::days(8))
            .await
            .unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(removed.old_versions, 1);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        // There should be one less data file
        assert_lt!(after_count.num_data_files, before_count.num_data_files);
        // And one less manifest file
        assert_lt!(
            after_count.num_manifest_files,
            before_count.num_manifest_files
        );
        assert_lt!(after_count.num_tx_files, before_count.num_tx_files);

        // The latest manifest should still be there, even if it is older than
        // the given time.
        assert_gt!(after_count.num_manifest_files, 0);
        assert_gt!(after_count.num_data_files, 0);
        // We should keep referenced tx files
        assert_gt!(after_count.num_tx_files, 0);
    }

    #[tokio::test]
    async fn do_not_cleanup_newer_data() {
        // Even though an old manifest is removed the data files should
        // remain if they are still referenced by newer manifests
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.append_some_data().await.unwrap();
        fixture.append_some_data().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();

        // 3 versions (plus one extra latest.manifest)
        assert_eq!(before_count.num_data_files, 3);
        assert_eq!(before_count.num_manifest_files, 4);

        let before = utc_now() - Duration::days(7);
        let removed = fixture.run_cleanup(before).await.unwrap();

        let after_count = fixture.count_files().await.unwrap();

        assert_eq!(removed.old_versions, 1);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        // The data files should all remain since they are referenced by
        // the latest version
        assert_eq!(after_count.num_data_files, 3);
        // Only the oldest manifest file should be removed
        assert_eq!(after_count.num_manifest_files, 3);
        assert_eq!(after_count.num_tx_files, 2);
    }

    #[tokio::test]
    async fn cleanup_recent_verified_files() {
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.clock.set_system_time(Duration::seconds(1));
        fixture.overwrite_some_data().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_data_files, 2);
        assert_eq!(before_count.num_manifest_files, 3);

        // Not much time has passed but we can still delete the old manifest
        // and the related data files
        let before = utc_now();
        let removed = fixture.run_cleanup(before).await.unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(removed.old_versions, 1);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_manifest_files, 2);
    }

    #[tokio::test]
    async fn dont_cleanup_recent_unverified_files() {
        for (override_opt, old_files) in [
            (Some(false), false), // User provides false, files are new - do not delete
            (Some(true), false),  // User provides true, files are new - delete
            (None, true),         // Default, files are old - delete
            (None, false),        // Default, files are new - do not delete
        ] {
            let mut fixture = MockDatasetFixture::try_new().unwrap();
            fixture.create_some_data().await.unwrap();
            fixture.block_commits();
            assert!(fixture.append_some_data().await.is_err());

            let age = if old_files {
                Duration::days(UNVERIFIED_THRESHOLD_DAYS + 1)
            } else {
                Duration::days(UNVERIFIED_THRESHOLD_DAYS - 1)
            };
            fixture.clock.set_system_time(age);

            // The above created some unreferenced data files but, since they
            // are not referenced in any manifest, and 7 days has not passed, we
            // cannot safely delete them unless the user overrides the safety check

            let before_count = fixture.count_files().await.unwrap();
            assert_eq!(before_count.num_data_files, 2);
            assert_eq!(before_count.num_manifest_files, 2);

            let before = utc_now();
            let removed = fixture
                .run_cleanup_with_override(before, override_opt)
                .await
                .unwrap();

            let should_delete = override_opt.unwrap_or(false) || old_files;

            let after_count = fixture.count_files().await.unwrap();
            assert_eq!(removed.old_versions, 0);
            assert_eq!(
                removed.bytes_removed,
                before_count.num_bytes - after_count.num_bytes
            );

            if should_delete {
                assert_gt!(removed.bytes_removed, 0);
            } else {
                assert_eq!(removed.bytes_removed, 0);
            }
        }
    }

    #[tokio::test]
    async fn cleanup_old_index() {
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.create_some_index().await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.overwrite_some_data().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_index_files, 1);
        // Two user data files and one lance file for the index
        assert_eq!(before_count.num_data_files, 3);
        // Creating an index creates a new manifest so there are 4 total
        assert_eq!(before_count.num_manifest_files, 4);

        let before = utc_now() - Duration::days(8);
        let removed = fixture.run_cleanup(before).await.unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(removed.old_versions, 2);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        assert_eq!(after_count.num_index_files, 0);
        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_manifest_files, 2);
        assert_eq!(after_count.num_tx_files, 1);
    }

    #[tokio::test]
    async fn clean_old_delete_files() {
        let fixture = MockDatasetFixture::try_new().unwrap();
        let mut data_gen = BatchGenerator::new().col(Box::new(
            IncrementingInt32::new().named("filter_me".to_owned()),
        ));

        fixture.create_with_data(data_gen.batch(16)).await.unwrap();
        fixture.append_data(data_gen.batch(16)).await.unwrap();
        // This will keep some data from the appended file and should
        // completely remove the first file
        fixture.delete_data("filter_me < 20").await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.overwrite_data(data_gen.batch(16)).await.unwrap();
        // This will delete half of the last fragment
        fixture.delete_data("filter_me >= 40").await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_data_files, 3);
        assert_eq!(before_count.num_delete_files, 2);
        assert_eq!(before_count.num_manifest_files, 6);

        let before = utc_now() - Duration::days(8);
        let removed = fixture.run_cleanup(before).await.unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(removed.old_versions, 3);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_delete_files, 1);
        assert_eq!(after_count.num_manifest_files, 3);
        assert_eq!(after_count.num_tx_files, 2);

        // Ensure we can still read the dataset
        let row_count_after = fixture.count_rows().await.unwrap();
        assert_eq!(row_count_after, 8);
    }

    #[tokio::test]
    async fn dont_clean_index_data_files() {
        // Indexes have .lance files in them that are not referenced
        // by any fragment.  We need to make sure the cleanup routine
        // doesn't over-zealously delete these
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.create_some_data().await.unwrap();
        fixture.create_some_index().await.unwrap();

        let before_count = fixture.count_files().await.unwrap();
        let before = utc_now() - Duration::days(8);
        let removed = fixture.run_cleanup(before).await.unwrap();
        assert_eq!(removed.old_versions, 0);
        assert_eq!(removed.bytes_removed, 0);

        let after_count = fixture.count_files().await.unwrap();

        assert_eq!(before_count, after_count);
    }

    #[tokio::test]
    async fn cleanup_failed_commit_data_file() {
        // We should clean up data files that are written but the commit failed
        // for whatever reason

        let mut fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.block_commits();
        assert!(fixture.append_some_data().await.is_err());
        fixture.clock.set_system_time(Duration::days(10));

        let before_count = fixture.count_files().await.unwrap();
        // This append will fail since the commit is blocked but it should have
        // deposited a data file
        assert_eq!(before_count.num_data_files, 2);
        assert_eq!(before_count.num_manifest_files, 2);
        assert_eq!(before_count.num_tx_files, 2);

        // All of our manifests are newer than the threshold but temp files
        // should still be deleted.
        let removed = fixture
            .run_cleanup(utc_now() - Duration::days(7))
            .await
            .unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(removed.old_versions, 0);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_manifest_files, 2);
        assert_eq!(after_count.num_tx_files, 1);
    }

    #[tokio::test]
    async fn dont_cleanup_in_progress_write() {
        // We should not cleanup data files newer than our threshold as they might
        // belong to in-progress writes

        // For testing purposes we actually create these files with a failed write
        // but the cleanup routine has no way of detecting this.  They should look
        // just like an in-progress write.
        let mut fixture = MockDatasetFixture::try_new().unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.create_some_data().await.unwrap();
        fixture.block_commits();
        assert!(fixture.append_some_data().await.is_err());

        let before_count = fixture.count_files().await.unwrap();

        let removed = fixture
            .run_cleanup(utc_now() - Duration::days(7))
            .await
            .unwrap();

        assert_eq!(removed.old_versions, 0);
        assert_eq!(removed.bytes_removed, 0);

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count, after_count);
    }

    #[tokio::test]
    async fn can_recover_delete_failure() {
        // We want to make sure that an I/O error during the cleanup process doesn't
        // prevent us from running cleanup again later.
        let mut fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.clock.set_system_time(Duration::days(10));
        fixture.overwrite_some_data().await.unwrap();

        // The delete operation should delete the first version and its
        // data file.  However, we will block the manifest file from getting
        // cleaned up by simulating an I/O error.
        fixture.block_delete_manifest();

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_data_files, 2);
        assert_eq!(before_count.num_manifest_files, 3);

        assert!(fixture
            .run_cleanup(utc_now() - Duration::days(7))
            .await
            .is_err());

        // This test currently relies on us sending in manifest files after
        // data files.  Also, the delete process is run in parallel.  However,
        // it seems stable to stably delete the data file even though the manifest delete fails.
        // My guess is that it is not possible to interrupt a task in flight and so it still
        // has to finish the buffered tasks even if they are ignored.
        let mid_count = fixture.count_files().await.unwrap();
        assert_eq!(mid_count.num_data_files, 1);
        assert_eq!(mid_count.num_manifest_files, 3);

        fixture.unblock_delete_manifest();

        let removed = fixture
            .run_cleanup(utc_now() - Duration::days(7))
            .await
            .unwrap();

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(removed.old_versions, 1);
        assert_eq!(
            removed.bytes_removed,
            mid_count.num_bytes - after_count.num_bytes
        );

        assert_eq!(after_count.num_data_files, 1);
        assert_eq!(after_count.num_manifest_files, 2);
    }

    #[tokio::test]
    async fn test_cleanup_partial_writes() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = ArrowSchema::new(vec![Field::new("a", DataType::Int32, false)]);
        let reader = RecordBatchIterator::new(vec![], Arc::new(schema));
        let dataset = Dataset::write(reader, test_uri, Default::default())
            .await
            .unwrap();
        let store = dataset.object_store();

        // Create a partial write
        let path1 = dataset.base.child("data").child("test");
        let (multipart_id, mut writer) = store.inner.put_multipart(&path1).await.unwrap();
        writer.write_all(b"test").await.unwrap();

        // paths are relative to the store data path
        let path1 = Path::from("test");
        // Add a non-existant path and id
        let path2 = Path::from("test2");
        let non_existent_multipart_id = "non-existant-id".to_string();
        let objects = vec![
            (&path1, &multipart_id),
            (&path2, &non_existent_multipart_id),
        ];

        cleanup_partial_writes(dataset.object_store(), objects)
            .await
            .unwrap();
    }
}
