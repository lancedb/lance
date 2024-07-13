// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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

use std::io;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::{fmt::Debug, fs::DirEntry};

use futures::{
    future::{self, BoxFuture},
    stream::BoxStream,
    StreamExt, TryStreamExt,
};
use object_store::{path::Path, Error as ObjectStoreError, ObjectStore as OSObjectStore};
use snafu::{location, Location};
use url::Url;

#[cfg(feature = "dynamodb")]
pub mod dynamodb;
pub mod external_manifest;

use lance_core::{Error, Result};
use lance_io::object_store::{ObjectStore, ObjectStoreExt, ObjectStoreParams};

#[cfg(feature = "dynamodb")]
use {
    self::external_manifest::{ExternalManifestCommitHandler, ExternalManifestStore},
    aws_credential_types::provider::error::CredentialsError,
    aws_credential_types::provider::ProvideCredentials,
    lance_io::object_store::{build_aws_credential, StorageOptions},
    object_store::aws::AmazonS3ConfigKey,
    object_store::aws::AwsCredentialProvider,
    std::borrow::Cow,
    std::time::{Duration, SystemTime},
};

use crate::format::{Index, Manifest};

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const MANIFEST_EXTENSION: &str = "manifest";

/// Function that writes the manifest to the object store.
pub type ManifestWriter = for<'a> fn(
    object_store: &'a dyn OSObjectStore,
    manifest: &'a mut Manifest,
    indices: Option<Vec<Index>>,
    path: &'a Path,
) -> BoxFuture<'a, Result<()>>;

/// Get the manifest file path for a version.
pub fn manifest_path(base: &Path, version: u64) -> Path {
    base.child(VERSIONS_DIR)
        .child(format!("{version}.{MANIFEST_EXTENSION}"))
}

pub fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
}

#[derive(Debug)]
pub struct ManifestLocation {
    /// The version the manifest corresponds to.
    pub version: u64,
    /// Path of the manifest file, relative to the table root.
    pub path: Path,
    /// Size, in bytes, of the manifest file. If it is not known, this field should be `None`.
    pub size: Option<u64>,
}

/// Get the latest manifest path
async fn current_manifest_path(
    object_store: &ObjectStore,
    base: &Path,
) -> Result<ManifestLocation> {
    if object_store.is_local() {
        if let Ok(Some(location)) = current_manifest_local(base) {
            return Ok(location);
        }
    }

    // We use `list_with_delimiter` to avoid listing the contents of child directories.
    let manifest_files = object_store
        .inner
        .list_with_delimiter(Some(&base.child(VERSIONS_DIR)))
        .await?;

    let current = manifest_files
        .objects
        .into_iter()
        .filter(|meta| {
            meta.location.filename().is_some()
                && meta
                    .location
                    .filename()
                    .unwrap()
                    .ends_with(MANIFEST_EXTENSION)
        })
        .filter_map(|meta| {
            let version = meta
                .location
                .filename()
                .unwrap()
                .split_once('.')
                .and_then(|(version_str, _)| version_str.parse::<u64>().ok())?;
            Some((version, meta))
        })
        .max_by_key(|(version, _)| *version);

    if let Some((version, meta)) = current {
        Ok(ManifestLocation {
            version,
            path: meta.location,
            size: Some(meta.size as u64),
        })
    } else {
        Err(Error::NotFound {
            uri: manifest_path(base, 1).to_string(),
            location: location!(),
        })
    }
}

// This is an optimized function that searches for the latest manifest. In
// object_store, list operations lookup metadata for each file listed. This
// method only gets the metadata for the found latest manifest.
fn current_manifest_local(base: &Path) -> std::io::Result<Option<ManifestLocation>> {
    let path = lance_io::local::to_local_path(&base.child(VERSIONS_DIR));
    let entries = std::fs::read_dir(path)?;

    let mut latest_entry: Option<(u64, DirEntry)> = None;

    for entry in entries {
        let entry = entry?;
        let filename_raw = entry.file_name();
        let filename = filename_raw.to_string_lossy();
        if !filename.ends_with(MANIFEST_EXTENSION) {
            // Need to ignore temporary files, such as
            // .tmp_7.manifest_9c100374-3298-4537-afc6-f5ee7913666d
            continue;
        }
        let Some(version) = filename
            .split_once('.')
            .and_then(|(version_str, _)| version_str.parse::<u64>().ok())
        else {
            continue;
        };

        if let Some((latest_version, _)) = &latest_entry {
            if version > *latest_version {
                latest_entry = Some((version, entry));
            }
        } else {
            latest_entry = Some((version, entry));
        }
    }

    if let Some((version, entry)) = latest_entry {
        let path = Path::from_filesystem_path(entry.path())
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        Ok(Some(ManifestLocation {
            version,
            path,
            size: Some(entry.metadata()?.len()),
        }))
    } else {
        Ok(None)
    }
}

async fn list_manifests<'a>(
    base_path: &Path,
    object_store: &'a dyn OSObjectStore,
) -> Result<BoxStream<'a, Result<Path>>> {
    let base_path = base_path.clone();
    Ok(object_store
        .read_dir_all(&base_path.child(VERSIONS_DIR), None)
        .await?
        .try_filter_map(|obj_meta| {
            if obj_meta.location.extension() == Some(MANIFEST_EXTENSION) {
                future::ready(Ok(Some(obj_meta.location)))
            } else {
                future::ready(Ok(None))
            }
        })
        .boxed())
}

pub fn parse_version_from_path(path: &Path) -> Result<u64> {
    path.filename()
        .and_then(|name| name.split_once('.'))
        .filter(|(_, extension)| *extension == MANIFEST_EXTENSION)
        .and_then(|(version, _)| version.parse::<u64>().ok())
        .ok_or(Error::Internal {
            message: format!("Expected manifest file, but found {}", path),
            location: location!(),
        })
}

fn make_staging_manifest_path(base: &Path) -> Result<Path> {
    let id = uuid::Uuid::new_v4().to_string();
    Path::parse(format!("{base}-{id}")).map_err(|e| Error::IO {
        source: Box::new(e),
        location: location!(),
    })
}

async fn write_latest_manifest(
    from_path: &Path,
    base_path: &Path,
    object_store: &dyn OSObjectStore,
) -> Result<()> {
    let latest_path = latest_manifest_path(base_path);
    let staging_path = make_staging_manifest_path(from_path)?;
    object_store
        .copy(from_path, &staging_path)
        .await
        .map_err(|err| CommitError::OtherError(err.into()))?;
    object_store.rename(&staging_path, &latest_path).await?;
    Ok(())
}

#[cfg(feature = "dynamodb")]
const DDB_URL_QUERY_KEY: &str = "ddbTableName";

/// Handle commits that prevent conflicting writes.
///
/// Commit implementations ensure that if there are multiple concurrent writers
/// attempting to write the next version of a table, only one will win. In order
/// to work, all writers must use the same commit handler type.
/// This trait is also responsible for resolving where the manifests live.
///
// TODO: pub(crate)
#[async_trait::async_trait]
pub trait CommitHandler: Debug + Send + Sync {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> Result<ManifestLocation> {
        Ok(current_manifest_path(object_store, base_path).await?)
    }

    /// Get the path to the latest version manifest of a dataset at the base_path
    async fn resolve_latest_version(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<Path, Error> {
        // TODO: we need to pade 0's to the version number on the manifest file path
        Ok(current_manifest_path(object_store, base_path).await?.path)
    }

    // for default implementation, parse the version from the path
    async fn resolve_latest_version_id(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> Result<u64> {
        Ok(current_manifest_path(object_store, base_path)
            .await?
            .version)
    }

    /// Get the path to a specific versioned manifest of a dataset at the base_path
    async fn resolve_version(
        &self,
        base_path: &Path,
        version: u64,
        _object_store: &dyn OSObjectStore,
    ) -> std::result::Result<Path, Error> {
        Ok(manifest_path(base_path, version))
    }

    /// List manifests that are available for a dataset at the base_path
    async fn list_manifests<'a>(
        &self,
        base_path: &Path,
        object_store: &'a dyn OSObjectStore,
    ) -> Result<BoxStream<'a, Result<Path>>> {
        list_manifests(base_path, object_store).await
    }

    /// Commit a manifest.
    ///
    /// This function should return an [CommitError::CommitConflict] if another
    /// transaction has already been committed to the path.
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn OSObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError>;
}


pub trait WrappingCommitHandler: Debug + Send + Sync {
    fn wrap(&self, original: Arc<dyn CommitHandler>) -> Arc<dyn CommitHandler>;
}


/// Adapt an object_store credentials into AWS SDK creds
#[cfg(feature = "dynamodb")]
#[derive(Debug)]
struct OSObjectStoreToAwsCredAdaptor(AwsCredentialProvider);

#[cfg(feature = "dynamodb")]
impl ProvideCredentials for OSObjectStoreToAwsCredAdaptor {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(async {
            let creds = self
                .0
                .get_credential()
                .await
                .map_err(|e| CredentialsError::provider_error(Box::new(e)))?;
            Ok(aws_credential_types::Credentials::new(
                &creds.key_id,
                &creds.secret_key,
                creds.token.clone(),
                Some(
                    SystemTime::now()
                        .checked_add(Duration::from_secs(
                            60 * 10, //  10 min
                        ))
                        .expect("overflow"),
                ),
                "",
            ))
        })
    }
}

#[cfg(feature = "dynamodb")]
async fn build_dynamodb_external_store(
    table_name: &str,
    creds: AwsCredentialProvider,
    region: &str,
    endpoint: Option<String>,
    app_name: &str,
) -> Result<Arc<dyn ExternalManifestStore>> {
    use super::commit::dynamodb::DynamoDBExternalManifestStore;
    use aws_sdk_dynamodb::{
        config::{IdentityCache, Region},
        Client,
    };

    let mut dynamodb_config = aws_sdk_dynamodb::config::Builder::new()
        .region(Some(Region::new(region.to_string())))
        .credentials_provider(OSObjectStoreToAwsCredAdaptor(creds))
        // caching should be handled by passed AwsCredentialProvider
        .identity_cache(IdentityCache::no_cache());

    if let Some(endpoint) = endpoint {
        dynamodb_config = dynamodb_config.endpoint_url(endpoint);
    }
    let client = Client::from_conf(dynamodb_config.build());

    DynamoDBExternalManifestStore::new_external_store(client.into(), table_name, app_name).await
}

pub async fn commit_handler_from_url(
    url_or_path: &str,
    // This looks unused if dynamodb feature disabled
    #[allow(unused_variables)] options: &Option<ObjectStoreParams>,
) -> Result<Arc<dyn CommitHandler>> {
    let url = match Url::parse(url_or_path) {
        Ok(url) if url.scheme().len() == 1 && cfg!(windows) => {
            // On Windows, the drive is parsed as a scheme
            return Ok(Arc::new(RenameCommitHandler));
        }
        Ok(url) => url,
        Err(_) => {
            return Ok(Arc::new(RenameCommitHandler));
        }
    };

    match url.scheme() {
        // TODO: for Cloudflare R2 and Minio, we can provide a PutIfNotExist commit handler
        // See: https://docs.rs/object_store/latest/object_store/aws/enum.S3ConditionalPut.html#variant.ETagMatch
        "s3" => Ok(Arc::new(UnsafeCommitHandler)),
        #[cfg(not(feature = "dynamodb"))]
        "s3+ddb" => Err(Error::InvalidInput {
            source: "`s3+ddb://` scheme requires `dynamodb` feature to be enabled".into(),
            location: location!(),
        }),
        #[cfg(feature = "dynamodb")]
        "s3+ddb" => {
            if url.query_pairs().count() != 1 {
                return Err(Error::InvalidInput {
                    source: "`s3+ddb://` scheme and expects exactly one query `ddbTableName`"
                        .into(),
                    location: location!(),
                });
            }
            let table_name = match url.query_pairs().next() {
                Some((Cow::Borrowed(key), Cow::Borrowed(table_name)))
                    if key == DDB_URL_QUERY_KEY =>
                {
                    if table_name.is_empty() {
                        return Err(Error::InvalidInput {
                            source: "`s3+ddb://` scheme requires non empty dynamodb table name"
                                .into(),
                            location: location!(),
                        });
                    }
                    table_name
                }
                _ => {
                    return Err(Error::InvalidInput {
                        source: "`s3+ddb://` scheme and expects exactly one query `ddbTableName`"
                            .into(),
                        location: location!(),
                    });
                }
            };
            let options = options.clone().unwrap_or_default();
            let storage_options = StorageOptions(options.storage_options.unwrap_or_default());
            let dynamo_endpoint = get_dynamodb_endpoint(&storage_options);
            let storage_options = storage_options.as_s3_options();

            let region = storage_options
                .get(&AmazonS3ConfigKey::Region)
                .map(|s| s.to_string());

            let (aws_creds, region) = build_aws_credential(
                options.s3_credentials_refresh_offset,
                options.aws_credentials.clone(),
                Some(&storage_options),
                region,
            )
            .await?;

            Ok(Arc::new(ExternalManifestCommitHandler {
                external_manifest_store: build_dynamodb_external_store(
                    table_name,
                    aws_creds.clone(),
                    &region,
                    dynamo_endpoint,
                    "lancedb",
                )
                .await?,
            }))
        }
        "gs" | "az" | "file" | "file-object-store" | "memory" => Ok(Arc::new(RenameCommitHandler)),

        unknow_scheme => {
            let err = lance_core::Error::from(object_store::Error::NotSupported {
                source: format!("Unsupported URI scheme: {}", unknow_scheme).into(),
            });
            Err(err)
        }
    }
}

#[cfg(feature = "dynamodb")]
fn get_dynamodb_endpoint(storage_options: &StorageOptions) -> Option<String> {
    if let Some(endpoint) = storage_options.0.get("dynamodb_endpoint") {
        Some(endpoint.to_string())
    } else {
        std::env::var("DYNAMODB_ENDPOINT").ok()
    }
}

/// Errors that can occur when committing a manifest.
#[derive(Debug)]
pub enum CommitError {
    /// Another transaction has already been written to the path
    CommitConflict,
    /// Something else went wrong
    OtherError(Error),
}

impl From<Error> for CommitError {
    fn from(e: Error) -> Self {
        Self::OtherError(e)
    }
}

impl From<CommitError> for Error {
    fn from(e: CommitError) -> Self {
        match e {
            CommitError::CommitConflict => Self::Internal {
                message: "Commit conflict".to_string(),
                location: location!(),
            },
            CommitError::OtherError(e) => e,
        }
    }
}

/// Whether we have issued a warning about using the unsafe commit handler.
static WARNED_ON_UNSAFE_COMMIT: AtomicBool = AtomicBool::new(false);

/// A naive commit implementation that does not prevent conflicting writes.
///
/// This will log a warning the first time it is used.
pub struct UnsafeCommitHandler;

#[async_trait::async_trait]
impl CommitHandler for UnsafeCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn OSObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        // Log a one-time warning
        if !WARNED_ON_UNSAFE_COMMIT.load(std::sync::atomic::Ordering::Relaxed) {
            WARNED_ON_UNSAFE_COMMIT.store(true, std::sync::atomic::Ordering::Relaxed);
            log::warn!(
                "Using unsafe commit handler. Concurrent writes may result in data loss. \
                 Consider providing a commit handler that prevents conflicting writes."
            );
        }

        let version_path = self
            .resolve_version(base_path, manifest.version, object_store)
            .await?;
        // Write the manifest naively
        manifest_writer(object_store, manifest, indices, &version_path).await?;

        write_latest_manifest(&version_path, base_path, object_store).await?;

        Ok(())
    }
}

impl Debug for UnsafeCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeCommitHandler").finish()
    }
}

/// A commit implementation that uses a lock to prevent conflicting writes.
#[async_trait::async_trait]
pub trait CommitLock: Debug {
    type Lease: CommitLease;

    /// Attempt to lock the table for the given version.
    ///
    /// If it is already locked by another transaction, wait until it is unlocked.
    /// Once it is unlocked, return [CommitError::CommitConflict] if the version
    /// has already been committed. Otherwise, return the lock.
    ///
    /// To prevent poisoned locks, it's recommended to set a timeout on the lock
    /// of at least 30 seconds.
    ///
    /// It is not required that the lock tracks the version. It is provided in
    /// case the locking is handled by a catalog service that needs to know the
    /// current version of the table.
    async fn lock(&self, version: u64) -> std::result::Result<Self::Lease, CommitError>;
}

#[async_trait::async_trait]
pub trait CommitLease: Send + Sync {
    /// Return the lease, indicating whether the commit was successful.
    async fn release(&self, success: bool) -> std::result::Result<(), CommitError>;
}

#[async_trait::async_trait]
impl<T: CommitLock + Send + Sync> CommitHandler for T {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn OSObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        let path = self
            .resolve_version(base_path, manifest.version, object_store)
            .await?;
        // NOTE: once we have the lease we cannot use ? to return errors, since
        // we must release the lease before returning.
        let lease = self.lock(manifest.version).await?;

        // Head the location and make sure it's not already committed
        match object_store.head(&path).await {
            Ok(_) => {
                // The path already exists, so it's already committed
                // Release the lock
                lease.release(false).await?;

                return Err(CommitError::CommitConflict);
            }
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(e) => {
                // Something else went wrong
                // Release the lock
                lease.release(false).await?;

                return Err(CommitError::OtherError(e.into()));
            }
        }
        let res = manifest_writer(object_store, manifest, indices, &path).await;

        write_latest_manifest(&path, base_path, object_store).await?;

        // Release the lock
        lease.release(res.is_ok()).await?;

        res.map_err(|err| err.into())
    }
}

#[async_trait::async_trait]
impl<T: CommitLock + Send + Sync> CommitHandler for Arc<T> {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn OSObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        self.as_ref()
            .commit(manifest, indices, base_path, object_store, manifest_writer)
            .await
    }
}

/// A commit implementation that uses a temporary path and renames the object.
///
/// This only works for object stores that support atomic rename if not exist.
pub struct RenameCommitHandler;

#[async_trait::async_trait]
impl CommitHandler for RenameCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn OSObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        // Create a temporary object, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary object.

        let path = self
            .resolve_version(base_path, manifest.version, object_store)
            .await?;

        // Add .tmp_ prefix to the path
        let mut parts: Vec<_> = path.parts().collect();
        // Add a UUID to the end of the filename to avoid conflicts
        let uuid = uuid::Uuid::new_v4();
        let new_name = format!(
            ".tmp_{}_{}",
            parts.last().unwrap().as_ref(),
            uuid.as_hyphenated()
        );
        let _ = std::mem::replace(parts.last_mut().unwrap(), new_name.into());
        let tmp_path: Path = parts.into_iter().collect();

        // Write the manifest to the temporary path
        manifest_writer(object_store, manifest, indices, &tmp_path).await?;

        let res = match object_store.rename_if_not_exists(&tmp_path, &path).await {
            Ok(_) => Ok(()),
            Err(ObjectStoreError::AlreadyExists { .. }) => {
                // Another transaction has already been committed
                // Attempt to clean up temporary object, but ignore errors if we can't
                let _ = object_store.delete(&tmp_path).await;

                return Err(CommitError::CommitConflict);
            }
            Err(e) => {
                // Something else went wrong
                return Err(CommitError::OtherError(e.into()));
            }
        };

        write_latest_manifest(&path, base_path, object_store).await?;

        res
    }
}

impl Debug for RenameCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenameCommitHandler").finish()
    }
}

#[derive(Debug, Clone)]
pub struct CommitConfig {
    pub num_retries: u32,
    // TODO: add isolation_level
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self { num_retries: 20 }
    }
}
