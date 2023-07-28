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

use crate::format::Manifest;
use object_store::{local::LocalFileSystem, path::Path};

/// A store that can atomically commit a manifest to some path.
#[async_trait::async_trait]
pub trait CommitStore: Send + Sync + std::fmt::Debug {
    async fn try_commit_manifest(
        &self,
        manifest: &Manifest,
        path: &Path,
    ) -> Result<(), CommitError>;
}

pub enum CommitError {
    /// Another transaction has already been written to the path
    CommitConflict,
    /// Something else went wrong
    OtherError(crate::Error),
}

impl From<crate::Error> for CommitError {
    fn from(e: crate::Error) -> Self {
        CommitError::OtherError(e)
    }
}

impl From<CommitError> for crate::Error {
    fn from(e: CommitError) -> Self {
        match e {
            CommitError::CommitConflict => crate::Error::Internal {
                message: "Commit conflict".to_string(),
            },
            CommitError::OtherError(e) => e,
        }
    }
}

#[async_trait::async_trait]
impl CommitStore for LocalFileSystem {
    async fn try_commit_manifest(
        &self,
        manifest: &Manifest,
        path: &Path,
    ) -> Result<(), CommitError> {
        // Create a temporary file, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary file.
        todo!()
    }
}

#[async_trait::async_trait]
impl CommitStore for object_store::gcp::GoogleCloudStorage {
    async fn try_commit_manifest(
        &self,
        manifest: &Manifest,
        path: &Path,
    ) -> Result<(), CommitError> {
        // Create a temporary object, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary object.
        todo!()
    }
}

#[async_trait::async_trait]
impl CommitStore for object_store::azure::MicrosoftAzure {
    async fn try_commit_manifest(
        &self,
        manifest: &Manifest,
        path: &Path,
    ) -> Result<(), CommitError> {
        // Create a temporary object, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary object.
        todo!()
    }
}

#[async_trait::async_trait]
impl CommitStore for object_store::aws::AmazonS3 {
    async fn try_commit_manifest(
        &self,
        manifest: &Manifest,
        path: &Path,
    ) -> Result<(), CommitError> {
        // Options:
        // 1. Just commit naively.
        // 2. Use S3 commit queue protocol, if that works.
        todo!()
    }
}

#[async_trait::async_trait]
impl CommitStore for object_store::memory::InMemory {
    async fn try_commit_manifest(
        &self,
        manifest: &Manifest,
        path: &Path,
    ) -> Result<(), CommitError> {
        // Create a temporary object, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary object.
        todo!()
    }
}
