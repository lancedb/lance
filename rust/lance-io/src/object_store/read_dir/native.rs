// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One directory level at a time over `object_store`'s own paginated list API.

use std::sync::Arc;

use object_store::list::{PaginatedListOptions, PaginatedListStore};

use lance_core::Result;

use super::{DELIMITER, DirCursor, DirPage, PaginatedDirLister, Resume, keyed_entries};

/// [`PaginatedDirLister`] over `object_store`'s paginated listing API, used by the native S3,
/// GCS and Azure stores.
///
/// Resumes from the store's own continuation token, which is exact and needs no key order to
/// be correct. That is what lets S3 Express and an Azure account with a hierarchical namespace
/// be paged: neither lists in key order, and neither has to.
pub struct NativeDirLister(Arc<dyn PaginatedListStore>);

impl NativeDirLister {
    pub(crate) fn for_store(store: Arc<dyn PaginatedListStore>) -> Arc<dyn PaginatedDirLister> {
        Arc::new(Self(store))
    }
}

impl std::fmt::Debug for NativeDirLister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NativeDirLister")
    }
}

#[async_trait::async_trait]
impl PaginatedDirLister for NativeDirLister {
    async fn list_page(
        &self,
        prefix: Option<&str>,
        resume: Option<&DirCursor>,
        limit: Option<usize>,
    ) -> Result<DirPage> {
        // A key cursor only ever starts a listing: it comes from a caller resuming a walk that
        // began on a store with no continuation token of its own. Every page after the first
        // resumes from the token this one hands back. `object_store` addresses keys by their
        // `Path` spelling, which is also how it reported the key, so the two join as they are.
        let (page_token, offset) = match resume.map(DirCursor::resume) {
            Some(Resume::Backend(token)) => (Some(token.to_string()), None),
            Some(Resume::Key(key)) => (None, Some(format!("{}{key}", prefix.unwrap_or_default()))),
            None => (None, None),
        };
        let page = self
            .0
            .list_paginated(
                prefix,
                PaginatedListOptions {
                    offset,
                    delimiter: Some(DELIMITER.into()),
                    max_keys: limit,
                    page_token,
                    ..Default::default()
                },
            )
            .await?;

        let mut entries = keyed_entries(&page.result, prefix);
        // An offset is a key, and the entry it names can come back anyway: Azure's `startFrom`
        // is inclusive, and a directory is a prefix whose keys all sort after it, so it
        // collapses back into the same entry even under an exclusive `start-after`.
        if let Some(Resume::Key(key)) = resume.map(DirCursor::resume) {
            entries.retain(|child| child.key.as_str() > key.as_str());
        }
        Ok(DirPage {
            entries: entries.into_iter().map(|child| child.entry).collect(),
            next: page.page_token.map(DirCursor::backend),
        })
    }
}
