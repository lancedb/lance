// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One directory level at a time over `object_store`'s own paginated list API.

use std::sync::Arc;

use object_store::list::{PaginatedListOptions, PaginatedListStore};

use lance_core::Result;

use super::{DELIMITER, DirCursor, DirPage, PaginatedDirLister, keyed_entries};

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
        // Only this lister's own token. `offset` is left unset: a caller-supplied key means
        // something different on every store — S3 excludes it, Azure includes it, Azurite
        // drops it — and a continuation token needs none of that.
        let page_token = match resume {
            Some(cursor) => Some(cursor.expect_backend()?.to_string()),
            None => None,
        };
        let page = self
            .0
            .list_paginated(
                prefix,
                PaginatedListOptions {
                    delimiter: Some(DELIMITER.into()),
                    max_keys: limit,
                    page_token,
                    ..Default::default()
                },
            )
            .await?;

        let entries = keyed_entries(&page.result, prefix);
        Ok(DirPage {
            entries: entries.into_iter().map(|child| child.entry).collect(),
            next: page.page_token.map(DirCursor::backend),
        })
    }
}
