// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One directory level at a time over `object_store`'s own paginated list API.

use std::sync::Arc;

use object_store::list::{PaginatedListOptions, PaginatedListStore};

use lance_core::Result;

use super::{DELIMITER, DirPage, PaginatedDirLister};

/// [`PaginatedDirLister`] over `object_store`'s paginated listing API, used by the native
/// S3, GCS and Azure stores.
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
        start_after: Option<&str>,
        limit: Option<usize>,
    ) -> Result<DirPage> {
        // `object_store` addresses keys by their `Path` spelling, which is also how it
        // reported the cursor, so the two join as they are.
        let offset = start_after.map(|cursor| format!("{}{cursor}", prefix.unwrap_or_default()));
        let page = self
            .0
            .list_paginated(
                prefix,
                PaginatedListOptions {
                    offset,
                    delimiter: Some(DELIMITER.into()),
                    max_keys: limit,
                    // Resuming by key rather than by token: see `PaginatedDirLister`.
                    page_token: None,
                    ..Default::default()
                },
            )
            .await?;
        Ok(DirPage {
            children: page.result,
            has_more: page.page_token.is_some(),
        })
    }
}
