// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, future, stream::BoxStream};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions,
};
use object_store_opendal::OpendalStore as InnerOpendalStore;
use opendal::Operator;

/// Adapts OpenDAL listing paths to Lance's raw object-store path convention.
///
/// The upstream bridge builds listed locations with [`Path::from`], which
/// percent-encodes reserved characters. Lance builds dataset base paths with
/// [`Path::from_url_path`], so listed locations must be decoded to match.
#[derive(Debug, Clone)]
pub(super) struct OpendalStore {
    inner: InnerOpendalStore,
}

impl OpendalStore {
    pub(super) fn new(operator: Operator) -> Self {
        Self {
            inner: InnerOpendalStore::new(operator),
        }
    }
}

impl fmt::Display for OpendalStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

fn normalize_location(location: &Path) -> object_store::Result<Path> {
    Path::from_url_path(location.as_ref()).map_err(|source| object_store::Error::Generic {
        store: "OpendalStore",
        source: Box::new(source),
    })
}

fn normalize_object_meta(mut meta: ObjectMeta) -> object_store::Result<ObjectMeta> {
    meta.location = normalize_location(&meta.location)?;
    Ok(meta)
}

#[async_trait]
impl OSObjectStore for OpendalStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner
            .list(prefix)
            .map(|result| result.and_then(normalize_object_meta))
            .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        if self.inner.info().capability().list_with_start_after {
            self.inner
                .list_with_offset(prefix, offset)
                .map(|result| result.and_then(normalize_object_meta))
                .boxed()
        } else {
            // The bridge's fallback compares its encoded output with the raw
            // offset. Filter normalized locations so both sides use one form.
            let offset = offset.clone();
            self.list(prefix)
                .try_filter(move |meta| future::ready(meta.location > offset))
                .boxed()
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let mut result = self.inner.list_with_delimiter(prefix).await?;
        for object in &mut result.objects {
            object.location = normalize_location(&object.location)?;
        }
        for common_prefix in &mut result.common_prefixes {
            *common_prefix = normalize_location(common_prefix)?;
        }
        Ok(result)
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::TryStreamExt;
    use object_store::ObjectStoreExt;
    use opendal::services::Memory;

    use super::*;

    #[tokio::test]
    async fn test_list_preserves_raw_locations() {
        let operator = Operator::new(Memory::default()).unwrap();
        let store = OpendalStore::new(operator);
        let base = Path::from_url_path("tables/run~1/t.lance").unwrap();
        let direct_location = Path::from_url_path("tables/run~1/t.lance/manifest.lance").unwrap();
        let nested_location = Path::from_url_path("tables/run~1/t.lance/data/part.lance").unwrap();
        for location in [&direct_location, &nested_location] {
            store
                .put(location, Bytes::from_static(b"data").into())
                .await
                .unwrap();
        }

        let listed = store
            .list(Some(&base))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut listed_locations = listed
            .into_iter()
            .map(|meta| meta.location)
            .collect::<Vec<_>>();
        listed_locations.sort();
        let mut expected_locations = vec![direct_location.clone(), nested_location.clone()];
        expected_locations.sort();
        assert_eq!(listed_locations, expected_locations);
        assert!(
            listed_locations
                .iter()
                .all(|location| location.prefix_matches(&base))
        );

        let listed_after_nested = store
            .list_with_offset(Some(&base), &nested_location)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(listed_after_nested.len(), 1);
        assert_eq!(listed_after_nested[0].location, direct_location);

        let delimited = store.list_with_delimiter(Some(&base)).await.unwrap();
        assert_eq!(delimited.objects.len(), 1);
        assert_eq!(delimited.objects[0].location, direct_location);
        assert_eq!(
            delimited.common_prefixes,
            vec![Path::from_url_path("tables/run~1/t.lance/data").unwrap()]
        );
    }
}
