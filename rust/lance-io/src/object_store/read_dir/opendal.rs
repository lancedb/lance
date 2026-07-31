// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One directory level at a time over OpenDAL's `Lister`, for OpenDAL-backed stores.

use std::borrow::Cow;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use object_store::{ListResult, ObjectMeta, path::Path};
use opendal::{Entry, Operator, raw::Timestamp, raw::percent_decode_path};

use lance_core::{Error, Result};

use super::{DirPage, PaginatedDirLister};

/// How many children to fetch when the caller does not ask for a specific number.
/// Matches the page size object stores default to.
const DEFAULT_PAGE_SIZE: usize = 1000;

/// Where the operator for a listing comes from.
///
/// Stores with vended credentials rebuild their operator whenever the credentials change,
/// so the operator has to be resolved per request rather than captured once.
pub(in crate::object_store) enum OperatorSource {
    Static(Operator),
    #[cfg(any(feature = "oss", feature = "huggingface", feature = "tos"))]
    Dynamic(Arc<super::super::dynamic_opendal::DynamicOpenDalStore>),
}

/// [`PaginatedDirLister`] over OpenDAL's `Lister`.
pub struct OpendalDirLister {
    source: OperatorSource,
}

impl OpendalDirLister {
    /// A lister for `operator`, or `None` when the service cannot resume a listing.
    ///
    /// Without `start_after` every page would restart at the beginning of the directory,
    /// which is worse than falling back to a full listing: the fallback at least goes
    /// through the wrappers around the store.
    pub(crate) fn for_operator(operator: Operator) -> Option<Arc<dyn PaginatedDirLister>> {
        operator.info().capability().list_with_start_after.then(|| {
            Arc::new(Self {
                source: OperatorSource::Static(operator),
            }) as Arc<dyn PaginatedDirLister>
        })
    }

    /// A lister for a store whose operator is rebuilt as credentials are refreshed.
    ///
    /// The service's capabilities are not known until an operator exists, so this always
    /// installs a lister; [`Self::list_page`] falls back to a full listing per page if the
    /// service turns out not to support `start_after`.
    #[cfg(any(feature = "oss", feature = "huggingface", feature = "tos"))]
    pub(in crate::object_store) fn for_dynamic_store(
        store: Arc<super::super::dynamic_opendal::DynamicOpenDalStore>,
    ) -> Arc<dyn PaginatedDirLister> {
        Arc::new(Self {
            source: OperatorSource::Dynamic(store),
        })
    }

    async fn operator(&self) -> Result<Operator> {
        match &self.source {
            OperatorSource::Static(operator) => Ok(operator.clone()),
            #[cfg(any(feature = "oss", feature = "huggingface", feature = "tos"))]
            OperatorSource::Dynamic(store) => store.current_operator().await,
        }
    }
}

impl std::fmt::Debug for OpendalDirLister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpendalDirLister")
    }
}

#[async_trait::async_trait]
impl PaginatedDirLister for OpendalDirLister {
    async fn list_page(
        &self,
        prefix: Option<&str>,
        start_after: Option<&str>,
        limit: Option<usize>,
    ) -> Result<DirPage> {
        let operator = self.operator().await?;
        let can_resume = operator.info().capability().list_with_start_after;

        // OpenDAL takes the service's own keys; `object_store` may hand us a percent-encoded
        // prefix. The cursor needs no such treatment: it already arrives in the service's
        // spelling, which is what `backend_key` below produces.
        let path = prefix.map(percent_decode_path).unwrap_or_default();
        let mut request = operator.lister_with(&path);

        // Without pushdown the page has to be the whole directory: limiting an unfiltered
        // listing would cut it off before reaching the children the cursor asked for.
        let page_size = match (start_after, can_resume) {
            (Some(cursor), true) => {
                request = request.start_after(&format!("{path}{cursor}"));
                Some(limit.unwrap_or(DEFAULT_PAGE_SIZE))
            }
            (Some(_), false) => None,
            (None, _) => Some(limit.unwrap_or(DEFAULT_PAGE_SIZE)),
        };
        if let Some(page_size) = page_size {
            // One more than the page, so whether another child exists is answered from the
            // entries already fetched instead of by another request.
            request = request.limit(page_size.saturating_add(1));
        }

        let mut lister = request.await.map_err(|err| list_error(&path, err))?;
        let mut entries = Vec::new();
        while page_size.is_none_or(|page_size| entries.len() < page_size) {
            let next = lister
                .try_next()
                .await
                .map_err(|err| list_error(&path, err))?;
            let Some(entry) = next else {
                return Ok(to_dir_page(entries, false));
            };
            entries.push(entry);
        }
        let has_more = lister
            .try_next()
            .await
            .map_err(|err| list_error(&path, err))?
            .is_some();
        Ok(to_dir_page(entries, has_more))
    }

    /// `OpendalStore` percent-decodes every path handed to it, so the paths above are encoded
    /// keys and decoding one recovers the key the service listed it under.
    fn backend_key<'a>(&self, path: &'a str) -> Cow<'a, str> {
        Cow::Owned(percent_decode_path(path))
    }
}

fn list_error(path: &str, err: opendal::Error) -> Error {
    Error::io(format!("failed to list '{path}': {err}"))
}

fn to_datetime(timestamp: Timestamp) -> Option<DateTime<Utc>> {
    let timestamp = timestamp.into_inner();
    DateTime::from_timestamp(timestamp.as_second(), timestamp.subsec_nanosecond() as u32)
}

/// Split OpenDAL entries into the directory/object shape a [`ListResult`] carries.
///
/// Keys are percent-encoded on the way out, the way `object_store_opendal` reports them,
/// because that is the spelling `OpendalStore` addresses a key by: it decodes every path
/// handed to it, so a location spelled any other way could not be read back. The service's
/// own spelling, which the listing is ordered by, is recovered by
/// [`OpendalDirLister::backend_key`].
///
/// File metadata is whatever the listing reported. Unlike `object_store_opendal`, a missing
/// timestamp does not trigger a `stat` per child: a page should cost one request, and
/// callers that need full metadata can ask for it. Sizes need no such care, since OpenDAL's
/// own completion layer guarantees a file entry carries its content length.
fn to_dir_page(entries: Vec<Entry>, has_more: bool) -> DirPage {
    let mut children = ListResult {
        common_prefixes: Vec::new(),
        objects: Vec::new(),
    };
    for entry in entries {
        let meta = entry.metadata();
        let location = Path::from(entry.path());
        if meta.is_dir() {
            children.common_prefixes.push(location);
        } else {
            children.objects.push(ObjectMeta {
                location,
                last_modified: meta
                    .last_modified()
                    .and_then(to_datetime)
                    .unwrap_or_default(),
                size: meta.content_length(),
                e_tag: meta.etag().map(String::from),
                version: meta.version().map(String::from),
            });
        }
    }
    DirPage { children, has_more }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use object_store::{ObjectStoreExt, PutPayload};
    use object_store_opendal::OpendalStore;
    use opendal::raw::{
        OpCopier, OpCopy, OpCreateDir, OpList, OpPresign, OpRead, OpRename, OpStat, OpWrite,
        RpCreateDir, RpPresign, RpRename, RpStat, Service, ServiceInfo, oio,
    };
    use opendal::services::Memory;
    use opendal::{Capability, EntryMode, Metadata, OperationContext};

    use super::*;
    use crate::object_store::read_dir::DELIMITER;

    /// The arguments of one list request, as the service saw them.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ListCall {
        path: String,
        start_after: Option<String>,
        limit: Option<usize>,
    }

    /// A service holding one flat directory, so a test can see exactly what the lister
    /// asked the backend for. Directories end in a delimiter, as OpenDAL reports them.
    #[derive(Debug)]
    struct FakeService {
        children: Vec<String>,
        supports_start_after: bool,
        calls: Arc<Mutex<Vec<ListCall>>>,
    }

    impl Service for FakeService {
        type Reader = ();
        type Writer = ();
        type Lister = FakeLister;
        type Deleter = ();
        type Copier = ();

        fn info(&self) -> ServiceInfo {
            ServiceInfo::new("fake", "/", "fake")
        }

        fn capability(&self) -> Capability {
            Capability {
                list: true,
                list_with_start_after: self.supports_start_after,
                list_with_limit: true,
                ..Default::default()
            }
        }

        fn list(
            &self,
            _ctx: &OperationContext,
            path: &str,
            args: OpList,
        ) -> opendal::Result<Self::Lister> {
            self.calls.lock().unwrap().push(ListCall {
                path: path.to_string(),
                start_after: args.start_after().map(String::from),
                limit: args.limit(),
            });
            let start_after = args
                .start_after()
                .filter(|_| self.supports_start_after)
                .map(String::from);
            let entries = self
                .children
                .iter()
                .filter(|child| child.starts_with(path) && child.as_str() != path)
                .filter(|child| {
                    start_after
                        .as_deref()
                        .is_none_or(|start_after| child.as_str() > start_after)
                })
                .take(args.limit().unwrap_or(usize::MAX))
                .map(|child| {
                    let metadata = if child.ends_with('/') {
                        Metadata::new(EntryMode::DIR)
                    } else {
                        // Real services report a size for every file entry in a listing,
                        // so the fake does too.
                        Metadata::new(EntryMode::FILE).with_content_length(child.len() as u64)
                    };
                    oio::Entry::new(child, metadata)
                })
                .collect::<Vec<_>>();
            Ok(FakeLister(entries.into_iter()))
        }

        // Everything but listing goes to `()`, OpenDAL's service that refuses every operation.
        async fn create_dir(
            &self,
            ctx: &OperationContext,
            path: &str,
            args: OpCreateDir,
        ) -> opendal::Result<RpCreateDir> {
            ().create_dir(ctx, path, args).await
        }

        async fn stat(
            &self,
            ctx: &OperationContext,
            path: &str,
            args: OpStat,
        ) -> opendal::Result<RpStat> {
            ().stat(ctx, path, args).await
        }

        fn read(
            &self,
            ctx: &OperationContext,
            path: &str,
            args: OpRead,
        ) -> opendal::Result<Self::Reader> {
            ().read(ctx, path, args)
        }

        fn write(
            &self,
            ctx: &OperationContext,
            path: &str,
            args: OpWrite,
        ) -> opendal::Result<Self::Writer> {
            ().write(ctx, path, args)
        }

        fn delete(&self, ctx: &OperationContext) -> opendal::Result<Self::Deleter> {
            ().delete(ctx)
        }

        fn copy(
            &self,
            ctx: &OperationContext,
            from: &str,
            to: &str,
            args: OpCopy,
            opts: OpCopier,
        ) -> opendal::Result<Self::Copier> {
            ().copy(ctx, from, to, args, opts)
        }

        async fn rename(
            &self,
            ctx: &OperationContext,
            from: &str,
            to: &str,
            args: OpRename,
        ) -> opendal::Result<RpRename> {
            ().rename(ctx, from, to, args).await
        }

        async fn presign(
            &self,
            ctx: &OperationContext,
            path: &str,
            args: OpPresign,
        ) -> opendal::Result<RpPresign> {
            ().presign(ctx, path, args).await
        }
    }

    struct FakeLister(std::vec::IntoIter<oio::Entry>);

    impl oio::List for FakeLister {
        async fn next(&mut self) -> opendal::Result<Option<oio::Entry>> {
            Ok(self.0.next())
        }
    }

    const CHILDREN: &[&str] = &["db/a.lance/", "db/b.lance/", "db/c.lance/", "db/loose.txt"];

    fn fake_operator(supports_start_after: bool) -> (Operator, Arc<Mutex<Vec<ListCall>>>) {
        operator_over(CHILDREN, supports_start_after)
    }

    fn operator_over(
        children: &[&str],
        supports_start_after: bool,
    ) -> (Operator, Arc<Mutex<Vec<ListCall>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let service = FakeService {
            children: children.iter().map(|c| c.to_string()).collect(),
            supports_start_after,
            calls: calls.clone(),
        };
        let operator = Operator::from_parts(OperationContext::default(), Arc::new(service));
        (operator, calls)
    }

    /// The children of a page, with directories marked by a trailing delimiter so that the
    /// expectations below also pin which bucket each child was reported in.
    fn names(page: &DirPage) -> Vec<String> {
        let directories = page
            .children
            .common_prefixes
            .iter()
            .map(|prefix| format!("{prefix}{DELIMITER}"));
        let files = page
            .children
            .objects
            .iter()
            .map(|object| object.location.as_ref().to_string());
        directories.chain(files).collect()
    }

    /// What the lister asks the service for, and what it makes of the answer.
    #[rstest::rstest]
    // A request that asks for no particular page still asks for a bounded one.
    #[case::whole_directory(None, None, Some(DEFAULT_PAGE_SIZE + 1), vec!["db/a.lance/", "db/b.lance/", "db/c.lance/", "db/loose.txt"], false)]
    #[case::truncated(Some("a.lance/"), Some(2), Some(3), vec!["db/b.lance/", "db/c.lance/"], true)]
    #[case::last_page(Some("b.lance/"), Some(5), Some(6), vec!["db/c.lance/", "db/loose.txt"], false)]
    #[tokio::test]
    async fn test_pushes_cursor_and_page_size_into_the_request(
        #[case] start_after: Option<&str>,
        #[case] page_size: Option<usize>,
        // One more than the page, to tell whether another child follows.
        #[case] limit: Option<usize>,
        #[case] expected: Vec<&str>,
        #[case] has_more: bool,
    ) {
        let (operator, calls) = fake_operator(true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister
            .list_page(Some("db/"), start_after, page_size)
            .await
            .unwrap();

        assert_eq!(names(&page), expected);
        assert_eq!(page.has_more, has_more);
        assert_eq!(
            *calls.lock().unwrap(),
            vec![ListCall {
                path: "db/".to_string(),
                start_after: start_after.map(|cursor| format!("db/{cursor}")),
                limit,
            }]
        );
    }

    /// Locations are spelled the way `OpendalStore` addresses a key, since that store
    /// percent-decodes whatever it is handed. Reporting the service's raw spelling instead
    /// would hand back locations that read a different key — or none at all.
    #[tokio::test]
    async fn test_listed_locations_round_trip_through_the_store() {
        let operator = Operator::new(Memory::default()).unwrap();
        let store = OpendalStore::new(operator.clone());
        // A key holding what looks like an escape sequence: decoding it once more would
        // address `db/a/b`, a key at another level of the store entirely.
        store
            .put(
                &Path::from("db/a%2Fb"),
                PutPayload::from_static(b"expected"),
            )
            .await
            .unwrap();
        // Directly, since OpenDAL's memory service cannot resume a listing and so gets no
        // lister of its own. What a location has to be is the same either way.
        let lister = OpendalDirLister {
            source: OperatorSource::Static(operator),
        };

        let page = lister.list_page(Some("db/"), None, None).await.unwrap();
        let listed = &page.children.objects[0].location;

        assert_eq!(
            store.get(listed).await.unwrap().bytes().await.unwrap(),
            b"expected".as_slice()
        );
    }

    /// The listing is ordered by the service's own keys, and `backend_key` recovers them from
    /// the locations above. Comparing the locations instead would reorder the listing — `a~` is
    /// reported as `a%7E`, which sorts before `az` — and paging would then drop whatever a page
    /// boundary landed on.
    #[tokio::test]
    async fn test_backend_key_recovers_the_spelling_the_service_used() {
        let (operator, _) = operator_over(&["db/az", "db/a~"], true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister.list_page(Some("db/"), None, None).await.unwrap();

        assert_eq!(names(&page), vec!["db/az", "db/a%7E"]);
        let keys: Vec<Cow<'_, str>> = page
            .children
            .objects
            .iter()
            .map(|object| lister.backend_key(object.location.as_ref()))
            .collect();
        assert_eq!(keys, vec!["db/az", "db/a~"]);
    }

    /// The cursor arrives relative to the prefix, and only the prefix needs decoding: it came
    /// from an `object_store` path, while the cursor is part of a key this lister reported.
    #[tokio::test]
    async fn test_resolves_the_cursor_against_the_decoded_prefix() {
        let (operator, calls) = operator_over(&["db~/a.lance/", "db~/b.lance/"], true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister
            .list_page(Some("db%7E/"), Some("a.lance/"), Some(1))
            .await
            .unwrap();

        assert_eq!(names(&page), vec!["db%7E/b.lance/"]);
        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].path, "db~/");
        assert_eq!(calls[0].start_after.as_deref(), Some("db~/a.lance/"));
    }

    /// A service that cannot resume gets no lister at all, so listing falls back through
    /// the store and its wrappers rather than restarting from the top on every page.
    #[tokio::test]
    async fn test_declines_a_service_that_cannot_resume() {
        let (operator, _) = fake_operator(false);
        assert!(OpendalDirLister::for_operator(operator).is_none());
    }

    /// Vended-credential stores install a lister before their capabilities are known, so
    /// the unsupported case still has to be correct: return everything, unpaged, and let
    /// the caller apply the cursor.
    #[tokio::test]
    async fn test_unsupported_cursor_returns_the_whole_directory() {
        let (operator, calls) = fake_operator(false);
        let lister = OpendalDirLister {
            source: OperatorSource::Static(operator),
        };

        let page = lister
            .list_page(Some("db/"), Some("b.lance/"), Some(1))
            .await
            .unwrap();

        assert_eq!(names(&page).len(), CHILDREN.len());
        assert!(!page.has_more);
        assert_eq!(calls.lock().unwrap()[0].limit, None);
    }
}
