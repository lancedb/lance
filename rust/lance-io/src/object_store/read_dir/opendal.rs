// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One directory level at a time over OpenDAL's `Lister`, for OpenDAL-backed stores.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use object_store::{ObjectMeta, path::Path};
use opendal::{Entry, Operator, raw::Timestamp, raw::percent_decode_path};

use lance_core::{Error, Result};

use super::{DELIMITER, DirCursor, DirEntry, DirEntryKind, DirPage, PaginatedDirLister};

/// How many children to fetch per underlying request when the caller does not ask for a
/// specific number. Matches the page size object stores default to.
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
///
/// OpenDAL keeps its continuation state inside the `Lister` and hands nothing back that would
/// survive the request, so pages here resume from a key: the last one the page reported. That
/// only reaches the rest of a directory on a service that lists in key order, which is why the
/// providers install this lister only for stores that do.
pub struct OpendalDirLister {
    source: OperatorSource,
}

impl OpendalDirLister {
    /// A lister for `operator`, or `None` when the service cannot resume a listing.
    ///
    /// Without `start_after` every page would restart at the beginning of the directory, which
    /// is worse than falling back to a full listing: the fallback at least goes through the
    /// wrappers around the store.
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
    /// installs a lister; [`Self::list_page`] falls back to walking the directory from the top
    /// per page if the service turns out not to support `start_after`.
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
        resume: Option<&DirCursor>,
        limit: Option<usize>,
    ) -> Result<DirPage> {
        let operator = self.operator().await?;
        let capability = operator.info().capability();

        // OpenDAL takes the service's own keys; `object_store` may hand us a percent-encoded
        // prefix. The cursor needs no such treatment: it is part of a key this lister reported.
        let path = prefix.map(percent_decode_path).unwrap_or_default();
        let resume = resume.map(DirCursor::expect_key).transpose()?;

        let mut request = operator.lister_with(&path);
        if let (Some(resume), true) = (resume, capability.list_with_start_after) {
            request = request.start_after(&format!("{path}{resume}"));
        }
        if capability.list_with_limit {
            // A hint for the requests behind the lister, not a bound on what it yields: the
            // lister keeps fetching, and the loop below is what stops at the page.
            request = request.limit(limit.unwrap_or(DEFAULT_PAGE_SIZE));
        }

        let mut lister = request.await.map_err(|err| list_error(&path, err))?;
        let mut entries = Vec::new();
        let mut last_key = None;
        let mut has_more = false;
        while let Some(entry) = lister
            .try_next()
            .await
            .map_err(|err| list_error(&path, err))?
        {
            let Some((key, child)) = dir_entry(&path, &entry) else {
                continue;
            };
            // Applied whether or not the service took the cursor. A service that did not is
            // listing from the top, and one that did still reports a resumed directory again:
            // every key inside `foo/` sorts after `foo/`, so they collapse back into it.
            if resume.is_some_and(|resume| key.as_str() <= resume) {
                continue;
            }
            // One entry past the page, so whether another child follows is answered from the
            // listing rather than by the caller coming back for an empty page.
            if limit.is_some_and(|limit| entries.len() == limit) {
                has_more = true;
                break;
            }
            last_key = Some(key);
            entries.push(child);
        }

        Ok(DirPage {
            entries,
            // A page that stopped short of the limit reached the end of the directory, so
            // there is nothing to resume from.
            next: has_more.then(|| DirCursor::key(last_key.unwrap_or_default())),
        })
    }
}

fn list_error(path: &str, err: opendal::Error) -> Error {
    Error::io(format!("failed to list '{path}': {err}"))
}

fn to_datetime(timestamp: Timestamp) -> Option<DateTime<Utc>> {
    let timestamp = timestamp.into_inner();
    DateTime::from_timestamp(timestamp.as_second(), timestamp.subsec_nanosecond() as u32)
}

/// A listed OpenDAL entry as a child of `prefix`, with the key the service listed it under
/// relative to that prefix, or `None` if it is not a child of this level: the lister reports
/// the directory being listed as well as what is in it.
///
/// The location on a file entry is spelled the way `OpendalStore` addresses a key, since that
/// store percent-decodes whatever it is handed. Reporting the service's raw spelling instead
/// would hand back a location that reads a different key — or none at all.
///
/// File metadata is whatever the listing reported. Unlike `object_store_opendal`, a missing
/// timestamp does not trigger a `stat` per child: a page should cost one request, and callers
/// that need full metadata can ask for it. Sizes need no such care, since OpenDAL's own
/// completion layer guarantees a file entry carries its content length.
fn dir_entry(prefix: &str, entry: &Entry) -> Option<(String, DirEntry)> {
    let key = entry.path().strip_prefix(prefix)?;
    let name = key.strip_suffix(DELIMITER).unwrap_or(key);
    if name.is_empty() {
        return None;
    }
    let meta = entry.metadata();
    let kind = match meta.is_dir() {
        true => DirEntryKind::Directory,
        false => DirEntryKind::File(ObjectMeta {
            location: Path::from(entry.path()),
            last_modified: meta
                .last_modified()
                .and_then(to_datetime)
                .unwrap_or_default(),
            size: meta.content_length(),
            e_tag: meta.etag().map(String::from),
            version: meta.version().map(String::from),
        }),
    };
    Some((
        key.to_string(),
        DirEntry {
            name: name.to_string(),
            kind,
        },
    ))
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

    /// The arguments of one list request, as the service saw them.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ListCall {
        path: String,
        start_after: Option<String>,
        limit: Option<usize>,
    }

    /// A service holding one flat directory, so a test can see exactly what the lister asked
    /// the backend for. Directories end in a delimiter, as OpenDAL reports them.
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

    fn names(page: &DirPage) -> Vec<&str> {
        page.entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect()
    }

    /// What the lister asks the service for, and what it makes of the answer.
    #[rstest::rstest]
    // A request that asks for no particular page still asks for a bounded one.
    #[case::whole_directory(None, None, Some(DEFAULT_PAGE_SIZE), vec!["a.lance", "b.lance", "c.lance", "loose.txt"], None)]
    #[case::truncated(Some("a.lance/"), Some(2), Some(2), vec!["b.lance", "c.lance"], Some("c.lance/"))]
    #[case::last_page(Some("b.lance/"), Some(5), Some(5), vec!["c.lance", "loose.txt"], None)]
    #[tokio::test]
    async fn test_pushes_cursor_and_page_size_into_the_request(
        #[case] resume: Option<&str>,
        #[case] limit: Option<usize>,
        #[case] requested_limit: Option<usize>,
        #[case] expected: Vec<&str>,
        #[case] next_key: Option<&str>,
    ) {
        let (operator, calls) = fake_operator(true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister
            .list_page(Some("db/"), resume.map(DirCursor::key).as_ref(), limit)
            .await
            .unwrap();

        assert_eq!(names(&page), expected);
        assert_eq!(page.next, next_key.map(DirCursor::key));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![ListCall {
                path: "db/".to_string(),
                start_after: resume.map(|cursor| format!("db/{cursor}")),
                limit: requested_limit,
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
        let DirEntryKind::File(meta) = &page.entries[0].kind else {
            panic!("expected a file entry");
        };

        assert_eq!(
            store
                .get(&meta.location)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            b"expected".as_slice()
        );
    }

    /// Paging follows the service's own keys, which is what the lister reports them under, so a
    /// name holding a character `Path::from` would encode still pages in the backend's order.
    /// Comparing the encoded locations instead would reorder the listing — `a~` is addressed as
    /// `a%7E`, which sorts before `az` — and paging would drop whatever the boundary landed on.
    #[tokio::test]
    async fn test_paging_follows_the_key_the_service_listed() {
        let (operator, _) = operator_over(&["db/az", "db/a~"], true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let first = lister.list_page(Some("db/"), None, Some(1)).await.unwrap();
        let second = lister
            .list_page(Some("db/"), first.next.as_ref(), Some(1))
            .await
            .unwrap();

        assert_eq!(names(&first), vec!["az"]);
        assert_eq!(first.next, Some(DirCursor::key("az")));
        assert_eq!(names(&second), vec!["a~"]);
        assert_eq!(second.next, None);
    }

    /// The cursor arrives relative to the prefix, and only the prefix needs decoding: it came
    /// from an `object_store` path, while the cursor is part of a key this lister reported.
    #[tokio::test]
    async fn test_resolves_the_cursor_against_the_decoded_prefix() {
        let (operator, calls) = operator_over(&["db~/a.lance/", "db~/b.lance/"], true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister
            .list_page(Some("db%7E/"), Some(&DirCursor::key("a.lance/")), Some(1))
            .await
            .unwrap();

        assert_eq!(names(&page), vec!["b.lance"]);
        let calls = calls.lock().unwrap();
        assert_eq!(calls[0].path, "db~/");
        assert_eq!(calls[0].start_after.as_deref(), Some("db~/a.lance/"));
    }

    /// A service that cannot resume gets no lister at all, so listing falls back through the
    /// store and its wrappers rather than restarting from the top on every page.
    #[tokio::test]
    async fn test_declines_a_service_that_cannot_resume() {
        let (operator, _) = fake_operator(false);
        assert!(OpendalDirLister::for_operator(operator).is_none());
    }

    /// Vended-credential stores install a lister before their capabilities are known, so the
    /// unsupported case still has to be correct: walk the directory from the top and apply the
    /// cursor here, which costs more requests but hands back the same page.
    #[tokio::test]
    async fn test_a_service_that_cannot_resume_is_walked_from_the_top() {
        let (operator, calls) = fake_operator(false);
        let lister = OpendalDirLister {
            source: OperatorSource::Static(operator),
        };

        let page = lister
            .list_page(Some("db/"), Some(&DirCursor::key("b.lance/")), Some(1))
            .await
            .unwrap();

        assert_eq!(names(&page), vec!["c.lance"]);
        assert_eq!(page.next, Some(DirCursor::key("c.lance/")));
        assert_eq!(calls.lock().unwrap()[0].start_after, None);
    }

    /// A continuation token is not something this lister can resume from, so one arriving here
    /// is rejected rather than silently listing the directory from the top.
    #[tokio::test]
    async fn test_rejects_a_continuation_token() {
        let (operator, _) = fake_operator(true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let err = lister
            .list_page(Some("db/"), Some(&DirCursor::backend("token")), None)
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    }
}
