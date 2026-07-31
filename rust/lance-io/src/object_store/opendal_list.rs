// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Paginated directory listing for OpenDAL-backed stores.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use object_store::{ListResult, ObjectMeta, path::Path};
use opendal::{Entry, Operator, raw::Timestamp, raw::percent_decode_path};

use lance_core::{Error, Result};

use super::read_dir::{DirPage, PaginatedDirLister};

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
    Dynamic(Arc<super::dynamic_opendal::DynamicOpenDalStore>),
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
        operator
            .info()
            .full_capability()
            .list_with_start_after
            .then(|| {
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
        store: Arc<super::dynamic_opendal::DynamicOpenDalStore>,
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
        let can_resume = operator.info().full_capability().list_with_start_after;

        // OpenDAL takes raw paths; `object_store` may hand us percent-encoded ones.
        let path = prefix.map(percent_decode_path).unwrap_or_default();
        let mut request = operator.lister_with(&path);

        // Without pushdown the page has to be the whole directory: limiting an unfiltered
        // listing would cut it off before reaching the children the cursor asked for.
        let page_size = match (start_after, can_resume) {
            (Some(start_after), true) => {
                request = request.start_after(&percent_decode_path(start_after));
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

    use opendal::OperatorBuilder;
    use opendal::raw::{Access, AccessorInfo, OpList, RpList, oio};
    use opendal::{Capability, EntryMode, Metadata};

    use super::*;

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

    impl Access for FakeService {
        type Reader = ();
        type Writer = ();
        type Lister = FakeLister;
        type Deleter = ();
        type Copier = ();

        fn info(&self) -> Arc<AccessorInfo> {
            let info = AccessorInfo::default();
            info.set_scheme("fake").set_name("fake").set_root("/");
            info.set_native_capability(Capability {
                list: true,
                list_with_start_after: self.supports_start_after,
                list_with_limit: true,
                ..Default::default()
            });
            Arc::new(info)
        }

        async fn list(&self, path: &str, args: OpList) -> opendal::Result<(RpList, Self::Lister)> {
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
                        // OpenDAL's completion layer stats any file entry that arrives
                        // without a content length, which real services never do, so the
                        // fake has to set one too.
                        Metadata::new(EntryMode::FILE).with_content_length(child.len() as u64)
                    };
                    oio::Entry::new(child, metadata)
                })
                .collect::<Vec<_>>();
            Ok((RpList::default(), FakeLister(entries.into_iter())))
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
        let calls = Arc::new(Mutex::new(Vec::new()));
        let service = FakeService {
            children: CHILDREN.iter().map(|c| c.to_string()).collect(),
            supports_start_after,
            calls: calls.clone(),
        };
        (OperatorBuilder::new(service).finish(), calls)
    }

    fn names(page: &DirPage) -> Vec<String> {
        page.children
            .common_prefixes
            .iter()
            .chain(page.children.objects.iter().map(|object| &object.location))
            .map(|location| location.as_ref().to_string())
            .collect()
    }

    #[tokio::test]
    async fn test_pushes_cursor_and_page_size_into_the_request() {
        let (operator, calls) = fake_operator(true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister
            .list_page(Some("db/"), Some("db/a.lance/"), Some(2))
            .await
            .unwrap();

        assert_eq!(names(&page), vec!["db/b.lance", "db/c.lance"]);
        assert!(page.has_more);
        assert_eq!(
            *calls.lock().unwrap(),
            vec![ListCall {
                path: "db/".to_string(),
                start_after: Some("db/a.lance/".to_string()),
                // One more than the page, to tell whether another child follows.
                limit: Some(3),
            }]
        );
    }

    #[tokio::test]
    async fn test_last_page_reports_no_more() {
        let (operator, _) = fake_operator(true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister
            .list_page(Some("db/"), Some("db/b.lance/"), Some(5))
            .await
            .unwrap();

        assert_eq!(names(&page), vec!["db/c.lance", "db/loose.txt"]);
        assert!(!page.has_more);
    }

    #[tokio::test]
    async fn test_separates_directories_from_files() {
        let (operator, _) = fake_operator(true);
        let lister = OpendalDirLister::for_operator(operator).unwrap();

        let page = lister.list_page(Some("db/"), None, None).await.unwrap();

        assert_eq!(page.children.common_prefixes.len(), 3);
        assert_eq!(page.children.objects.len(), 1);
        assert_eq!(page.children.objects[0].location.as_ref(), "db/loose.txt");
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
            .list_page(Some("db/"), Some("db/b.lance/"), Some(1))
            .await
            .unwrap();

        assert_eq!(names(&page).len(), CHILDREN.len());
        assert!(!page.has_more);
        assert_eq!(calls.lock().unwrap()[0].limit, None);
    }
}
