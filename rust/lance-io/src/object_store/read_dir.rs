// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Paginated listing of a single directory level.
//!
//! [`ObjectStore::read_dir_stream`] yields the immediate children of a prefix and, where
//! the backend supports it, pushes the resume position and the page size down into the
//! list request. A caller that needs the first few children then pays for the first few
//! children rather than for the whole prefix.

use std::collections::VecDeque;
use std::sync::Arc;

use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use object_store::list::{PaginatedListOptions, PaginatedListStore};
use object_store::{ListResult, ObjectMeta, ObjectStore as OSObjectStore, path::Path};
use tracing::instrument;

use lance_core::{Error, Result};

use super::ObjectStore;
use crate::utils::tracking_store::IOTracker;

#[cfg(feature = "metrics")]
use crate::object_store::metrics::{InFlightGuard, record_outcome};
#[cfg(feature = "metrics")]
use std::time::Instant;

/// The path delimiter that separates directory levels.
const DELIMITER: &str = "/";

/// The character immediately after [`DELIMITER`] in key order.
///
/// Resuming a listing after a child directory means resuming after everything inside it,
/// and every key inside `foo/` starts with `foo/`. Replacing the trailing delimiter with
/// the next character gives a position that sits after all of them: `foo0` is greater than
/// `foo/anything`, and nothing can sort between the two, because `/` and `0` are adjacent.
const AFTER_DELIMITER: &str = "0";

/// Operation label for the metrics and IO statistics a paginated listing records.
const LIST_OP: &str = "list_paginated";

/// A position within a directory listing, used to resume where a previous listing stopped.
///
/// A cursor is only meaningful for the directory it came from. It is deliberately not a
/// plain child name: a child directory and a child file with the same name sit at
/// different positions, so they are constructed differently. Prefer [`DirEntry::cursor`]
/// over building one by hand.
///
/// Resuming after a child directory `foo` means resuming after every key under `foo/`, so
/// the position is written `foo0`: `0` is the character after `/`, which puts it after
/// everything inside the directory and before any sibling. A sibling file named exactly
/// `foo0` shares that position and is skipped along with the directory. Nothing else can
/// sort between the two.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DirCursor(String);

impl DirCursor {
    /// Resume immediately after the child directory `name`, skipping everything inside it.
    ///
    /// `name` is a single child name, as [`DirEntry::name`] reports it; a trailing delimiter
    /// is accepted and ignored, since that is how listings and URLs often spell a directory.
    pub fn after_directory(name: impl AsRef<str>) -> Self {
        let name = name
            .as_ref()
            .strip_suffix(DELIMITER)
            .unwrap_or(name.as_ref());
        Self(format!("{name}{AFTER_DELIMITER}"))
    }

    /// Resume immediately after the child file `name`, which is a single child name.
    pub fn after_file(name: impl AsRef<str>) -> Self {
        Self(name.as_ref().to_string())
    }
}

/// What kind of child a [`DirEntry`] is.
#[derive(Debug, Clone)]
pub enum DirEntryKind {
    /// A child directory: a common prefix, which has no metadata of its own.
    Directory,
    /// A child object, with the metadata the listing reported for it.
    File(ObjectMeta),
}

/// An immediate child of a directory.
#[derive(Debug, Clone)]
pub struct DirEntry {
    /// The child's name relative to the directory that was listed, with no trailing delimiter.
    pub name: String,
    /// Whether the child is a directory or a file, and the file's metadata if it is one.
    pub kind: DirEntryKind,
}

impl DirEntry {
    /// Whether this child is a directory rather than an object.
    pub fn is_dir(&self) -> bool {
        matches!(self.kind, DirEntryKind::Directory)
    }

    /// A cursor that resumes listing immediately after this entry.
    pub fn cursor(&self) -> DirCursor {
        match self.kind {
            DirEntryKind::Directory => DirCursor::after_directory(&self.name),
            DirEntryKind::File(_) => DirCursor::after_file(&self.name),
        }
    }

    /// The entry's position within its directory, as a cursor that sits just past it.
    /// Sorting by this key gives the order the store lists the entries in, which is not the
    /// same as ordering by name: `foo-bar/` sorts before `foo/` because `-` sorts before
    /// `/`. Replacing the delimiter with the character after it keeps that order, since
    /// nothing sorts between the two.
    fn sort_key(&self) -> String {
        self.cursor().0
    }
}

/// Options for [`ObjectStore::read_dir_stream`].
#[derive(Debug, Clone, Default)]
pub struct ReadDirOptions {
    /// Resume immediately after this position instead of starting at the beginning.
    pub resume_from: Option<DirCursor>,
    /// How many entries to request per underlying list call. Must be at least one.
    ///
    /// This is a hint; backends may return fewer. Set it to the number of entries actually
    /// wanted so that a short listing does not fetch a full page.
    pub page_size: Option<usize>,
}

/// One page of a directory listing.
pub struct DirPage {
    /// The children found, as the backend reported them.
    pub children: ListResult,
    /// Whether the backend has more children after this page. A page can be short and
    /// still be followed by more, so this cannot be inferred from `children`.
    pub has_more: bool,
}

/// A backend that can list one directory level a page at a time.
///
/// Kept separate from [`ObjectStore::inner`] because paginated listing is not part of the
/// `ObjectStore` trait and so cannot be reached through a `dyn ObjectStore`.
///
/// This is deliberately cursor-shaped rather than token-shaped. `object_store` resumes
/// from either an offset or an opaque continuation token, but OpenDAL's `Lister` keeps
/// its continuation state to itself and only accepts a `start_after` key, so a token is
/// not something every backend can hand back. A key is.
#[async_trait::async_trait]
pub trait PaginatedDirLister: std::fmt::Debug + Send + Sync + 'static {
    /// List the immediate children of `prefix` in key order, starting after `start_after`.
    ///
    /// `prefix` carries a trailing delimiter, and `start_after` is a key relative to the
    /// store root, so a child directory's key ends in one too. `limit` caps the number of
    /// children in the page; it is a request, and a backend may return fewer and still set
    /// [`DirPage::has_more`].
    ///
    /// A backend that cannot resume from `start_after` must return the whole directory
    /// with `has_more` false, so that the caller can apply the cursor itself.
    async fn list_page(
        &self,
        prefix: Option<&str>,
        start_after: Option<&str>,
        limit: Option<usize>,
    ) -> Result<DirPage>;
}

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
        let page = self
            .0
            .list_paginated(
                prefix,
                PaginatedListOptions {
                    offset: start_after.map(String::from),
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

impl ObjectStore {
    /// Stream the immediate children of `dir`, one directory level deep.
    ///
    /// Entries come back in the order the store lists them, which is by storage key rather
    /// than by name, and the stream paginates internally, so a caller that filters entries
    /// and stops early only pays for the pages it consumed.
    ///
    /// On backends with a paginated list API — S3, GCS and Azure, plus OpenDAL-backed
    /// stores whose service can resume from a key — the resume position and page size are
    /// pushed into the list request. Elsewhere the directory is listed in full and the
    /// cursor is applied locally, which is correct but no cheaper than [`Self::read_dir`].
    ///
    /// ```
    /// # use futures::TryStreamExt;
    /// # use lance_io::object_store::{ObjectStore, ReadDirOptions};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<Vec<String>> {
    /// let mut children = store.read_dir_stream("my_db", ReadDirOptions::default());
    /// let mut tables = Vec::new();
    /// while let Some(entry) = children.try_next().await? {
    ///     if let Some(name) = entry.name.strip_suffix(".lance") {
    ///         tables.push(name.to_string());
    ///     }
    ///     if tables.len() == 10 {
    ///         break;
    ///     }
    /// }
    /// # Ok(tables)
    /// # }
    /// ```
    pub fn read_dir_stream(
        &self,
        dir: impl Into<Path>,
        options: ReadDirOptions,
    ) -> BoxStream<'static, Result<DirEntry>> {
        let dir = dir.into();
        // A page of nothing cannot advance a listing, and the two paths below would disagree
        // about what it means: the pushdown path would report an empty directory while the
        // full listing ignored the page size and returned everything.
        if options.page_size == Some(0) {
            return stream::once(async {
                Err(Error::invalid_input(
                    "read_dir_stream page_size must be at least 1, got 0",
                ))
            })
            .boxed();
        }
        match &self.paginated_lister {
            Some(lister) => paginated_stream(
                lister.clone(),
                dir,
                options,
                self.io_tracker.clone(),
                #[cfg(feature = "metrics")]
                self.store_prefix.clone(),
            ),
            // Goes through `inner`, so the wrappers around it instrument the request. The
            // pushdown path talks to the backend directly and instruments itself.
            None => full_listing_stream(self.inner.clone(), dir, options),
        }
    }
}

/// The prefix to list under, carrying the trailing delimiter that the paginated API expects.
/// `None` for the root of the store, which has no prefix at all.
fn list_prefix(dir: &Path) -> Option<String> {
    let dir = dir.as_ref();
    (!dir.is_empty()).then(|| format!("{dir}{DELIMITER}"))
}

/// Build an entry for a listed child, or `None` if it is not one.
///
/// Skips the directory marker that some stores keep for the prefix itself, which lists as
/// an object whose location is the directory.
fn dir_entry(dir: &Path, location: &Path, meta: Option<&ObjectMeta>) -> Option<DirEntry> {
    if location == dir {
        return None;
    }
    Some(DirEntry {
        name: location.filename()?.to_string(),
        kind: match meta {
            Some(meta) => DirEntryKind::File(meta.clone()),
            None => DirEntryKind::Directory,
        },
    })
}

/// Collect one page of listed children into stream order, dropping anything at or before
/// the cursor.
///
/// Both steps matter. Stores return common prefixes and objects as separate lists, so the
/// two have to be merged to get a single ordered sequence. And the entry the cursor points
/// at can come back again: the cursor names a prefix rather than a key, so S3's exclusive
/// `start-after` does not exclude the keys inside it, and GCS's `startOffset` is inclusive
/// to begin with.
fn page_entries(
    common_prefixes: &[Path],
    objects: &[ObjectMeta],
    dir: &Path,
    resume_from: Option<&str>,
) -> Page {
    let entries = common_prefixes
        .iter()
        .filter_map(|prefix| dir_entry(dir, prefix, None))
        .chain(
            objects
                .iter()
                .filter_map(|object| dir_entry(dir, &object.location, Some(object))),
        );
    let mut keyed: Vec<(String, DirEntry)> =
        entries.map(|entry| (entry.sort_key(), entry)).collect();
    keyed.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

    // The last key the backend reported, whether or not it survives the filter. The next
    // page resumes from here, so it has to advance even when a page filters away entirely.
    let last_key = keyed.last().map(|(key, _)| key.clone());
    let entries = keyed
        .into_iter()
        .filter(|(key, _)| resume_from.is_none_or(|cursor| key.as_str() > cursor))
        .map(|(_, entry)| entry)
        .collect();
    Page { entries, last_key }
}

struct Page {
    entries: VecDeque<DirEntry>,
    last_key: Option<String>,
}

struct PaginatedState {
    lister: Arc<dyn PaginatedDirLister>,
    dir: Path,
    prefix: Option<String>,
    page_size: Option<usize>,
    /// Where the next page resumes, relative to the directory being listed (the same
    /// space as [`DirEntry::sort_key`]). `None` before the first request of an unresumed
    /// listing.
    cursor: Option<String>,
    has_more: bool,
    started: bool,
    buffered: VecDeque<DirEntry>,
    io_tracker: IOTracker,
    #[cfg(feature = "metrics")]
    store_prefix: String,
}

impl PaginatedState {
    /// Fetch one page from the backend.
    ///
    /// Instrumented here rather than by the wrappers around [`ObjectStore::inner`]: this
    /// path holds the backend directly, so it never passes through them.
    #[instrument(
        level = "debug",
        skip_all,
        fields(prefix = self.prefix.as_deref(), start_after = self.cursor.as_deref())
    )]
    async fn list_page(&self) -> Result<DirPage> {
        self.io_tracker
            .record_read(LIST_OP, self.dir.clone(), 0, None);
        #[cfg(feature = "metrics")]
        let _in_flight = InFlightGuard::new(&self.store_prefix, LIST_OP);
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        // The backend resumes from a key, which carries the directory prefix.
        let start_after = self
            .cursor
            .as_ref()
            .map(|cursor| format!("{}{}", self.prefix.as_deref().unwrap_or(""), cursor));
        // Backends whose resume position is inclusive hand back the entry the cursor points
        // at, which is dropped again here. Asking for one more keeps a page of the wanted
        // size after that drop, and keeps a page of size one from being nothing but the
        // entry the previous page ended on.
        let limit = match (&start_after, self.page_size) {
            (Some(_), Some(page_size)) => Some(page_size.saturating_add(1)),
            (_, page_size) => page_size,
        };
        let page = self
            .lister
            .list_page(self.prefix.as_deref(), start_after.as_deref(), limit)
            .await;

        #[cfg(feature = "metrics")]
        record_outcome(&self.store_prefix, LIST_OP, start, 0, page.is_err());
        page
    }
}

fn paginated_stream(
    lister: Arc<dyn PaginatedDirLister>,
    dir: Path,
    options: ReadDirOptions,
    io_tracker: IOTracker,
    #[cfg(feature = "metrics")] store_prefix: String,
) -> BoxStream<'static, Result<DirEntry>> {
    let state = PaginatedState {
        lister,
        cursor: options.resume_from.map(|cursor| cursor.0),
        prefix: list_prefix(&dir),
        dir,
        page_size: options.page_size,
        has_more: false,
        started: false,
        buffered: VecDeque::new(),
        io_tracker,
        #[cfg(feature = "metrics")]
        store_prefix,
    };

    stream::try_unfold(state, |mut state| async move {
        while state.buffered.is_empty() {
            if state.started && !state.has_more {
                return Ok(None);
            }

            let page = state.list_page().await?;
            let listed = page_entries(
                &page.children.common_prefixes,
                &page.children.objects,
                &state.dir,
                state.cursor.as_deref(),
            );

            state.started = true;
            match listed.last_key {
                // Advance past everything the backend reported, not just what survived the
                // cursor, so a page that filters away entirely still makes progress.
                Some(last_key) if Some(&last_key) > state.cursor.as_ref() => {
                    state.cursor = Some(last_key);
                    state.has_more = page.has_more;
                }
                Some(_) => match state.page_size {
                    // The page held nothing beyond the position it started from. Widening it
                    // is the only way forward, and it terminates: once the page covers the
                    // rest of the directory the backend stops reporting more.
                    Some(page_size) => {
                        state.page_size = Some(page_size.saturating_mul(2));
                        state.has_more = page.has_more;
                    }
                    // Nothing new and no page size to widen, so asking again would repeat the
                    // same request. Stop, as for the empty page below.
                    None => state.has_more = false,
                },
                // A backend claiming more pages while returning nothing leaves no position
                // to resume from; end the listing rather than ask again forever.
                None => state.has_more = false,
            }
            state.buffered = listed.entries;
        }

        Ok(state.buffered.pop_front().map(|entry| (entry, state)))
    })
    .boxed()
}

fn full_listing_stream(
    store: Arc<dyn OSObjectStore>,
    dir: Path,
    options: ReadDirOptions,
) -> BoxStream<'static, Result<DirEntry>> {
    stream::once(async move {
        let listed = store.list_with_delimiter(Some(&dir)).await?;
        let entries = page_entries(
            &listed.common_prefixes,
            &listed.objects,
            &dir,
            options.resume_from.as_ref().map(|cursor| cursor.0.as_str()),
        );
        Result::Ok(stream::iter(entries.entries.into_iter().map(Result::Ok)))
    })
    .try_flatten()
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::object_store::{ObjectStoreParams, ObjectStoreRegistry};
    use chrono::Utc;
    use object_store::memory::InMemory;
    use object_store::{ListResult, ObjectStoreExt, PutPayload};
    use rstest::rstest;

    /// How the store under test resolves a listing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Backend {
        /// No paginated API: list the whole directory and apply the cursor locally.
        FullListing,
        /// A paginated API whose offset is exclusive, like S3's `start-after`.
        ExclusiveOffset,
        /// A paginated API whose offset is inclusive, like GCS's `startOffset`.
        InclusiveOffset,
    }

    /// One list request, as the backend saw it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ListRequest {
        start_after: Option<String>,
        limit: Option<usize>,
    }

    /// A stand-in for a backend with a paginated list API, modelled on S3.
    ///
    /// `limit` is spent on the keys the backend scans rather than on the entries it
    /// returns, which is what a real store with a delimiter does: a common prefix
    /// collapses many keys into one entry but still costs what it collapsed. A page can
    /// therefore come back short and still be truncated, so `has_more` is the only way to
    /// know whether a listing is finished.
    #[derive(Debug)]
    struct FakeDirLister {
        keys: Vec<String>,
        offset_is_inclusive: bool,
        requests: Arc<Mutex<Vec<ListRequest>>>,
    }

    #[async_trait::async_trait]
    impl PaginatedDirLister for FakeDirLister {
        async fn list_page(
            &self,
            prefix: Option<&str>,
            start_after: Option<&str>,
            limit: Option<usize>,
        ) -> Result<DirPage> {
            self.requests.lock().unwrap().push(ListRequest {
                start_after: start_after.map(String::from),
                limit,
            });
            let prefix = prefix.unwrap_or("");
            let mut budget = limit.unwrap_or(1000);

            let mut children = ListResult {
                common_prefixes: Vec::new(),
                objects: Vec::new(),
            };
            let mut has_more = false;
            let mut idx = 0;

            while idx < self.keys.len() {
                let key = self.keys[idx].clone();
                let Some(rest) = key.strip_prefix(prefix) else {
                    idx += 1;
                    continue;
                };
                if let Some(start_after) = start_after {
                    let before = if self.offset_is_inclusive {
                        key.as_str() < start_after
                    } else {
                        key.as_str() <= start_after
                    };
                    if before {
                        idx += 1;
                        continue;
                    }
                }
                if budget == 0 {
                    has_more = true;
                    break;
                }
                match rest.find(DELIMITER) {
                    Some(end) => {
                        let child = format!("{}{}{}", prefix, &rest[..end], DELIMITER);
                        children.common_prefixes.push(Path::from(child.as_str()));
                        // A common prefix collapses every key beneath it into one entry,
                        // and is charged for every key it collapsed.
                        while idx < self.keys.len() && self.keys[idx].starts_with(&child) {
                            idx += 1;
                            budget = budget.saturating_sub(1);
                        }
                    }
                    None => {
                        children.objects.push(ObjectMeta {
                            location: Path::from(key.as_str()),
                            last_modified: Utc::now(),
                            size: 1,
                            e_tag: None,
                            version: None,
                        });
                        idx += 1;
                        budget -= 1;
                    }
                }
            }

            Ok(DirPage { children, has_more })
        }
    }

    struct TestStore {
        store: ObjectStore,
        requests: Arc<Mutex<Vec<ListRequest>>>,
    }

    impl TestStore {
        async fn names(&self, dir: &str, options: ReadDirOptions) -> Vec<String> {
            self.store
                .read_dir_stream(Path::from(dir), options)
                .map_ok(|entry| entry.name)
                .try_collect()
                .await
                .unwrap()
        }
    }

    async fn test_store(backend: Backend, keys: &[&str]) -> TestStore {
        let inner = Arc::new(InMemory::new());
        for key in keys {
            inner
                .put(&Path::from(*key), PutPayload::from_static(b"x"))
                .await
                .unwrap();
        }
        #[allow(deprecated)]
        let params = ObjectStoreParams {
            object_store: Some((inner, url::Url::parse("memory:///").unwrap())),
            ..Default::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory:///",
            &params,
        )
        .await
        .unwrap();
        let mut store = Arc::try_unwrap(store).unwrap();

        let requests = Arc::new(Mutex::new(Vec::new()));
        if backend != Backend::FullListing {
            let mut sorted: Vec<String> = keys.iter().map(|key| key.to_string()).collect();
            sorted.sort();
            store.paginated_lister = Some(Arc::new(FakeDirLister {
                keys: sorted,
                offset_is_inclusive: backend == Backend::InclusiveOffset,
                requests: requests.clone(),
            }));
        }
        TestStore { store, requests }
    }

    const TABLES: &[&str] = &[
        "db/a.lance/_versions/1.manifest",
        "db/a.lance/data/1.lance",
        "db/b.lance/data/1.lance",
        "db/c.lance/data/1.lance",
        "db/loose.txt",
        "other/d.lance/data/1.lance",
    ];

    #[rstest]
    #[tokio::test]
    async fn test_lists_one_level(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        assert_eq!(
            store.names("db", ReadDirOptions::default()).await,
            vec!["a.lance", "b.lance", "c.lance", "loose.txt"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_resumes_after_directory(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        let options = ReadDirOptions {
            resume_from: Some(DirCursor::after_directory("a.lance")),
            ..Default::default()
        };
        assert_eq!(
            store.names("db", options).await,
            vec!["b.lance", "c.lance", "loose.txt"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_resumes_after_file(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, &["db/a.txt", "db/b.txt", "db/c.txt"]).await;
        let options = ReadDirOptions {
            resume_from: Some(DirCursor::after_file("a.txt")),
            ..Default::default()
        };
        assert_eq!(store.names("db", options).await, vec!["b.txt", "c.txt"]);
    }

    /// Walking page by page must return every child exactly once, including names that sort
    /// after the cursor by name but before it by key (`foo-bar/` sorts before `foo/`).
    #[rstest]
    #[tokio::test]
    async fn test_paging_by_cursor_is_complete(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
        #[values(1, 2, 3)] page_size: usize,
    ) {
        let keys = [
            "db/foo-bar.lance/data/1.lance",
            "db/foo.lance/data/1.lance",
            "db/foo.lance/data/2.lance",
            "db/foo2.lance/data/1.lance",
            "db/zzz.txt",
        ];
        let store = test_store(backend, &keys).await;

        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let options = ReadDirOptions {
                resume_from: cursor.take(),
                page_size: Some(page_size),
            };
            let page: Vec<DirEntry> = store
                .store
                .read_dir_stream(Path::from("db"), options)
                .take(page_size)
                .try_collect()
                .await
                .unwrap();
            let Some(last) = page.last() else {
                break;
            };
            cursor = Some(last.cursor());
            seen.extend(page.into_iter().map(|entry| entry.name));
        }

        assert_eq!(
            seen,
            vec!["foo-bar.lance", "foo.lance", "foo2.lance", "zzz.txt"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_empty_directory(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        assert!(
            store
                .names("nonexistent", ReadDirOptions::default())
                .await
                .is_empty()
        );
    }

    /// A directory cursor spelled with a trailing delimiter is the same position as one
    /// without. Keeping it would resume inside the directory, which serves it up again.
    #[rstest]
    #[tokio::test]
    async fn test_resuming_after_a_directory_ignores_a_trailing_delimiter(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        assert_eq!(
            DirCursor::after_directory("a.lance/"),
            DirCursor::after_directory("a.lance")
        );

        let store = test_store(backend, TABLES).await;
        let options = ReadDirOptions {
            resume_from: Some(DirCursor::after_directory("a.lance/")),
            ..Default::default()
        };
        assert_eq!(
            store.names("db", options).await,
            vec!["b.lance", "c.lance", "loose.txt"]
        );
    }

    /// A page of nothing is rejected rather than left to mean whatever the backend makes of
    /// it: pushing it down reports an empty directory, and a full listing ignores it.
    #[rstest]
    #[tokio::test]
    async fn test_zero_page_size_is_rejected(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        let listed: Result<Vec<DirEntry>> = store
            .store
            .read_dir_stream(
                Path::from("db"),
                ReadDirOptions {
                    page_size: Some(0),
                    ..Default::default()
                },
            )
            .try_collect()
            .await;

        let err = listed.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(err.to_string().contains("page_size must be at least 1"));
        assert!(store.requests.lock().unwrap().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn test_entry_kinds(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        let entries: Vec<DirEntry> = store
            .store
            .read_dir_stream(Path::from("db"), ReadDirOptions::default())
            .try_collect()
            .await
            .unwrap();

        assert!(entries[0].is_dir());
        assert!(matches!(entries[0].kind, DirEntryKind::Directory));
        assert_eq!(entries[0].cursor(), DirCursor::after_directory("a.lance"));

        let loose = entries.last().unwrap();
        assert!(!loose.is_dir());
        let DirEntryKind::File(meta) = &loose.kind else {
            panic!("expected a file entry, got {:?}", loose.kind);
        };
        assert_eq!(meta.size, 1);
        assert_eq!(loose.cursor(), DirCursor::after_file("loose.txt"));
    }

    /// The point of the pushdown: a short listing costs one request no matter how many
    /// children the directory has.
    #[rstest]
    #[tokio::test]
    async fn test_short_listing_makes_one_request(
        #[values(Backend::ExclusiveOffset, Backend::InclusiveOffset)] backend: Backend,
    ) {
        let keys: Vec<String> = (0..500)
            .map(|i| format!("db/table_{i:03}.lance/data/1.lance"))
            .collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let store = test_store(backend, &keys).await;

        let first: Vec<DirEntry> = store
            .store
            .read_dir_stream(
                Path::from("db"),
                ReadDirOptions {
                    page_size: Some(1),
                    ..Default::default()
                },
            )
            .take(1)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(first[0].name, "table_000.lance");
        assert_eq!(store.requests.lock().unwrap().len(), 1);
    }

    /// The pushdown path holds the backend directly, so it has to record its own IO. A
    /// listing that is invisible to `io_tracker` would also be invisible to the metrics
    /// and tracing layers that sit in the same chain.
    #[rstest]
    #[tokio::test]
    async fn test_listing_is_recorded_in_io_stats(
        #[values(
            Backend::FullListing,
            Backend::ExclusiveOffset,
            Backend::InclusiveOffset
        )]
        backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        assert_eq!(store.store.io_tracker().stats().read_iops, 0);

        let _ = store
            .store
            .read_dir_stream(
                Path::from("db"),
                ReadDirOptions {
                    page_size: Some(2),
                    ..Default::default()
                },
            )
            .try_next()
            .await
            .unwrap();

        assert_eq!(store.store.io_tracker().stats().read_iops, 1);
    }

    /// A short page is not the end of a listing. With a delimiter the backend spends its
    /// limit on keys it collapses away, so a page can hold one entry, be nowhere near the
    /// limit, and still be followed by more. Stopping on a short page would drop children.
    #[rstest]
    #[tokio::test]
    async fn test_short_page_is_not_the_end_of_the_listing(
        #[values(Backend::ExclusiveOffset, Backend::InclusiveOffset)] backend: Backend,
    ) {
        // `big.lance` holds more keys than a page is allowed to scan, so the page that
        // reports it comes back with a single entry and `has_more` set.
        let store = test_store(
            backend,
            &[
                "db/big.lance/data/1.lance",
                "db/big.lance/data/2.lance",
                "db/big.lance/data/3.lance",
                "db/big.lance/data/4.lance",
                "db/zzz.txt",
            ],
        )
        .await;

        let names = store
            .names(
                "db",
                ReadDirOptions {
                    page_size: Some(3),
                    ..Default::default()
                },
            )
            .await;

        assert_eq!(names, vec!["big.lance", "zzz.txt"]);
        // The first page held one entry against a limit of three, and was still followed.
        let requests = store.requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            2,
            "the short first page should not have ended the listing: {requests:?}"
        );
    }

    /// Resuming asks for one entry more than the page, so that a backend re-serving the
    /// entry the cursor points at cannot use up the whole page.
    #[rstest]
    #[tokio::test]
    async fn test_resuming_asks_for_one_more_than_the_page(
        #[values(Backend::ExclusiveOffset, Backend::InclusiveOffset)] backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;

        let _ = store
            .names(
                "db",
                ReadDirOptions {
                    resume_from: Some(DirCursor::after_directory("a.lance")),
                    page_size: Some(1),
                },
            )
            .await;

        let requests = store.requests.lock().unwrap();
        assert_eq!(requests[0].start_after.as_deref(), Some("db/a.lance0"));
        assert_eq!(requests[0].limit, Some(2));
    }

    /// A backend that reports more pages while returning nothing has given no position to
    /// resume from. The listing has to end rather than ask for the same page forever.
    #[tokio::test]
    async fn test_empty_page_claiming_more_ends_the_listing() {
        #[derive(Debug)]
        struct AlwaysEmptyLister;

        #[async_trait::async_trait]
        impl PaginatedDirLister for AlwaysEmptyLister {
            async fn list_page(
                &self,
                _prefix: Option<&str>,
                _start_after: Option<&str>,
                _limit: Option<usize>,
            ) -> Result<DirPage> {
                Ok(DirPage {
                    children: ListResult {
                        common_prefixes: Vec::new(),
                        objects: Vec::new(),
                    },
                    has_more: true,
                })
            }
        }

        let mut store = test_store(Backend::ExclusiveOffset, TABLES).await;
        store.store.paginated_lister = Some(Arc::new(AlwaysEmptyLister));

        assert!(
            store
                .names("db", ReadDirOptions::default())
                .await
                .is_empty()
        );
    }

    /// A backend that keeps re-serving the entry a page started from is answered by widening
    /// the page. With no page size there is nothing to widen, so the listing has to stop
    /// instead of repeating the same request.
    #[tokio::test]
    async fn test_stuck_page_without_a_page_size_ends_the_listing() {
        /// Always reports the same child, and errors once it is clear the caller is looping.
        /// A repeat of ready futures never yields to the runtime, so a regression would hang
        /// rather than time out; failing from inside the backend keeps it a test failure.
        #[derive(Debug, Default)]
        struct StuckLister {
            calls: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl PaginatedDirLister for StuckLister {
            async fn list_page(
                &self,
                _prefix: Option<&str>,
                _start_after: Option<&str>,
                _limit: Option<usize>,
            ) -> Result<DirPage> {
                if self.calls.fetch_add(1, Ordering::SeqCst) >= 4 {
                    return Err(Error::io("the listing kept asking for the same page"));
                }
                Ok(DirPage {
                    children: ListResult {
                        common_prefixes: vec![Path::from("db/a.lance")],
                        objects: Vec::new(),
                    },
                    has_more: true,
                })
            }
        }

        let lister = Arc::new(StuckLister::default());
        let mut store = test_store(Backend::ExclusiveOffset, TABLES).await;
        store.store.paginated_lister = Some(lister.clone());

        assert_eq!(
            store.names("db", ReadDirOptions::default()).await,
            ["a.lance"]
        );
        // One page to find the child, one to discover there is nothing past it.
        assert_eq!(lister.calls.load(Ordering::SeqCst), 2);
    }

    /// A recording [`PaginatedListStore`], so the translation `NativeDirLister` performs
    /// can be checked without a cloud backend. This is the adapter the native S3, GCS and
    /// Azure stores actually use.
    #[derive(Debug, Default)]
    struct RecordingListStore {
        calls: Mutex<Vec<(Option<String>, PaginatedListOptions)>>,
        next_page_token: Option<String>,
    }

    #[async_trait::async_trait]
    impl PaginatedListStore for RecordingListStore {
        async fn list_paginated(
            &self,
            prefix: Option<&str>,
            opts: PaginatedListOptions,
        ) -> object_store::Result<object_store::list::PaginatedListResult> {
            self.calls
                .lock()
                .unwrap()
                .push((prefix.map(String::from), opts));
            Ok(object_store::list::PaginatedListResult {
                result: ListResult {
                    common_prefixes: vec![Path::from("db/a.lance")],
                    objects: Vec::new(),
                },
                page_token: self.next_page_token.clone(),
            })
        }
    }

    #[tokio::test]
    async fn test_native_lister_pushes_the_cursor_and_page_size_down() {
        let store = Arc::new(RecordingListStore::default());
        let lister = NativeDirLister::for_store(store.clone());

        let page = lister
            .list_page(Some("db/"), Some("db/a.lance0"), Some(7))
            .await
            .unwrap();

        let calls = store.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (prefix, opts) = &calls[0];
        assert_eq!(prefix.as_deref(), Some("db/"));
        assert_eq!(opts.offset.as_deref(), Some("db/a.lance0"));
        // Without a delimiter the listing would be recursive rather than one level deep.
        assert_eq!(opts.delimiter.as_deref(), Some("/"));
        assert_eq!(opts.max_keys, Some(7));
        // Resumption is by key, so the store's own continuation token is never used.
        assert_eq!(opts.page_token, None);
        assert_eq!(page.children.common_prefixes.len(), 1);
    }

    /// The native API reports truncation by handing back a token, not by filling the page.
    #[rstest]
    #[case::truncated(Some("token"), true)]
    #[case::last_page(None, false)]
    #[tokio::test]
    async fn test_native_lister_reports_more_from_the_page_token(
        #[case] next_page_token: Option<&str>,
        #[case] expected: bool,
    ) {
        let store = Arc::new(RecordingListStore {
            calls: Mutex::new(Vec::new()),
            next_page_token: next_page_token.map(String::from),
        });
        let lister = NativeDirLister::for_store(store);

        let page = lister.list_page(Some("db/"), None, None).await.unwrap();

        assert_eq!(page.has_more, expected);
    }
}
