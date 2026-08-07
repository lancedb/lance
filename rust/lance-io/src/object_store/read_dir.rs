// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Paginated listing of a single directory level.
//!
//! [`ObjectStore::read_dir_page`] returns one page of the immediate children of a prefix, plus
//! a token that resumes after it. Where the backend has a paginated list API the page size and
//! the resume position are pushed into the list request, so a caller that wants the first few
//! children pays for the first few children rather than for the whole prefix.
//!
//! The token is opaque. It carries whatever resumes the store it came from — a continuation
//! token where the store has one, a key where it does not — and a caller only ever hands it
//! back. That is what lets a store without key-ordered listings, such as S3 Express, be paged
//! at all: nothing outside this module compares one token to another.
//!
//! This module holds the listing itself: the cursor, the paging loop, and the
//! [`PaginatedDirLister`] contract a backend implements. The backends live beside it, one per
//! way of asking. There is one so far — [`native`], over `object_store`'s own paginated API,
//! which covers S3, GCS and Azure. Everything else lists in full and pages locally, which is
//! correct but costs what the whole directory costs.

use std::sync::Arc;

use object_store::{ListResult, ObjectMeta, ObjectStore as OSObjectStore, path::Path};
use tracing::instrument;

use lance_core::{Error, Result};

use super::ObjectStore;
use crate::utils::tracking_store::IOTracker;

#[cfg(all(test, feature = "aws", feature = "gcp", feature = "azure"))]
mod conformance;
#[cfg(all(test, feature = "aws", feature = "gcp", feature = "azure"))]
mod emulator;
// The paginated list API is only reached through the native cloud stores, and through the
// recording store the tests below stand in their place.
#[cfg(any(test, feature = "aws", feature = "azure", feature = "gcp"))]
pub mod native;
#[cfg(test)]
mod store_model;

#[cfg(feature = "metrics")]
use crate::object_store::metrics::{InFlightGuard, record_outcome};
#[cfg(feature = "metrics")]
use std::time::Instant;

/// The path delimiter that separates directory levels.
const DELIMITER: &str = "/";

/// Operation label for the metrics and IO statistics a paginated listing records.
const LIST_OP: &str = "list_paginated";

/// Marks the encoding a page token was written with, so that a token from another build is
/// rejected rather than read as something it is not.
const TOKEN_VERSION: char = '1';

/// Marks a cursor a backend minted, holding its own continuation token.
const BACKEND_TAG: char = 'b';
/// Marks a cursor the full-listing fallback minted, holding a key within the directory.
const KEY_TAG: char = 'k';

/// Where a listing resumes.
///
/// Opaque to callers: [`ObjectStore::read_dir_page`] mints one, hands it over as the string in
/// [`DirListing::next_token`], and takes it back in [`ReadDirOptions::page_token`]. What it
/// holds depends on the path that minted it, so a token means nothing anywhere else.
///
/// The string is the token exactly as a caller sees it: the version, then a tag naming the
/// path that minted it, then that path's own position. A cursor is only ever handed back to
/// the path that minted it, and the tag is what makes the other path reject it rather than
/// read it as something it is not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirCursor(String);

impl DirCursor {
    /// A backend's own continuation token, which resumes exactly where the page stopped.
    ///
    /// Nothing is compared against it, so a store whose listings are not in key order — S3
    /// Express, or an Azure account with a hierarchical namespace — pages correctly on one.
    pub(crate) fn backend(token: impl AsRef<str>) -> Self {
        Self::tagged(BACKEND_TAG, token.as_ref())
    }

    /// A key within the directory, which the next page resumes strictly after: the last key
    /// the page reported. What the full-listing fallback mints, having no token of its own.
    pub(crate) fn key(key: impl AsRef<str>) -> Self {
        Self::tagged(KEY_TAG, key.as_ref())
    }

    fn tagged(tag: char, value: &str) -> Self {
        Self(format!("{TOKEN_VERSION}{tag}{value}"))
    }

    /// The continuation token to resume from, for a backend that pages by its own token.
    #[cfg(any(test, feature = "aws", feature = "azure", feature = "gcp"))]
    pub(crate) fn expect_backend(&self) -> Result<&str> {
        self.untag(BACKEND_TAG).ok_or_else(|| {
            Error::invalid_input(
                "this page token came from a store that lists a directory in full, \
                 which this one cannot resume from",
            )
        })
    }

    /// The key to resume after, for a store that lists a directory in full.
    pub(crate) fn expect_key(&self) -> Result<&str> {
        self.untag(KEY_TAG).ok_or_else(|| {
            Error::invalid_input(
                "this page token came from a store that resumes by continuation token, \
                 which this one cannot use",
            )
        })
    }

    /// The position this cursor holds, if `tag` is the path that minted it.
    fn untag(&self, tag: char) -> Option<&str> {
        self.0.strip_prefix(TOKEN_VERSION)?.strip_prefix(tag)
    }

    fn encode(self) -> String {
        self.0
    }

    fn decode(token: &str) -> Result<Self> {
        let invalid = || Error::invalid_input(format!("not a directory page token: '{token}'"));
        let rest = token.strip_prefix(TOKEN_VERSION).ok_or_else(invalid)?;
        match rest.chars().next() {
            Some(BACKEND_TAG | KEY_TAG) => Ok(Self(token.to_string())),
            _ => Err(invalid()),
        }
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
}

/// Options for [`ObjectStore::read_dir_page`].
#[derive(Debug, Clone, Default)]
pub struct ReadDirOptions {
    /// Resume after the page a previous call returned, using the token it handed back.
    pub page_token: Option<String>,
    /// The most children to return. Must be at least one. `None` reads the whole directory.
    pub limit: Option<usize>,
}

/// One page of a directory listing.
#[derive(Debug)]
pub struct DirListing {
    /// The children found, at most [`ReadDirOptions::limit`] of them.
    pub values: Vec<DirEntry>,
    /// The token that resumes after this page, or `None` when the directory is exhausted.
    ///
    /// A page can hold fewer children than the limit asked for and still be followed by more,
    /// so this is the only thing that ends a walk.
    pub next_token: Option<String>,
}

/// One page as a backend reports it.
#[derive(Debug)]
pub struct DirPage {
    /// The children found, in the order the backend listed them.
    pub entries: Vec<DirEntry>,
    /// Where the next page resumes, or `None` when there is nothing after this one.
    pub next: Option<DirCursor>,
}

/// A backend that can list one directory level a page at a time.
///
/// Kept separate from [`ObjectStore::inner`] because paginated listing is not part of the
/// `ObjectStore` trait and so cannot be reached through a `dyn ObjectStore`.
///
/// This is where a backend joins the fast path: implement it, hand one to
/// [`ObjectStore::paginated_lister`], and pages are pushed down instead of listed in full.
///
/// A backend mints [`DirCursor::backend`] and resumes from its own token, which is exact and
/// needs no key order to be correct. A cursor minted by the full-listing fallback is not one
/// of its own: a walk cannot start on one path and finish on the other, and
/// [`DirCursor::expect_backend`] says so rather than guessing at a position.
#[async_trait::async_trait]
pub trait PaginatedDirLister: std::fmt::Debug + Send + Sync + 'static {
    /// One page of the immediate children of `prefix`, resuming after `resume`.
    ///
    /// `prefix` carries a trailing delimiter, and is `None` at the root of the store. `limit`
    /// is the most entries the page may hold; a backend may return fewer and still have more
    /// to give, which is what [`DirPage::next`] reports.
    ///
    /// Two things the caller relies on. Every entry in the page is strictly after `resume`,
    /// so nothing is handed over twice. And a page with a `next` has moved: the caller asks
    /// again from it, so a cursor that stood still would never finish — a page holding no
    /// entries, which is what a directory marker or a run of collapsed keys can cost, still
    /// has to leave the listing somewhere new to resume from.
    async fn list_page(
        &self,
        prefix: Option<&str>,
        resume: Option<&DirCursor>,
        limit: Option<usize>,
    ) -> Result<DirPage>;
}

impl ObjectStore {
    /// One page of the immediate children of `dir`, one directory level deep.
    ///
    /// On backends with a paginated list API — S3, GCS and Azure — the resume position and the
    /// page size are pushed into the list request, so the page costs what the page holds.
    /// Elsewhere the directory is listed in full and paged locally, which is correct but no
    /// cheaper than [`Self::read_dir`].
    ///
    /// A page can hold fewer children than `limit` asked for and still be followed by more:
    /// with a delimiter a backend spends its page budget on keys it collapses away. Walk until
    /// [`DirListing::next_token`] is `None` rather than until a page comes back short.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, ReadDirOptions};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<Vec<String>> {
    /// let mut tables = Vec::new();
    /// let mut page_token = None;
    /// loop {
    ///     let page = store
    ///         .read_dir_page("my_db", ReadDirOptions { page_token, limit: Some(10) })
    ///         .await?;
    ///     tables.extend(
    ///         page.values
    ///             .iter()
    ///             .filter_map(|entry| entry.name.strip_suffix(".lance"))
    ///             .map(String::from),
    ///     );
    ///     page_token = page.next_token;
    ///     if page_token.is_none() || tables.len() >= 10 {
    ///         break;
    ///     }
    /// }
    /// # Ok(tables)
    /// # }
    /// ```
    pub async fn read_dir_page(
        &self,
        dir: impl Into<Path>,
        options: ReadDirOptions,
    ) -> Result<DirListing> {
        let dir = dir.into();
        // A page of nothing cannot advance a listing, and the two paths below would disagree
        // about what it means: the pushdown path would report an empty directory while the
        // full listing ignored the limit and returned everything.
        if options.limit == Some(0) {
            return Err(Error::invalid_input(
                "read_dir_page limit must be at least 1, got 0",
            ));
        }
        let resume = options
            .page_token
            .as_deref()
            .map(DirCursor::decode)
            .transpose()?;

        match &self.paginated_lister {
            Some(lister) => {
                let lister = InstrumentedDirLister {
                    inner: lister.clone(),
                    io_tracker: self.io_tracker.clone(),
                    dir: dir.clone(),
                    #[cfg(feature = "metrics")]
                    store_prefix: self.store_prefix.clone(),
                };
                fill_page(&lister, &dir, resume, options.limit).await
            }
            // Goes through `inner`, so the wrappers around it instrument the request. The
            // pushdown path talks to the backend directly and instruments itself.
            None => full_listing_page(self.inner.as_ref(), &dir, resume, options.limit).await,
        }
    }
}

/// The prefix to list under, carrying the trailing delimiter that the paginated API expects.
/// `None` for the root of the store, which has no prefix at all.
fn list_prefix(dir: &Path) -> Option<String> {
    let dir = dir.as_ref();
    (!dir.is_empty()).then(|| format!("{dir}{DELIMITER}"))
}

/// Ask the backend for pages until the limit is met or the directory runs out.
///
/// More than one request only where a page came back holding less than it could: a backend
/// spends its page budget on keys, and a directory marker or a common prefix collapsing many
/// keys can leave a page short of entries — or empty — while the listing is far from over.
async fn fill_page(
    lister: &dyn PaginatedDirLister,
    dir: &Path,
    mut resume: Option<DirCursor>,
    limit: Option<usize>,
) -> Result<DirListing> {
    let prefix = list_prefix(dir);
    let mut values: Vec<DirEntry> = Vec::new();
    loop {
        // `None` once the page is full, which also covers a backend that overshot the limit.
        let remaining = match limit {
            Some(limit) => limit.checked_sub(values.len()).filter(|left| *left > 0),
            None => Some(usize::MAX),
        };
        let Some(remaining) = remaining else {
            break;
        };
        let page = lister
            .list_page(prefix.as_deref(), resume.as_ref(), limit.map(|_| remaining))
            .await?;
        values.extend(page.entries);
        match page.next {
            // A page that hands back the position it was given has not moved, and asking
            // again from it would repeat forever. The contract forbids it; a backend that
            // does it anyway fails the listing rather than hanging the caller.
            Some(next) if resume.as_ref() == Some(&next) => {
                return Err(Error::io(format!(
                    "listing '{dir}' did not advance past the position it resumed from"
                )));
            }
            Some(next) => resume = Some(next),
            // The backend says there is nothing more, so there is nothing more, however short
            // the page came back.
            None => {
                return Ok(DirListing {
                    values,
                    next_token: None,
                });
            }
        }
    }
    Ok(DirListing {
        values,
        next_token: resume.map(DirCursor::encode),
    })
}

/// One page of a directory on a store with no paginated list API: list the level in full and
/// page it locally.
///
/// The page has to be the smallest `limit` children past the cursor rather than any `limit` of
/// them, since the next call lists the same directory again and keeps only what sorts after
/// the key this page hands back. That means putting the listing in key order, which
/// `list_with_delimiter` does not promise — a sort over children already in memory, costing no
/// extra request.
async fn full_listing_page(
    store: &dyn OSObjectStore,
    dir: &Path,
    resume: Option<DirCursor>,
    limit: Option<usize>,
) -> Result<DirListing> {
    let listed = store.list_with_delimiter(Some(dir)).await?;
    let mut children = keyed_entries(&listed, list_prefix(dir).as_deref());
    if let Some(resume) = &resume {
        let key = resume.expect_key()?;
        children.retain(|child| child.key.as_str() > key);
    }
    let taken = limit.unwrap_or(children.len()).min(children.len());
    let next_token =
        (taken < children.len()).then(|| DirCursor::key(&children[taken - 1].key).encode());
    children.truncate(taken);
    Ok(DirListing {
        values: children.into_iter().map(|child| child.entry).collect(),
        next_token,
    })
}

/// A child of the directory being listed, with the key the backend listed it under.
pub struct KeyedEntry {
    /// The key relative to the directory, which is what a [`DirCursor::key`] cursor names. A
    /// child directory keeps its trailing delimiter, since that is the prefix its keys share
    /// and so where it sits in the listing; a child file is its name.
    pub key: String,
    pub entry: DirEntry,
}

/// The children in `listed`, in key order, each with the key it was listed under relative to
/// `prefix`.
///
/// Stores report common prefixes and objects as two separate lists, so the two are put back
/// into one order here. Anything that is not a child of this level is dropped, which covers
/// the marker object some stores keep for a directory: it lists as an object whose key is the
/// directory's own prefix.
pub fn keyed_entries(listed: &ListResult, prefix: Option<&str>) -> Vec<KeyedEntry> {
    let listed = listed
        .common_prefixes
        .iter()
        .map(|location| (location, None))
        .chain(
            listed
                .objects
                .iter()
                .map(|object| (&object.location, Some(object))),
        );
    let mut children: Vec<KeyedEntry> = listed
        .filter_map(|(location, meta)| {
            let key = listed_key(prefix, location, meta.is_none())?;
            let entry = DirEntry {
                name: location.filename()?.to_string(),
                kind: match meta {
                    Some(meta) => DirEntryKind::File(meta.clone()),
                    None => DirEntryKind::Directory,
                },
            };
            Some(KeyedEntry { key, entry })
        })
        .collect();
    children.sort_unstable_by(|left, right| left.key.cmp(&right.key));
    children
}

/// Where a listed location sits in the directory being listed, which is the space cursors live
/// in, or `None` if it is not a child of that directory at all.
fn listed_key(prefix: Option<&str>, location: &Path, is_dir: bool) -> Option<String> {
    let location = location.as_ref();
    let relative = match prefix {
        // Both halves of the prefix, so a location that merely starts with the directory's
        // name — `dbx/y` against `db/` — is reported as not being under it, and so is the
        // directory's own marker, whose location is `db`.
        Some(prefix) => location.strip_prefix(prefix)?,
        None => location,
    };
    (!relative.is_empty()).then(|| match is_dir {
        true => format!("{relative}{DELIMITER}"),
        false => relative.to_string(),
    })
}

/// Records what a pushed-down listing costs.
///
/// The pushdown path holds the backend directly, so its requests never pass through the
/// wrappers around [`ObjectStore::inner`] that would otherwise record them. The full-listing
/// fallback does go through those wrappers and is not wrapped in this.
#[derive(Debug)]
struct InstrumentedDirLister {
    inner: Arc<dyn PaginatedDirLister>,
    io_tracker: IOTracker,
    dir: Path,
    #[cfg(feature = "metrics")]
    store_prefix: String,
}

#[async_trait::async_trait]
impl PaginatedDirLister for InstrumentedDirLister {
    #[instrument(level = "debug", skip_all, fields(prefix = prefix))]
    async fn list_page(
        &self,
        prefix: Option<&str>,
        resume: Option<&DirCursor>,
        limit: Option<usize>,
    ) -> Result<DirPage> {
        self.io_tracker
            .record_read(LIST_OP, self.dir.clone(), 0, None);
        #[cfg(feature = "metrics")]
        let _in_flight = InFlightGuard::new(&self.store_prefix, LIST_OP);
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        let page = self.inner.list_page(prefix, resume, limit).await;

        #[cfg(feature = "metrics")]
        record_outcome(&self.store_prefix, LIST_OP, start, 0, page.is_err());
        page
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::native::NativeDirLister;
    use super::store_model::{BudgetMode, KeyOrder, Resume as ModelResume, StoreModel};
    use super::*;
    use crate::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
    use crate::object_store::{ObjectStoreParams, ObjectStoreRegistry};
    use chrono::Utc;
    use object_store::list::{PaginatedListOptions, PaginatedListStore};
    use object_store::memory::InMemory;
    use object_store::{ListResult, ObjectStoreExt, PutPayload};
    use rstest::rstest;

    /// How the store under test resolves a listing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Backend {
        /// No paginated API: list the whole directory and page it locally.
        FullListing,
        /// A paginated API over a store that lists in key order, like S3, GCS, or Azure over a
        /// flat namespace.
        KeyOrdered,
        /// A paginated API over a store that lists in no particular order, like S3 Express or
        /// an Azure account with a hierarchical namespace.
        Unordered,
    }
    use Backend::{FullListing, KeyOrdered, Unordered};

    /// One list request, as the backend saw it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ListRequest {
        resume: Option<DirCursor>,
        limit: Option<usize>,
    }

    /// A stand-in for a backend with a paginated list API.
    ///
    /// The store's behaviour comes from [`StoreModel`], which the wire-level emulator in
    /// [`conformance`](super::conformance) shares, so the two cannot drift. `limit` is spent on
    /// the keys the backend scans rather than on the entries it returns, which is the harsher
    /// of the two things a real store does with a delimiter: a common prefix collapses many
    /// keys into one entry but still costs what it collapsed. A page can therefore come back
    /// short and still be truncated, so [`DirPage::next`] is the only way to know whether a
    /// listing is finished.
    #[derive(Debug)]
    struct FakeDirLister {
        model: StoreModel,
        requests: Arc<Mutex<Vec<ListRequest>>>,
    }

    #[async_trait::async_trait]
    impl PaginatedDirLister for FakeDirLister {
        async fn list_page(
            &self,
            prefix: Option<&str>,
            resume: Option<&DirCursor>,
            limit: Option<usize>,
        ) -> Result<DirPage> {
            self.requests.lock().unwrap().push(ListRequest {
                resume: resume.cloned(),
                limit,
            });
            let prefix = prefix.unwrap_or("");
            let resume = match resume {
                Some(cursor) => ModelResume::Token(cursor.expect_backend()?),
                None => ModelResume::Start,
            };
            let page = self.model.list_level(prefix, resume, limit);

            // Keys are reported as `Path::parse`, which is how `object_store`'s own S3, GCS
            // and Azure clients report them: verbatim, so that the entries a caller sees
            // sort the way the backend sorted them.
            let listed = ListResult {
                common_prefixes: page
                    .prefixes
                    .iter()
                    .map(|prefix| Path::parse(prefix).unwrap())
                    .collect(),
                objects: page
                    .objects
                    .iter()
                    .map(|key| ObjectMeta {
                        location: Path::parse(key).unwrap(),
                        last_modified: Utc::now(),
                        size: 1,
                        e_tag: None,
                        version: None,
                    })
                    .collect(),
            };
            let entries = keyed_entries(&listed, Some(prefix).filter(|p| !p.is_empty()));
            Ok(DirPage {
                entries: entries.into_iter().map(|child| child.entry).collect(),
                next: page.next_token.map(DirCursor::backend),
            })
        }
    }

    struct TestStore {
        store: ObjectStore,
        requests: Arc<Mutex<Vec<ListRequest>>>,
    }

    impl TestStore {
        /// Every child of `dir`, taken a page at a time, which is how a caller walks a
        /// directory: the token ends the walk, never a short page.
        async fn walk(&self, dir: &str, limit: Option<usize>) -> Result<Vec<String>> {
            let mut names = Vec::new();
            let mut page_token = None;
            loop {
                let page = self
                    .store
                    .read_dir_page(Path::from(dir), ReadDirOptions { page_token, limit })
                    .await?;
                names.extend(page.values.into_iter().map(|entry| entry.name));
                page_token = page.next_token;
                if page_token.is_none() {
                    return Ok(names);
                }
            }
        }

        async fn names(&self, dir: &str, limit: Option<usize>) -> Vec<String> {
            self.walk(dir, limit).await.unwrap()
        }

        /// The first page only, as a caller wanting a bounded number of children would take it.
        async fn first_page(&self, dir: &str, limit: Option<usize>) -> DirListing {
            self.store
                .read_dir_page(
                    Path::from(dir),
                    ReadDirOptions {
                        page_token: None,
                        limit,
                    },
                )
                .await
                .unwrap()
        }
    }

    async fn test_store(backend: Backend, keys: &[&str]) -> TestStore {
        let inner = Arc::new(InMemory::new());
        for key in keys {
            // `Path::parse`, so that a key holding a character `Path::from` would encode is
            // stored under the name it was given.
            inner
                .put(&Path::parse(key).unwrap(), PutPayload::from_static(b"x"))
                .await
                .unwrap();
        }
        #[allow(deprecated)]
        let params = ObjectStoreParams {
            object_store: Some((inner, url::Url::parse("memory:///").unwrap())),
            // The deprecated hand-built path assumes nothing about the store it was given.
            list_is_lexically_ordered: Some(backend != Unordered),
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
        if backend != FullListing {
            let order = match backend {
                Unordered => KeyOrder::Reversed,
                _ => KeyOrder::ByKey,
            };
            store.paginated_lister = Some(Arc::new(FakeDirLister {
                model: StoreModel::new(keys.to_vec())
                    .with_order(order)
                    .with_budget(BudgetMode::PerScannedKey),
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

    /// Walking a directory hands back every child exactly once, however the store resolves the
    /// listing and however small the pages are. That it holds whatever order the store lists
    /// in is [`test_an_unordered_store_is_still_paged`].
    #[rstest]
    #[case::whole_directory(TABLES, "db", vec!["a.lance", "b.lance", "c.lance", "loose.txt"])]
    #[case::empty_directory(TABLES, "nonexistent", vec![])]
    #[case::only_files(&["db/a.txt", "db/b.txt", "db/c.txt"], "db", vec!["a.txt", "b.txt", "c.txt"])]
    // A directory and the sibling that follows it: `foo/` and `foo0` are adjacent in key order
    // with nothing between them, so resuming past `foo`'s contents must not swallow `foo0`.
    #[case::the_sibling_after_a_directory(&["db/foo/inside", "db/foo0"], "db", vec!["foo", "foo0"])]
    // Siblings where one name is a prefix of another, which is where a page boundary is easiest
    // to get wrong: `foo/` and `foo-bar/` differ at `/` against `-`.
    #[case::a_prefix_shaped_sibling(&["db/foo/inside", "db/foo-bar/inside", "db/zzz.txt"], "db", vec!["foo", "foo-bar", "zzz.txt"])]
    // A store that keeps a marker object for a directory reports the directory itself when
    // that directory is listed. Dropping the marker must not also drop the progress the page
    // made, or a page holding nothing but the marker reads as the end of the listing.
    #[case::a_directory_marker(&["db/marked/", "db/marked/a.txt", "db/marked/b.txt"], "db/marked", vec!["a.txt", "b.txt"])]
    // A name holding a character `Path::from` would percent-encode is still reported, and
    // sorted, under the name it was stored with.
    #[case::an_encodable_name(&["db/az", "db/a~"], "db", vec!["az", "a~"])]
    #[tokio::test]
    async fn test_walking_a_directory_is_complete(
        #[values(FullListing, KeyOrdered)] backend: Backend,
        #[values(None, Some(1), Some(2), Some(3))] limit: Option<usize>,
        #[case] keys: &[&str],
        #[case] dir: &str,
        #[case] expected: Vec<&str>,
    ) {
        let store = test_store(backend, keys).await;

        // The order children come back in is the store's, so the walk is checked for holding
        // every child once rather than for holding them in one particular order.
        let mut listed = store.names(dir, limit).await;
        let seen = listed.clone();
        listed.sort();
        assert_eq!(listed, expected, "from {seen:?}");
    }

    /// A store that lists in no particular order — S3 Express — is still paged, because a
    /// continuation token is never compared to anything. This is the case a cursor spelled as
    /// a key could not serve.
    #[tokio::test]
    async fn test_an_unordered_store_is_still_paged() {
        let store = test_store(Unordered, TABLES).await;

        let mut names = store.names("db", Some(1)).await;
        names.sort();

        assert_eq!(names, vec!["a.lance", "b.lance", "c.lance", "loose.txt"]);
        assert!(
            !store.requests.lock().unwrap().is_empty(),
            "the paginated lister should have been used"
        );
    }

    /// The point of the pushdown: a caller that wants one child of a directory holding more
    /// makes one request, for one child.
    #[tokio::test]
    async fn test_a_bounded_page_makes_one_request() {
        let store = test_store(KeyOrdered, TABLES).await;

        let page = store.first_page("db", Some(1)).await;

        assert_eq!(page.values.len(), 1);
        assert!(page.next_token.is_some());
        assert_eq!(
            *store.requests.lock().unwrap(),
            vec![ListRequest {
                resume: None,
                limit: Some(1)
            }]
        );
    }

    /// A backend page that comes back short is asked again rather than handed on. With a
    /// delimiter the backend spends its limit on keys it collapses away, so a page can hold one
    /// entry, be nowhere near the limit, and still be followed by more; a caller that stopped
    /// there would see a page far smaller than the one it asked for.
    #[tokio::test]
    async fn test_a_short_backend_page_is_filled_from_the_next() {
        // `big.lance` holds more keys than a page is allowed to scan, so the page that reports
        // it comes back with a single entry and more to come.
        let store = test_store(
            KeyOrdered,
            &[
                "db/big.lance/data/1.lance",
                "db/big.lance/data/2.lance",
                "db/big.lance/data/3.lance",
                "db/big.lance/data/4.lance",
                "db/zzz.txt",
            ],
        )
        .await;

        let page = store.first_page("db", Some(3)).await;

        assert_eq!(
            page.values
                .iter()
                .map(|entry| &entry.name)
                .collect::<Vec<_>>(),
            vec!["big.lance", "zzz.txt"]
        );
        assert_eq!(
            store.requests.lock().unwrap().len(),
            2,
            "the short first page should have been followed up"
        );
    }

    /// A page whose whole budget went on things that are not children still has to move. The
    /// directory marker is the cheap way to arrange that: on a store that lists in key order it
    /// sorts first, so with a page of one it arrives on its own, leaving nothing to hand back.
    #[tokio::test]
    async fn test_a_page_of_no_entries_keeps_going() {
        let store = test_store(KeyOrdered, &["db/marked/", "db/marked/a.txt"]).await;

        let page = store.first_page("db/marked", Some(1)).await;

        assert_eq!(
            page.values.iter().map(|e| &e.name).collect::<Vec<_>>(),
            vec!["a.txt"]
        );
        assert!(
            store.requests.lock().unwrap().len() > 1,
            "the marker page should have been followed by another"
        );
    }

    /// A page of nothing is rejected rather than left to mean whatever the backend makes of it:
    /// pushing it down reports an empty directory, and a full listing ignores it. The rejection
    /// comes before the store is consulted, so one backend covers it.
    #[tokio::test]
    async fn test_zero_limit_is_rejected() {
        let store = test_store(KeyOrdered, TABLES).await;

        let err = store.walk("db", Some(0)).await.unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(err.to_string().contains("limit must be at least 1"));
        assert!(store.requests.lock().unwrap().is_empty());
    }

    /// A token is opaque, so a string that did not come from a listing is rejected rather than
    /// read as a key and silently skipping part of the directory.
    #[rstest]
    #[case::not_a_token("a.lance")]
    #[case::wrong_version("9ka.lance")]
    #[case::unknown_kind("1za.lance")]
    #[case::empty("")]
    #[tokio::test]
    async fn test_a_token_from_nowhere_is_rejected(#[case] page_token: &str) {
        let store = test_store(KeyOrdered, TABLES).await;

        let err = store
            .store
            .read_dir_page(
                Path::from("db"),
                ReadDirOptions {
                    page_token: Some(page_token.to_string()),
                    limit: None,
                },
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    }

    /// A token round-trips through its string form unchanged, including one holding the
    /// characters the encoding itself uses.
    #[rstest]
    #[case(DirCursor::backend("1kabc"))]
    #[case(DirCursor::key("a.lance/"))]
    #[case(DirCursor::key(""))]
    fn test_a_token_round_trips(#[case] cursor: DirCursor) {
        let token = cursor.clone().encode();
        assert_eq!(DirCursor::decode(&token).unwrap(), cursor);
    }

    #[rstest]
    #[tokio::test]
    async fn test_entry_kinds(#[values(FullListing, KeyOrdered)] backend: Backend) {
        let store = test_store(backend, TABLES).await;

        let page = store.first_page("db", None).await;
        let directory = page.values.iter().find(|e| e.name == "a.lance").unwrap();
        let file = page.values.iter().find(|e| e.name == "loose.txt").unwrap();

        assert!(directory.is_dir());
        assert!(!file.is_dir());
        let DirEntryKind::File(meta) = &file.kind else {
            panic!("expected a file entry, got {:?}", file.kind);
        };
        assert_eq!(meta.size, 1);
    }

    /// The pushdown path holds the backend directly, so it has to record its own IO. A listing
    /// invisible to `io_tracker` would also be invisible to the metrics and tracing layers that
    /// sit in the same chain. Every request a page cost is recorded, not just its first.
    #[rstest]
    #[tokio::test]
    async fn test_listing_is_recorded_in_io_stats(
        #[values(FullListing, KeyOrdered)] backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        assert_eq!(store.store.io_tracker().stats().read_iops, 0);

        let _ = store.first_page("db", Some(2)).await;

        // The full listing reaches the store through its wrappers, which record it there.
        let requests = store.requests.lock().unwrap().len().max(1) as u64;
        assert_eq!(store.store.io_tracker().stats().read_iops, requests);
    }

    /// A backend stuck on one page: it reports the same children every time and always claims
    /// there are more. The contract says a page with a `next` has moved, so a backend that
    /// breaks it must not be able to spin the caller forever.
    #[derive(Debug)]
    struct StuckLister {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl PaginatedDirLister for StuckLister {
        async fn list_page(
            &self,
            _prefix: Option<&str>,
            _resume: Option<&DirCursor>,
            _limit: Option<usize>,
        ) -> Result<DirPage> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(DirPage {
                entries: Vec::new(),
                next: Some(DirCursor::backend("stuck")),
            })
        }
    }

    /// Such a backend fails the listing instead of spinning on it. A page that hands back the
    /// position it was given is asking to be requested forever, and a repeat of ready futures
    /// never yields to the runtime, so it would hang rather than time out.
    #[tokio::test]
    async fn test_a_backend_that_never_moves_fails_the_listing() {
        let lister = Arc::new(StuckLister {
            calls: AtomicUsize::new(0),
        });
        let mut store = test_store(KeyOrdered, TABLES).await;
        store.store.paginated_lister = Some(lister.clone());

        let err = store.walk("db", Some(1)).await.unwrap_err();

        assert!(err.to_string().contains("did not advance"), "{err:?}");
        assert_eq!(lister.calls.load(Ordering::SeqCst), 2);
    }

    /// A recording [`PaginatedListStore`], so the translation `NativeDirLister` performs can be
    /// checked without a cloud backend. This is the adapter the native S3, GCS and Azure stores
    /// actually use.
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

    fn recording_store(next_page_token: Option<&str>) -> Arc<RecordingListStore> {
        Arc::new(RecordingListStore {
            calls: Mutex::new(Vec::new()),
            next_page_token: next_page_token.map(String::from),
        })
    }

    /// The native lister resumes from the store's own continuation token, and reads truncation
    /// from the token it gets back rather than from how full the page is.
    #[rstest]
    #[case::truncated(Some("token"), true)]
    #[case::last_page(None, false)]
    #[tokio::test]
    async fn test_native_lister_pushes_the_token_and_limit_down(
        #[case] next_page_token: Option<&str>,
        #[case] has_more: bool,
    ) {
        let store = recording_store(next_page_token);
        let lister = NativeDirLister::for_store(store.clone());

        let page = lister
            .list_page(Some("db/"), Some(&DirCursor::backend("prev")), Some(7))
            .await
            .unwrap();

        let calls = store.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (prefix, opts) = &calls[0];
        assert_eq!(prefix.as_deref(), Some("db/"));
        assert_eq!(opts.page_token.as_deref(), Some("prev"));
        // A continuation token is a position of its own, so no offset goes with it.
        assert_eq!(opts.offset, None);
        // Without a delimiter the listing would be recursive rather than one level deep.
        assert_eq!(opts.delimiter.as_deref(), Some("/"));
        assert_eq!(opts.max_keys, Some(7));
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.next.is_some(), has_more);
    }

    /// A key cursor is what the full-listing fallback mints, and a walk cannot cross from one
    /// path to the other. Resuming from it would mean sending the store an offset, which every
    /// store interprets its own way, so the listing fails instead of guessing at a position.
    #[tokio::test]
    async fn test_native_lister_rejects_a_key_cursor() {
        let store = recording_store(None);
        let lister = NativeDirLister::for_store(store.clone());

        let err = lister
            .list_page(Some("db/"), Some(&DirCursor::key("a.lance/")), Some(7))
            .await
            .expect_err("a key cursor is not this lister's to resume from");

        assert!(
            err.to_string().contains("lists a directory in full"),
            "unexpected error: {err}"
        );
        assert!(
            store.calls.lock().unwrap().is_empty(),
            "the listing has to fail before it reaches the store"
        );
    }

    /// Throttling only adds waiting, so a throttled store — S3, GCS and Azure all reach their
    /// lister through [`with_throttling`](crate::object_store::throttle::with_throttling) —
    /// must page exactly as an unthrottled one does.
    #[rstest]
    #[tokio::test]
    async fn test_throttling_does_not_change_the_listing(#[values(false, true)] throttled: bool) {
        let mut store = test_store(KeyOrdered, TABLES).await;
        let lister = Arc::new(FakeDirLister {
            model: StoreModel::new(TABLES.to_vec()).with_budget(BudgetMode::PerScannedKey),
            requests: store.requests.clone(),
        }) as Arc<dyn PaginatedDirLister>;
        store.store.paginated_lister = Some(match throttled {
            true => AimdThrottledStore::new(
                Arc::new(InMemory::new()) as Arc<dyn OSObjectStore>,
                AimdThrottleConfig::default(),
            )
            .unwrap()
            .wrap_paginated(lister),
            false => lister,
        });

        assert_eq!(
            store.names("db", Some(1)).await,
            vec!["a.lance", "b.lance", "c.lance", "loose.txt"]
        );
    }
}
