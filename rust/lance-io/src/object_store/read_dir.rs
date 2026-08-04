// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Paginated listing of a single directory level.
//!
//! [`ObjectStore::read_dir_stream`] yields the immediate children of a prefix and, where
//! the backend supports it, pushes the resume position and the page size down into the
//! list request. A caller that needs the first few children then pays for the first few
//! children rather than for the whole prefix.
//!
//! This module holds the listing itself: the cursors it resumes from, the paging state
//! machine, and the [`PaginatedDirLister`] contract a backend implements. The backends live
//! beside it, one per way of asking — [`native`] over `object_store`'s own paginated API and
//! [`opendal`] over OpenDAL's `Lister`.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::Arc;

use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
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
#[cfg(any(
    feature = "aws",
    feature = "gcp",
    feature = "azure",
    feature = "oss",
    feature = "tencent",
    feature = "huggingface",
    feature = "tos",
    feature = "goosefs",
))]
pub mod opendal;
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

/// How many times in a row a page that got nowhere is asked for again, twice as big, before
/// the listing stops paging and takes the rest in one full listing.
///
/// A couple of doublings clear the ordinary reasons a page gets nowhere — it was spent on a
/// directory marker, or on a handful of keys behind one common prefix. Past that the prefix is
/// large enough that the backend's own page bound is in the way, and no size the listing asks
/// for will grow the page: one full listing beats many requests that come back the same.
const MAX_WIDENINGS: usize = 2;

/// A position within a directory listing, used to resume where a previous listing stopped.
///
/// A cursor is only meaningful for the directory it came from. Prefer [`DirEntry::cursor`]
/// over building one by hand.
///
/// A cursor is the storage key the entry it points at starts from, relative to the directory,
/// and a listing resumes strictly after it. That is deliberately not a plain child name: a
/// child directory `foo` starts at `foo/`, the prefix its keys share, while a child file `foo`
/// starts at `foo`. A child name never contains the delimiter, so no two children can share
/// a cursor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DirCursor(String);

impl DirCursor {
    /// Resume immediately after the child directory `name`, skipping everything inside it.
    ///
    /// `name` is a single child name, as [`DirEntry::name`] reports it; a trailing delimiter
    /// is accepted and ignored, since that is how listings and URLs often spell a directory.
    ///
    /// The position is the directory's own prefix, so a backend resuming from it hands back
    /// the directory one more time and the listing drops it. Naming a position past the
    /// prefix instead would take a key that sorts after everything under `name/` and before
    /// a sibling file named, say, `name0`, and no such key exists.
    pub fn after_directory(name: impl AsRef<str>) -> Self {
        let name = name
            .as_ref()
            .strip_suffix(DELIMITER)
            .unwrap_or(name.as_ref());
        Self(format!("{name}{DELIMITER}"))
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
    /// The children found, located the way the store they came from addresses them, so that a
    /// caller can read what was listed. That is not always the spelling the backend ordered
    /// the listing by: see [`PaginatedDirLister::backend_key`].
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
    /// `prefix` carries a trailing delimiter. `start_after` is a position within it, relative
    /// to `prefix` and in the backend's own key spelling — the one [`Self::backend_key`]
    /// produces — so a child directory ends in a delimiter too; the implementation joins the
    /// two. Joining is left to the backend because the two halves need not be spelled the same
    /// way: `prefix` comes from an `object_store` [`Path`], while a cursor is a backend key.
    ///
    /// `limit` caps the number of children in the page; it is a request, and a backend may
    /// return fewer and still set [`DirPage::has_more`].
    ///
    /// A backend that cannot resume from `start_after` must return the whole directory
    /// with `has_more` false, so that the caller can apply the cursor itself. One that
    /// reports children before `start_after` anyway is not asked again with a bigger page;
    /// the listing falls back to a full one.
    async fn list_page(
        &self,
        prefix: Option<&str>,
        start_after: Option<&str>,
        limit: Option<usize>,
    ) -> Result<DirPage>;

    /// The key the backend orders `path` by, given a path spelled the way the store this
    /// lister sits beside addresses it.
    ///
    /// The two are the same for most stores, so this defaults to the identity. They come apart
    /// when the store rewrites keys: `object_store_opendal` percent-encodes a service's keys on
    /// the way out and decodes them on the way back in, so its paths are encoded keys. A
    /// listing that pages by key has to compare in the backend's spelling or it drops children
    /// at page boundaries, since the two orders disagree — raw `az` sorts before `a~`, but
    /// encoded `a%7E` sorts before `az`.
    fn backend_key<'a>(&self, path: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(path)
    }
}

impl ObjectStore {
    /// Stream the immediate children of `dir`, one directory level deep.
    ///
    /// Entries come back in the order the store lists them, which is by storage key rather
    /// than by name, and the stream paginates internally, so a caller that filters entries
    /// and stops early only pays for the pages it consumed.
    ///
    /// On backends with a paginated list API that list in key order — S3, GCS and Azure, plus
    /// OpenDAL-backed stores whose service can resume from a key — the resume position and page
    /// size are pushed into the list request. Elsewhere the directory is listed in full and the
    /// cursor is applied locally, which is correct but no cheaper than [`Self::read_dir`].
    ///
    /// The listing is complete either way. A directory that turns out not to be pageable — a
    /// store that ignores the resume position, or a page the backend will not grow — finishes
    /// with that same full listing rather than stopping short of the children beyond it.
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
            // A cursor is a key, so resuming from one only reaches the rest of the directory if
            // the store lists in key order. S3 Express does not, and a caller can say the same
            // of any store, so the two have to be checked together: paging an unordered listing
            // would filter away children that came back before the cursor.
            Some(lister) if self.list_is_lexically_ordered => paginated_stream(
                lister.clone(),
                self.inner.clone(),
                dir,
                options,
                self.io_tracker.clone(),
                #[cfg(feature = "metrics")]
                self.store_prefix.clone(),
            ),
            // Goes through `inner`, so the wrappers around it instrument the request. The
            // pushdown path talks to the backend directly and instruments itself.
            _ => full_listing_stream(
                self.inner.clone(),
                dir,
                options.resume_from.map(|cursor| cursor.0),
            ),
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

/// Where a reported location sits in the listing, relative to the directory being listed,
/// which is the space cursors live in. Both arguments are backend keys, so that sorting and
/// comparing by the result follows the order the backend paged by.
///
/// This is the position of anything the backend reported, including what did not become an
/// entry: a directory marker reports the directory itself, whose position is the empty key.
/// Advancing past that still has to work, or a page holding nothing but the marker looks like
/// a page holding nothing at all.
fn listed_key(dir: &str, location: &str, is_dir: bool) -> Option<String> {
    if location == dir {
        return Some(String::new());
    }
    let relative = match dir.is_empty() {
        true => location,
        // Both halves of the prefix, so a location that merely starts with the directory's
        // name — `dbx/y` against `db` — is reported as not being under it at all.
        false => location.strip_prefix(dir)?.strip_prefix(DELIMITER)?,
    };
    Some(match is_dir {
        true => format!("{relative}{DELIMITER}"),
        false => relative.to_string(),
    })
}

/// How `lister` spells `path` for ordering, or `path` itself where no lister is paging.
fn backend_key<'a>(lister: Option<&dyn PaginatedDirLister>, path: &'a str) -> Cow<'a, str> {
    match lister {
        Some(lister) => lister.backend_key(path),
        None => Cow::Borrowed(path),
    }
}

/// Collect one page of listed children into stream order, dropping anything at or before
/// the cursor.
///
/// Both steps matter. Stores return common prefixes and objects as separate lists, so the
/// two have to be merged to get a single ordered sequence. And the entry the cursor points
/// at can come back again: a cursor is the key that entry starts at, which Azure's inclusive
/// `startFrom` returns, and which for a directory is a prefix whose keys all sort after it and
/// so survive even the exclusive `start-after` that S3 and GCS take.
fn page_entries(
    children: &ListResult,
    dir: &Path,
    resume_from: Option<&str>,
    lister: Option<&dyn PaginatedDirLister>,
) -> Page {
    let listed = children
        .common_prefixes
        .iter()
        .map(|prefix| (prefix, None))
        .chain(
            children
                .objects
                .iter()
                .map(|object| (&object.location, Some(object))),
        );

    let dir_key = backend_key(lister, dir.as_ref());
    // The furthest position reported by something that did not become an entry, so that a page
    // spent entirely on a directory marker still leaves the listing somewhere to resume from.
    let mut skipped_key = None;
    let mut resume_ignored = false;
    let mut keyed: Vec<(String, DirEntry)> =
        Vec::with_capacity(children.common_prefixes.len() + children.objects.len());
    for (location, meta) in listed {
        let key = listed_key(
            &dir_key,
            &backend_key(lister, location.as_ref()),
            meta.is_none(),
        );
        let Some(key) = key else {
            continue;
        };
        resume_ignored |= resume_from.is_some_and(|cursor| key.as_str() < cursor);
        match dir_entry(dir, location, meta) {
            Some(entry) => keyed.push((key, entry)),
            None => skipped_key = skipped_key.max(Some(key)),
        }
    }
    keyed.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

    // The next page resumes from the furthest position the backend reported, whether or not it
    // became an entry, so that a page yielding no entries at all still makes progress.
    let last_key = keyed.last().map(|(key, _)| key.clone()).max(skipped_key);
    let entries = keyed
        .into_iter()
        .filter(|(key, _)| resume_from.is_none_or(|cursor| key.as_str() > cursor))
        .map(|(_, entry)| entry)
        .collect();
    Page {
        entries,
        last_key,
        resume_ignored,
    }
}

struct Page {
    entries: VecDeque<DirEntry>,
    last_key: Option<String>,
    /// Whether the backend reported a child before the position it was asked to resume from,
    /// which means it did not honour that position: a bigger page would fetch the same keys
    /// again rather than reach the ones past it.
    resume_ignored: bool,
}

struct PaginatedState {
    lister: Arc<dyn PaginatedDirLister>,
    /// The store the lister sits beside, for a directory that turns out not to be pageable.
    store: Arc<dyn OSObjectStore>,
    dir: Path,
    prefix: Option<String>,
    page_size: Option<usize>,
    /// How many more times `page_size` may double after a page that got nowhere. Refilled by
    /// every page that makes progress, so the bound is on consecutive ineffective widenings.
    widenings_left: usize,
    /// Where the next page resumes, relative to the directory being listed and in the
    /// backend's key spelling. `None` before the first request of an unresumed listing.
    cursor: Option<String>,
    /// Whether the backend may have children past the cursor. True until a page says otherwise,
    /// so that the first request always happens.
    has_more: bool,
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

        // Resuming hands back the entry the cursor points at — always for a directory, and
        // for any entry on a backend whose resume position is inclusive — and it is dropped
        // again here. Asking for one more keeps a page of the wanted size after that drop,
        // and keeps a page of size one from being nothing but the entry the previous page
        // ended on.
        let limit = match (&self.cursor, self.page_size) {
            (Some(_), Some(page_size)) => Some(page_size.saturating_add(1)),
            (_, page_size) => page_size,
        };
        let page = self
            .lister
            .list_page(self.prefix.as_deref(), self.cursor.as_deref(), limit)
            .await;

        #[cfg(feature = "metrics")]
        record_outcome(&self.store_prefix, LIST_OP, start, 0, page.is_err());
        page
    }

    /// Double the page asked for after a page that got nowhere, reporting whether a bigger one
    /// is worth asking for at all.
    ///
    /// It is not when no page size was asked for to begin with — the backend is already
    /// returning as much as it will — when the size cannot double any further, or when the last
    /// few doublings changed nothing, which means the page is not going to grow.
    fn widen_page(&mut self) -> bool {
        let Some(widened) = self.page_size.and_then(|size| size.checked_mul(2)) else {
            return false;
        };
        let Some(left) = self.widenings_left.checked_sub(1) else {
            return false;
        };
        self.widenings_left = left;
        self.page_size = Some(widened);
        true
    }

    /// Stop paging, and take everything the listing has not reached yet from one listing of
    /// the whole directory.
    ///
    /// For when paging cannot get past where it is. The cursor already covers every entry
    /// handed to the caller, so filtering the full listing by it picks up exactly the
    /// remainder, in the same order the entries before it came back in.
    async fn rest_in_full(&mut self) -> Result<VecDeque<DirEntry>> {
        self.has_more = false;
        let listed = self.store.list_with_delimiter(Some(&self.dir)).await?;
        let rest = page_entries(
            &listed,
            &self.dir,
            self.cursor.as_deref(),
            Some(&*self.lister),
        );
        Ok(rest.entries)
    }
}

fn paginated_stream(
    lister: Arc<dyn PaginatedDirLister>,
    store: Arc<dyn OSObjectStore>,
    dir: Path,
    options: ReadDirOptions,
    io_tracker: IOTracker,
    #[cfg(feature = "metrics")] store_prefix: String,
) -> BoxStream<'static, Result<DirEntry>> {
    let state = PaginatedState {
        // The caller's cursor is spelled the way the store spells a name; the backend pages by
        // its own keys, and every comparison below is in that spelling.
        cursor: options
            .resume_from
            .map(|cursor| lister.backend_key(&cursor.0).into_owned()),
        lister,
        store,
        prefix: list_prefix(&dir),
        dir,
        page_size: options.page_size,
        widenings_left: MAX_WIDENINGS,
        has_more: true,
        buffered: VecDeque::new(),
        io_tracker,
        #[cfg(feature = "metrics")]
        store_prefix,
    };

    stream::try_unfold(state, |mut state| async move {
        loop {
            if let Some(entry) = state.buffered.pop_front() {
                return Ok(Some((entry, state)));
            }
            if !state.has_more {
                return Ok(None);
            }

            let page = state.list_page().await?;
            let listed = page_entries(
                &page.children,
                &state.dir,
                state.cursor.as_deref(),
                Some(&*state.lister),
            );

            state.buffered = match listed.last_key {
                // Advance past everything the backend reported, not just what survived the
                // cursor, so a page that filters away entirely still makes progress.
                Some(last_key) if Some(&last_key) > state.cursor.as_ref() => {
                    state.cursor = Some(last_key);
                    state.has_more = page.has_more;
                    state.widenings_left = MAX_WIDENINGS;
                    listed.entries
                }
                // The page reported no position past the cursor, so it was too small to reach
                // the next child: it may have been spent entirely on a directory marker, which
                // some stores hide from a listing and others report as the directory itself, or
                // on the keys a common prefix collapsed. Widening gets past that, as far as it
                // is worth trying.
                _ if page.has_more && !listed.resume_ignored && state.widen_page() => {
                    listed.entries
                }
                // Paging is out of moves: either the page will not grow, or the backend reported
                // a child before the cursor and so never honoured the resume position, which no
                // page size fixes because every page starts from the top of the directory again.
                _ if page.has_more => state.rest_in_full().await?,
                // The backend says there is nothing more, so there is nothing more.
                _ => {
                    state.has_more = false;
                    listed.entries
                }
            };
        }
    })
    .boxed()
}

/// List a whole directory level in one go and apply `cursor` locally, in the spelling the
/// store reports keys in.
fn full_listing_stream(
    store: Arc<dyn OSObjectStore>,
    dir: Path,
    cursor: Option<String>,
) -> BoxStream<'static, Result<DirEntry>> {
    stream::once(async move {
        let listed = store.list_with_delimiter(Some(&dir)).await?;
        let entries = page_entries(&listed, &dir, cursor.as_deref(), None);
        Result::Ok(stream::iter(entries.entries.into_iter().map(Result::Ok)))
    })
    .try_flatten()
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::native::NativeDirLister;
    use super::store_model::{BudgetMode, KeyOrder, OffsetMode, Resume, StoreModel};
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
        /// No paginated API: list the whole directory and apply the cursor locally.
        FullListing,
        /// A paginated API whose offset is exclusive: `start-after`, which S3 takes and
        /// which GCS takes too, since `object_store` lists it over the S3-compatible XML API.
        ExclusiveOffset,
        /// A paginated API whose offset is inclusive, like Azure's `startFrom`.
        InclusiveOffset,
        /// A paginated API over a store that orders each directory level by child name rather
        /// than by whole key, which Azure documents for accounts with a hierarchical namespace.
        NameOrdered,
    }
    use Backend::{ExclusiveOffset, FullListing, InclusiveOffset, NameOrdered};

    /// One list request, as the backend saw it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ListRequest {
        start_after: Option<String>,
        limit: Option<usize>,
    }

    /// A stand-in for a backend with a paginated list API.
    ///
    /// The store's behaviour comes from [`StoreModel`], which the wire-level emulator in
    /// [`conformance`](super::conformance) shares, so the two cannot
    /// drift. `limit` is spent on the keys the backend scans rather than on the entries it
    /// returns, which is the harsher of the two things a real store does with a delimiter: a
    /// common prefix collapses many keys into one entry but still costs what it collapsed. A
    /// page can therefore come back short and still be truncated, so `has_more` is the only
    /// way to know whether a listing is finished.
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
            start_after: Option<&str>,
            limit: Option<usize>,
        ) -> Result<DirPage> {
            self.requests.lock().unwrap().push(ListRequest {
                start_after: start_after.map(String::from),
                limit,
            });
            let prefix = prefix.unwrap_or("");
            // The cursor is relative to the prefix, and the backend joins the two.
            let start_after = start_after.map(|cursor| format!("{prefix}{cursor}"));
            let resume = match &start_after {
                Some(start_after) => Resume::Offset(start_after),
                None => Resume::Start,
            };
            let page = self.model.list_level(prefix, resume, limit);

            // Keys are reported as `Path::parse`, which is how `object_store`'s own S3, GCS
            // and Azure clients report them: verbatim, so that the entries a caller sees
            // sort the way the backend sorted them.
            Ok(DirPage {
                children: ListResult {
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
                },
                has_more: page.truncated,
            })
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
            list_is_lexically_ordered: Some(true),
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
            let offset = match backend {
                InclusiveOffset => OffsetMode::Inclusive,
                _ => OffsetMode::Exclusive,
            };
            let order = match backend {
                NameOrdered => KeyOrder::DelimiterLowest,
                _ => KeyOrder::ByKey,
            };
            store.paginated_lister = Some(Arc::new(FakeDirLister {
                model: StoreModel::new(keys.to_vec())
                    .with_offset(offset)
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

    /// Listing one level, over the positions and fixtures that make the key ordering visible.
    /// Every case runs on all three ways a listing resolves, so a pushdown that disagrees with
    /// a plain full listing fails here.
    #[rstest]
    #[case::whole_directory(TABLES, "db", None, None, vec!["a.lance", "b.lance", "c.lance", "loose.txt"])]
    #[case::empty_directory(TABLES, "nonexistent", None, None, vec![])]
    #[case::after_a_directory(TABLES, "db", Some(DirCursor::after_directory("a.lance")), None, vec!["b.lance", "c.lance", "loose.txt"])]
    #[case::after_a_file(&["db/a.txt", "db/b.txt", "db/c.txt"], "db", Some(DirCursor::after_file("a.txt")), None, vec!["b.txt", "c.txt"])]
    // Resuming after a directory must not take a sibling with it. `foo/` and `foo0` are
    // adjacent in key order with nothing between them, so a cursor that tried to name a
    // position past `foo/`'s contents would land on `foo0` and swallow it.
    #[case::the_sibling_after_a_directory(&["db/foo/inside", "db/foo0"], "db", Some(DirCursor::after_directory("foo")), Some(1), vec!["foo0"])]
    // A page can hold entries that are not children: a store that keeps a marker object for a
    // directory reports the directory itself when that directory is listed. Dropping the marker
    // must not also drop the position it was reported at, or a page holding nothing but the
    // marker reads as a page holding nothing at all and ends the listing. The marker sorts
    // first, so with a page of one it arrives on its own.
    #[case::a_page_of_only_a_directory_marker(&["db/marked/", "db/marked/a.txt"], "db/marked", None, Some(1), vec!["a.txt"])]
    // Paging compares cursors against the keys the backend reported, so a name holding a
    // character that `Path::from` would percent-encode still pages in the backend's order.
    // Encoding it would put `a~` before `az`, and the page that resumed after `az` would then
    // drop it.
    #[case::the_backend_key_order(&["db/az", "db/a~"], "db", None, Some(1), vec!["az", "a~"])]
    #[tokio::test]
    async fn test_lists_one_level(
        #[values(FullListing, ExclusiveOffset, InclusiveOffset)] backend: Backend,
        #[case] keys: &[&str],
        #[case] dir: &str,
        #[case] resume_from: Option<DirCursor>,
        #[case] page_size: Option<usize>,
        #[case] expected: Vec<&str>,
    ) {
        let store = test_store(backend, keys).await;
        let options = ReadDirOptions {
            resume_from,
            page_size,
        };
        assert_eq!(store.names(dir, options).await, expected);
    }

    /// A directory cursor spelled with a trailing delimiter is the same position as one
    /// without, so the listing above covers both. Keeping the delimiter would resume inside
    /// the directory, which serves it up again.
    #[test]
    fn test_a_directory_cursor_ignores_a_trailing_delimiter() {
        assert_eq!(
            DirCursor::after_directory("a.lance/"),
            DirCursor::after_directory("a.lance")
        );
    }

    /// A cursor is a key, so a store that does not list in key order cannot be paged by one:
    /// a page could come back holding children that sort before the cursor, and filtering
    /// them out would lose them. Such a store is listed in full even though it can paginate.
    #[tokio::test]
    async fn test_an_unordered_store_is_listed_in_full() {
        let mut store = test_store(ExclusiveOffset, TABLES).await;
        store.store.list_is_lexically_ordered = false;
        let options = ReadDirOptions {
            page_size: Some(1),
            ..Default::default()
        };

        assert_eq!(
            store.names("db", options).await,
            vec!["a.lance", "b.lance", "c.lance", "loose.txt"]
        );
        assert!(
            store.requests.lock().unwrap().is_empty(),
            "the paginated lister must not be used"
        );
    }

    /// Walking page by page must return every child exactly once, including names that sort
    /// after the cursor by name but before it by key (`foo-bar/` sorts before `foo/`).
    #[rstest]
    #[tokio::test]
    async fn test_paging_by_cursor_is_complete(
        #[values(FullListing, ExclusiveOffset, InclusiveOffset)] backend: Backend,
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

    /// The two orders a store can list a level in only disagree over siblings where one name
    /// is a prefix of another: `foo/` and `foo-bar/` differ at `/` against `-`, so a store
    /// ordering by whole key puts `foo-bar` first and one ordering by child name puts `foo`
    /// first. A cursor is a key, and the listing has to stay complete whichever order the key
    /// it names sits in.
    #[rstest]
    #[tokio::test]
    async fn test_paging_a_prefix_shaped_sibling_is_complete(
        #[values(FullListing, ExclusiveOffset, InclusiveOffset)] backend: Backend,
        #[values(1, 2, 3)] page_size: usize,
    ) {
        let keys = ["db/foo/inside", "db/foo-bar/inside", "db/zzz.txt"];
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

        // The order the children come back in is the store's, so only what came back is
        // asserted here: every child, and none of them twice.
        let mut listed = seen.clone();
        listed.sort();
        assert_eq!(listed, vec!["foo", "foo-bar", "zzz.txt"], "from {seen:?}");
    }

    /// Why a store that orders a directory level by child name is not paged, and why the
    /// providers that can recognise one install no paginated lister for it.
    ///
    /// A cursor is a key. On such a store a key that sorts before the cursor can still be a
    /// child the listing has not reached: it lists `foo` before `foo-bar`, while by key
    /// `foo-bar/` comes first, `-` sorting before `/`. Resuming after `foo` then reads
    /// `foo-bar/` as a child already handed over, and the fallback for a backend that appears
    /// to have ignored the resume position filters by the same key, so it drops it twice.
    ///
    /// Paging such a store would need the order it actually used, which no list API reports.
    #[tokio::test]
    async fn test_paging_a_name_ordered_store_loses_a_child() {
        let keys = ["db/foo/inside", "db/foo-bar/inside", "db/zzz.txt"];
        let store = test_store(NameOrdered, &keys).await;

        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let options = ReadDirOptions {
                resume_from: cursor.take(),
                page_size: Some(1),
            };
            let page: Vec<DirEntry> = store
                .store
                .read_dir_stream(Path::from("db"), options)
                .take(1)
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
            vec!["foo", "zzz.txt"],
            "foo-bar is lost, which is what the providers keep such a store away from"
        );
    }

    /// A page of nothing is rejected rather than left to mean whatever the backend makes of
    /// it: pushing it down reports an empty directory, and a full listing ignores it. The
    /// rejection comes before the store is consulted, so one backend covers it.
    #[tokio::test]
    async fn test_zero_page_size_is_rejected() {
        let store = test_store(ExclusiveOffset, TABLES).await;
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
        #[values(FullListing, ExclusiveOffset, InclusiveOffset)] backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;
        let entries: Vec<DirEntry> = store
            .store
            .read_dir_stream(Path::from("db"), ReadDirOptions::default())
            .try_collect()
            .await
            .unwrap();

        assert!(entries[0].is_dir());
        assert_eq!(entries[0].cursor(), DirCursor::after_directory("a.lance"));

        let loose = entries.last().unwrap();
        assert!(!loose.is_dir());
        let DirEntryKind::File(meta) = &loose.kind else {
            panic!("expected a file entry, got {:?}", loose.kind);
        };
        assert_eq!(meta.size, 1);
        assert_eq!(loose.cursor(), DirCursor::after_file("loose.txt"));
    }

    /// The point of the pushdown: a caller that wants one child of a directory holding more
    /// makes one request, for one child.
    #[rstest]
    #[tokio::test]
    async fn test_short_listing_makes_one_request(
        #[values(ExclusiveOffset, InclusiveOffset)] backend: Backend,
    ) {
        let store = test_store(backend, TABLES).await;

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

        assert_eq!(first[0].name, "a.lance");
        assert_eq!(
            *store.requests.lock().unwrap(),
            vec![ListRequest {
                start_after: None,
                limit: Some(1)
            }]
        );
    }

    /// The pushdown path holds the backend directly, so it has to record its own IO. A
    /// listing that is invisible to `io_tracker` would also be invisible to the metrics
    /// and tracing layers that sit in the same chain.
    #[rstest]
    #[tokio::test]
    async fn test_listing_is_recorded_in_io_stats(
        #[values(FullListing, ExclusiveOffset, InclusiveOffset)] backend: Backend,
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
        #[values(ExclusiveOffset, InclusiveOffset)] backend: Backend,
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
        let requests = store.requests.lock().unwrap();
        // The first page held one entry against a limit of three and was still followed. The
        // second resumed at `big.lance/` and spent its whole limit on the keys that prefix
        // collapses, leaving nothing past the cursor, so the third widened the page to get
        // past it. A real store spends its limit on the entries it returns rather than on the
        // keys it scans, so it would answer in two.
        assert_eq!(
            requests.len(),
            3,
            "the short first page should not have ended the listing: {requests:?}"
        );
    }

    /// Resuming asks for one entry more than the page, so that a backend re-serving the
    /// entry the cursor points at cannot use up the whole page.
    #[rstest]
    #[tokio::test]
    async fn test_resuming_asks_for_one_more_than_the_page(
        #[values(ExclusiveOffset, InclusiveOffset)] backend: Backend,
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
        assert_eq!(requests[0].start_after.as_deref(), Some("a.lance/"));
        assert_eq!(requests[0].limit, Some(2));
    }

    /// A backend stuck on one page: it reports the same children every time and always claims
    /// there are more. Errors once it is clear the caller is looping, because a repeat of ready
    /// futures never yields to the runtime, so a regression would hang rather than time out;
    /// failing from inside the backend keeps it a test failure.
    #[derive(Debug)]
    struct StuckLister {
        children: Vec<Path>,
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
                    common_prefixes: self.children.clone(),
                    objects: Vec::new(),
                },
                has_more: true,
            })
        }
    }

    /// A page the listing cannot get past, with no page size to widen — the backend is already
    /// returning as much as it will — is finished off by listing the rest of the directory in
    /// full. The listing must not end short of children that are there.
    #[rstest]
    // Nothing at all, so there is no position to resume from and nothing to try again with.
    #[case::an_empty_page(vec![], 1)]
    // The same child every time: one page to find it, one to find it again and give up.
    #[case::the_same_child_every_time(vec![Path::from("db/a.lance")], 2)]
    #[tokio::test]
    async fn test_a_page_that_cannot_be_paged_past_lists_the_rest_in_full(
        #[case] children: Vec<Path>,
        #[case] expected_calls: usize,
    ) {
        let lister = Arc::new(StuckLister {
            children,
            calls: AtomicUsize::new(0),
        });
        let mut store = test_store(ExclusiveOffset, TABLES).await;
        store.store.paginated_lister = Some(lister.clone());

        assert_eq!(
            store.names("db", ReadDirOptions::default()).await,
            vec!["a.lance", "b.lance", "c.lance", "loose.txt"]
        );
        assert_eq!(lister.calls.load(Ordering::SeqCst), expected_calls);
    }

    /// A store that ignores the resume position — Azurite does, with `startFrom` — serves the
    /// beginning of the directory again on every page. Widening cannot help, and with no page
    /// size there is nothing to widen anyway, so the listing must not end at the backend's own
    /// page bound with children still to come.
    #[rstest]
    #[tokio::test]
    async fn test_a_store_that_ignores_the_cursor_still_lists_everything(
        #[values(None, Some(1))] page_size: Option<usize>,
    ) {
        // More children than the store will put in one page, so the listing has to get past
        // the first page to find them all.
        let keys = ["db/a", "db/b", "db/c", "db/d", "db/e", "db/f"];
        let mut store = test_store(ExclusiveOffset, &keys).await;
        store.store.paginated_lister = Some(Arc::new(FakeDirLister {
            model: StoreModel::new(keys.to_vec())
                .with_offset(OffsetMode::Ignored)
                .with_budget(BudgetMode::PerEntry)
                .with_page_bound(4),
            requests: store.requests.clone(),
        }));

        let options = ReadDirOptions {
            page_size,
            ..Default::default()
        };

        assert_eq!(
            store.names("db", options).await,
            vec!["a", "b", "c", "d", "e", "f"]
        );
        // Paging is abandoned as soon as a page reports a child before the cursor, rather than
        // doubling the page size towards a bound the backend will not honour.
        assert!(
            store.requests.lock().unwrap().len() <= 4,
            "gave up too slowly: {:?}",
            store.requests.lock().unwrap()
        );
    }

    /// A common prefix holding more keys than the backend will scan for one page stalls the
    /// listing where no page size can help: past that bound the store answers the same however
    /// much is asked for. Widening has to give out after a few tries rather than doubling towards
    /// `usize`, so one such boundary costs a handful of requests instead of dozens.
    #[tokio::test]
    async fn test_widening_gives_up_on_a_page_the_backend_will_not_grow() {
        let mut keys: Vec<String> = (0..20).map(|key| format!("db/a/data/{key:02}")).collect();
        keys.push("db/z".to_string());
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let mut store = test_store(ExclusiveOffset, &key_refs).await;
        store.store.paginated_lister = Some(Arc::new(FakeDirLister {
            // A page bound below the number of keys behind `db/a/`, spent on the keys the prefix
            // collapses, so no page the listing asks for ever reaches `db/z`.
            model: StoreModel::new(keys)
                .with_budget(BudgetMode::PerScannedKey)
                .with_page_bound(4),
            requests: store.requests.clone(),
        }));

        let options = ReadDirOptions {
            page_size: Some(1),
            ..Default::default()
        };
        assert_eq!(store.names("db", options).await, vec!["a", "z"]);

        // One page reaching `db/a/`, `MAX_WIDENINGS` that get no further, and one more that
        // gives up and hands the rest to a full listing.
        let requests = store.requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            MAX_WIDENINGS + 2,
            "widened too far: {requests:?}"
        );
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

    /// The translation the lister performs on the way down, and how it reads truncation on the
    /// way back: the native API reports it by handing back a token, not by filling the page.
    #[rstest]
    #[case::truncated(Some("token"), true)]
    #[case::last_page(None, false)]
    #[tokio::test]
    async fn test_native_lister_pushes_the_cursor_and_page_size_down(
        #[case] next_page_token: Option<&str>,
        #[case] has_more: bool,
    ) {
        let store = Arc::new(RecordingListStore {
            calls: Mutex::new(Vec::new()),
            next_page_token: next_page_token.map(String::from),
        });
        let lister = NativeDirLister::for_store(store.clone());

        let page = lister
            .list_page(Some("db/"), Some("a.lance/"), Some(7))
            .await
            .unwrap();

        let calls = store.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (prefix, opts) = &calls[0];
        assert_eq!(prefix.as_deref(), Some("db/"));
        // The cursor arrives relative to the prefix and is resolved against it here.
        assert_eq!(opts.offset.as_deref(), Some("db/a.lance/"));
        // Without a delimiter the listing would be recursive rather than one level deep.
        assert_eq!(opts.delimiter.as_deref(), Some("/"));
        assert_eq!(opts.max_keys, Some(7));
        // Resumption is by key, so the store's own continuation token is never used.
        assert_eq!(opts.page_token, None);
        assert_eq!(page.children.common_prefixes.len(), 1);
        assert_eq!(page.has_more, has_more);
    }

    /// A lister over a store that does not spell a path the way the backend spells the key it
    /// listed. `object_store_opendal` is one: it percent-encodes what the service reported and
    /// decodes it again on the way back in, so a path is an encoded key.
    #[derive(Debug)]
    struct EncodingDirLister(FakeDirLister);

    #[async_trait::async_trait]
    impl PaginatedDirLister for EncodingDirLister {
        async fn list_page(
            &self,
            prefix: Option<&str>,
            start_after: Option<&str>,
            limit: Option<usize>,
        ) -> Result<DirPage> {
            let mut page = self.0.list_page(prefix, start_after, limit).await?;
            for location in page.children.common_prefixes.iter_mut() {
                *location = Path::from(location.as_ref());
            }
            for object in page.children.objects.iter_mut() {
                object.location = Path::from(object.location.as_ref());
            }
            Ok(page)
        }

        /// The inverse of `Path::from` for the one character the keys below need it for.
        fn backend_key<'a>(&self, path: &'a str) -> Cow<'a, str> {
            Cow::Owned(path.replace("%7E", "~"))
        }
    }

    /// Such a store still has to page in the backend's order. Ordering by the paths instead
    /// loses `a~`, which the backend reports after `az` but whose path `a%7E` sorts before it.
    ///
    /// Throttling only adds waiting, so a throttled store — S3, GCS and Azure all reach one of
    /// these listers through [`with_throttling`](crate::object_store::throttle::with_throttling)
    /// — must page by the same keys as an unthrottled one.
    #[rstest]
    #[tokio::test]
    async fn test_paging_follows_the_backend_key_rather_than_the_path(
        #[values(false, true)] throttled: bool,
    ) {
        let keys = ["db/az", "db/a~"];
        let mut store = test_store(ExclusiveOffset, &keys).await;
        let lister = Arc::new(EncodingDirLister(FakeDirLister {
            model: StoreModel::new(keys.to_vec()).with_budget(BudgetMode::PerScannedKey),
            requests: store.requests.clone(),
        })) as Arc<dyn PaginatedDirLister>;
        store.store.paginated_lister = Some(match throttled {
            true => AimdThrottledStore::new(
                Arc::new(InMemory::new()) as Arc<dyn OSObjectStore>,
                AimdThrottleConfig::default(),
            )
            .unwrap()
            .wrap_paginated(lister),
            false => lister,
        });

        let options = ReadDirOptions {
            page_size: Some(1),
            ..Default::default()
        };
        assert_eq!(store.names("db", options).await, vec!["az", "a%7E"]);

        // A cursor names a position the way the store spells it, so it means the same position
        // only once translated: `a~` is the last key, and nothing follows it.
        let options = ReadDirOptions {
            resume_from: Some(DirCursor::after_file("a%7E")),
            ..Default::default()
        };
        assert!(store.names("db", options).await.is_empty());
    }
}
