// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a paginated directory listing has to get right, over the wire.
//!
//! The tests in [`read_dir`](super) drive the paging logic through a fake lister. These drive
//! the whole path instead — a real `object_store` client, real query parameters, real XML —
//! against the [`emulator`](super::emulator), which can be told to behave the ways a store is
//! allowed to behave. The same fixtures and the same invariant then run against a real bucket
//! in the `#[ignore]`d tests at the bottom.
//!
//! The invariant is a differential one: for every page size, paging has to produce exactly
//! what one unpaginated listing of the same directory produces. Anything else — a child
//! dropped at a page boundary, a child served twice, an order that disagrees — fails, and
//! nothing has to be restated per backend as an expected list of names.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures::{StreamExt, TryStreamExt};
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::list::PaginatedListStore;
use object_store::{ClientOptions, ObjectStore as OSObjectStore, path::Path};
use rstest::rstest;

use super::super::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use super::emulator::{ListEmulator, Wire};
use super::native::NativeDirLister;
use super::store_model::{BudgetMode, StoreModel};
use super::{DirEntry, PaginatedDirLister, ReadDirOptions};

/// The directory under test. Kept free of characters that need encoding: it is the caller's
/// own path, and the interesting cases are the children.
const DIR: &str = "db";

/// The children a listing has to get right. Not an exhaustive character set — the point is
/// the shapes: names adjacent in key order, names that are prefixes of one another, and
/// names carrying characters that have to survive an XML document and a query string.
const SHAPES: &[&str] = &[
    "db/a.lance/_versions/1.manifest",
    "db/a.lance/data/1.lance",
    // A directory and a file that sit next to each other in key order with nothing that can
    // sort between them: `foo/` and `foo0`.
    "db/foo/inside",
    "db/foo0",
    // A file and a directory of the same name. The file's key sorts first.
    "db/twin",
    "db/twin/inside",
    // Names that order differently by key than by name: `-` sorts before `/`, `2` after.
    "db/foo-bar/inside",
    "db/foo2/inside",
    // Depth below the level being listed, which the delimiter collapses.
    "db/deep/a/b/c/d",
    // Characters `Path::from` percent-encodes, so a listing that re-encoded its keys would
    // reorder them.
    "db/tilde~/inside",
    "db/bracket[1]",
    "db/star*",
    // Characters that have to be escaped in the response XML.
    "db/amp&ersand",
    "db/less<than",
    // Characters whose treatment in a query string is the risk: a literal `+` and a space
    // both become `+` under form encoding, so a store that decodes the resume position as a
    // form value and one that decodes it as a URI path disagree about which key was meant.
    "db/plus+one",
    "db/with space",
    // Characters that are legal in a key but reserved in a URL.
    "db/per%cent",
    "db/equals=sign",
    "db/quote'single",
    "db/paren(s)",
    // Non-ASCII, one inside the basic multilingual plane and one outside it.
    "db/héllo/inside",
    "db/emoji😀",
    "db/zzz.txt",
];

/// A key ending in the delimiter: the directory marker consoles and older SDKs leave behind.
/// `Path` normalises the trailing delimiter away, so this cannot be created through
/// `object_store` and only the emulator can serve it.
const DIRECTORY_MARKER: &str = "db/marked/";

const MARKED_CHILDREN: &[&str] = &[DIRECTORY_MARKER, "db/marked/inside"];

/// Page sizes to hold the listing to. One is the interesting case — every entry is a page
/// boundary — and the rest move the boundary onto different entries.
const PAGE_SIZES: &[usize] = &[1, 2, 3, 4, 10];

/// The client under test. All three speak to the emulator without credentials; S3 and GCS
/// share a wire format, and Azure has its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Client {
    S3,
    Gcs,
    Azure,
}

impl Client {
    fn wire(&self) -> Wire {
        match self {
            Self::S3 | Self::Gcs => Wire::S3,
            Self::Azure => Wire::Azure,
        }
    }

    /// A client pointed at `endpoint`, as both an [`OSObjectStore`] and the paginated trait
    /// the pushdown path needs. Both handles are the same store, so the paginated and the
    /// full listing see the same keys.
    fn build(&self, endpoint: &str) -> (Arc<dyn OSObjectStore>, Arc<dyn PaginatedListStore>) {
        let allow_http = ClientOptions::new().with_allow_http(true);
        match self {
            Self::S3 => {
                let store = Arc::new(
                    AmazonS3Builder::new()
                        .with_bucket_name("bucket")
                        .with_region("us-east-1")
                        .with_endpoint(endpoint)
                        .with_allow_http(true)
                        .with_skip_signature(true)
                        .build()
                        .unwrap(),
                );
                (store.clone(), store)
            }
            Self::Gcs => {
                let store = Arc::new(
                    GoogleCloudStorageBuilder::new()
                        .with_bucket_name("bucket")
                        .with_base_url(endpoint)
                        .with_client_options(allow_http)
                        .with_skip_signature(true)
                        .build()
                        .unwrap(),
                );
                (store.clone(), store)
            }
            Self::Azure => {
                let store = Arc::new(
                    MicrosoftAzureBuilder::new()
                        .with_account("account")
                        .with_container_name("bucket")
                        .with_endpoint(endpoint.to_string())
                        .with_allow_http(true)
                        .with_skip_signature(true)
                        .build()
                        .unwrap(),
                );
                (store.clone(), store)
            }
        }
    }
}

/// How an entry is spelled when listings are compared: its name, plus a delimiter if it is a
/// directory. A file and a directory can share a name — the fixture holds both a `twin` file
/// and a `twin/` directory — so a name on its own cannot tell an entry served twice from two
/// entries that happen to be called the same thing.
fn label(entry: &DirEntry) -> String {
    match entry.is_dir() {
        true => format!("{}/", entry.name),
        false => entry.name.clone(),
    }
}

/// A pair of stores over the same backend: one that pushes paging down, and one that lists
/// the directory in full and is the reference the paginated one is held to.
struct Conformance {
    paginated: ObjectStore,
    reference: ObjectStore,
}

impl Conformance {
    /// Wraps `inner` in a lance store twice, once with the paginated lister installed.
    ///
    /// The lister is installed here rather than by a provider so the test needs no
    /// credentials and no provider-specific configuration; that the providers install it is
    /// covered in [`providers`](super::super::providers).
    async fn new(
        inner: Arc<dyn OSObjectStore>,
        lister: Arc<dyn PaginatedDirLister>,
        uri: &str,
    ) -> Self {
        let build = || async {
            #[allow(deprecated)]
            let params = ObjectStoreParams {
                object_store: Some((inner.clone(), url::Url::parse(uri).unwrap())),
                // What every store served here does, and what paging by cursor requires.
                list_is_lexically_ordered: Some(true),
                ..Default::default()
            };
            let (store, _) = ObjectStore::from_uri_and_params(
                Arc::new(ObjectStoreRegistry::default()),
                uri,
                &params,
            )
            .await
            .unwrap();
            Arc::try_unwrap(store).unwrap()
        };
        let mut paginated = build().await;
        paginated.paginated_lister = Some(lister);
        Self {
            paginated,
            reference: build().await,
        }
    }

    /// Every child of `dir`, taken `limit` at a time, which is how a caller walks a directory:
    /// the token ends the walk, never a short page.
    async fn names(store: &ObjectStore, dir: &str, limit: Option<usize>) -> Vec<String> {
        let mut names = Vec::new();
        let mut page_token = None;
        loop {
            let page = store
                .read_dir_page(Path::from(dir), ReadDirOptions { page_token, limit })
                .await
                .unwrap();
            names.extend(page.values.iter().map(label));
            page_token = page.next_token;
            if page_token.is_none() {
                return names;
            }
            assert!(
                names.len() <= SHAPES.len() + MARKED_CHILDREN.len(),
                "the walk is serving entries more than once: {names:?}"
            );
        }
    }

    /// The listing a caller gets with no paging at all, which is what paging has to match.
    async fn reference_names(&self, dir: &str) -> Vec<String> {
        let names = Self::names(&self.reference, dir, None).await;
        let unique: HashSet<&String> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "fixture has duplicates: {names:?}"
        );
        names
    }

    /// Paging must be invisible: whatever the page size, walking the directory a page at a
    /// time produces exactly the reference listing.
    async fn assert_paging_is_invisible(&self, dir: &str) {
        let reference = self.reference_names(dir).await;
        assert!(!reference.is_empty(), "fixture listed nothing under {dir}");

        for &page_size in PAGE_SIZES {
            assert_eq!(
                Self::names(&self.paginated, dir, Some(page_size)).await,
                reference,
                "a page size of {page_size} changed the listing"
            );
        }
    }
}

/// A conformance pair over an emulator behaving as `model` says.
async fn emulated(client: Client, model: StoreModel) -> (Conformance, ListEmulator) {
    let emulator = ListEmulator::start(model, client.wire()).await;
    let (inner, paginated) = client.build(emulator.url());
    let stores =
        Conformance::new(inner, NativeDirLister::for_store(paginated), "s3://bucket/").await;
    (stores, emulator)
}

/// Every client, against every way a store is allowed to spend a page limit. A listing that
/// only works when the store is generous is not a working listing.
#[rstest]
#[tokio::test]
async fn test_paging_is_invisible_over_the_wire(
    #[values(Client::S3, Client::Gcs, Client::Azure)] client: Client,
    #[values(BudgetMode::PerEntry, BudgetMode::PerScannedKey)] budget: BudgetMode,
) {
    let keys: Vec<&str> = SHAPES.iter().chain(MARKED_CHILDREN).copied().collect();
    let model = StoreModel::new(keys).with_budget(budget);
    let (stores, emulator) = emulated(client, model).await;

    // A listing that cannot make progress would spin rather than fail, so bound the wait
    // instead of hanging a CI run. Each page is a real request, so the timeout can fire.
    tokio::time::timeout(
        Duration::from_secs(30),
        stores.assert_paging_is_invisible(DIR),
    )
    .await
    .expect("paging did not finish");

    // Loose, but it catches a listing that only terminates by exhausting the directory over
    // and over. The tightest case is page_size 1 over ~25 children, twice per page size.
    assert!(
        emulator.requests() < 1000,
        "paging took {} requests",
        emulator.requests()
    );
}

/// The page size has to reach the backend. A listing that quietly fell back to listing the
/// whole directory would satisfy every invariant above, since it answers the same thing, so
/// what tells the two apart is the request: one entry at page size one asks for one key.
#[rstest]
#[tokio::test]
async fn test_a_page_of_one_asks_the_backend_for_one(
    #[values(Client::S3, Client::Gcs, Client::Azure)] client: Client,
) {
    let (stores, emulator) = emulated(client, StoreModel::new(SHAPES.to_vec())).await;

    let page = stores
        .paginated
        .read_dir_page(
            Path::from(DIR),
            ReadDirOptions {
                page_token: None,
                limit: Some(1),
            },
        )
        .await
        .unwrap();

    assert_eq!(page.values.len(), 1);
    assert_eq!(
        emulator.limits(),
        vec![Some(1)],
        "the page size was not pushed down to the backend"
    );
}

/// A caller that asks for no page size at all, over a directory larger than the store will
/// return in one request.
///
/// Nothing then bounds a page but the store's own limit, so the continuation token is the only
/// thing carrying the listing forward. The listing still has to be complete.
#[rstest]
#[tokio::test]
async fn test_a_listing_without_a_page_size_is_complete(
    #[values(Client::S3, Client::Gcs, Client::Azure)] client: Client,
) {
    let model = StoreModel::new(SHAPES.to_vec())
        // Small enough that the fixture takes several pages, so the listing has to resume.
        .with_page_bound(4);
    let (stores, _emulator) = emulated(client, model).await;

    let reference = stores.reference_names(DIR).await;
    let listed = Conformance::names(&stores.paginated, DIR, None).await;

    assert_eq!(listed, reference);
}

/// A store that reports a directory marker — a key that is just the directory itself — must
/// not have it served as a child of itself.
#[rstest]
#[tokio::test]
async fn test_a_directory_marker_is_not_its_own_child(
    #[values(Client::S3, Client::Gcs, Client::Azure)] client: Client,
) {
    let (stores, _emulator) = emulated(client, StoreModel::new(MARKED_CHILDREN.to_vec())).await;

    let listed = Conformance::names(&stores.paginated, "db/marked", Some(1)).await;
    assert_eq!(listed, vec!["inside"]);
}

/// A page boundary landing on a directory costs nothing extra. The continuation token names a
/// position past everything the directory collapsed, so resuming from it does not re-scan the
/// keys behind that prefix however the store spends its page limit.
#[rstest]
#[tokio::test]
async fn test_resuming_at_a_directory_costs_one_page_per_child(
    #[values(BudgetMode::PerEntry, BudgetMode::PerScannedKey)] budget: BudgetMode,
) {
    // Three tables, each holding more files than a page is allowed to return, so every page
    // boundary lands on a directory whose contents the next request has to get past.
    let keys: Vec<String> = ["a", "b", "c"]
        .iter()
        .flat_map(|table| (0..20).map(move |file| format!("db/{table}.lance/data/{file}.lance")))
        .collect();
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    let model = StoreModel::new(keys).with_budget(budget);
    let (stores, emulator) = emulated(Client::S3, model).await;

    let names = Conformance::names(&stores.paginated, DIR, Some(1)).await;

    assert_eq!(names, vec!["a.lance/", "b.lance/", "c.lance/"]);
    assert_eq!(
        emulator.requests(),
        3,
        "expected one request per child however the store spends its page limit"
    );
}

/// Same fixtures against a real bucket. Not run in CI: it needs credentials, and it exists
/// to answer the questions an emulator cannot — whether each store's resume position works
/// at all, and what a page limit is really spent on.
///
/// ```ignore
/// LANCE_READ_DIR_TEST_URI=s3://my-bucket/read-dir-conformance RUST_LOG=info \
///   cargo test -p lance-io --lib read_dir::conformance::cloud -- --ignored --nocapture
/// ```
mod cloud {
    use super::*;

    /// The bucket and prefix to run against, e.g. `s3://bucket/prefix`, `gs://…`, `az://…`.
    const URI_VAR: &str = "LANCE_READ_DIR_TEST_URI";
    /// How many objects the cost fixture puts in one directory.
    const OBJECTS_VAR: &str = "LANCE_READ_DIR_BUDGET_OBJECTS";

    fn base_uri() -> String {
        std::env::var(URI_VAR)
            .unwrap_or_else(|_| panic!("set {URI_VAR} to a bucket and prefix to run this test"))
            .trim_end_matches('/')
            .to_string()
    }

    /// Create `keys` under `root`, relative to the store's own prefix.
    async fn create(store: &ObjectStore, root: &Path, keys: &[String]) {
        futures::stream::iter(keys.iter().map(|key| {
            // `Path::parse`, not `Path::from`: a key holding a character `Path::from` would
            // encode has to reach the store as it was written, or the fixture is not the
            // fixture any more.
            let path = Path::parse(format!("{root}/{key}")).unwrap();
            async move { store.put(&path, b"x").await.unwrap() }
        }))
        .buffer_unordered(32)
        .collect::<Vec<_>>()
        .await;
    }

    async fn remove(store: &ObjectStore, root: &Path) {
        let paths: Vec<Path> = store
            .list(Some(root.clone()))
            .map_ok(|meta| meta.location)
            .try_collect()
            .await
            .unwrap();
        futures::stream::iter(paths.iter().map(|path| store.delete(path)))
            .buffer_unordered(32)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
    }

    /// The shapes fixture, minus the directory marker, which cannot be created through a
    /// `Path`. Names are created with `Path::parse` so the store holds them as written
    /// rather than percent-encoded.
    #[tokio::test]
    #[ignore = "needs cloud credentials; run by hand"]
    async fn test_shapes_against_a_real_store() {
        let uri = format!("{}/shapes", base_uri());
        // The store as a caller would get it, provider wiring and all.
        let (store, root) = ObjectStore::from_uri(&uri).await.unwrap();
        let keys: Vec<String> = SHAPES.iter().map(|key| key.to_string()).collect();

        remove(&store, &root).await;
        create(&store, &root, &keys).await;

        let stores = Conformance {
            reference: reference_of(&store),
            paginated: store.as_ref().clone(),
        };
        stores
            .assert_paging_is_invisible(&format!("{root}/{DIR}"))
            .await;

        remove(&store, &root).await;
    }

    /// What a page boundary on a large directory costs. Prints the request count per page
    /// size: that number is the reason this test exists.
    #[test_log::test(tokio::test)]
    #[ignore = "needs cloud credentials and creates thousands of objects; run by hand"]
    async fn test_page_cost_against_a_real_store() {
        let objects: usize = std::env::var(OBJECTS_VAR)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(10_000);
        let uri = format!("{}/cost", base_uri());
        // The store as a caller would get it, provider wiring and all.
        let (store, root) = ObjectStore::from_uri(&uri).await.unwrap();

        // One big table whose keys a resumed page has to get past, and two siblings after it
        // so a page boundary lands on the big one.
        let mut keys: Vec<String> = (0..objects)
            .map(|file| format!("{DIR}/big.lance/data/{file:07}.lance"))
            .collect();
        keys.push(format!("{DIR}/next.lance/data/1.lance"));
        keys.push(format!("{DIR}/zzz.txt"));

        remove(&store, &root).await;
        create(&store, &root, &keys).await;

        let dir = format!("{root}/{DIR}");
        for page_size in [1usize, 2, 10] {
            let before = store.io_tracker().stats().read_iops;
            let names = Conformance::names(&store, &dir, Some(page_size)).await;
            let requests = store.io_tracker().stats().read_iops - before;
            assert_eq!(names, vec!["big.lance/", "next.lance/", "zzz.txt"]);
            // The measurement this test exists for. `test-log` prints it under `--nocapture`.
            tracing::info!(page_size, requests, objects, "list requests for 3 children");
            // One request per child is the floor, and the continuation token should hold the
            // listing to it: resuming after `big.lance/` must not re-scan what that prefix
            // collapsed.
            assert!(
                requests <= 32,
                "page_size {page_size} took {requests} requests"
            );
        }

        remove(&store, &root).await;
    }

    /// The same store with the pushdown removed, so it lists directories in full.
    fn reference_of(store: &Arc<ObjectStore>) -> ObjectStore {
        let mut reference = store.as_ref().clone();
        reference.paginated_lister = None;
        reference
    }
}
