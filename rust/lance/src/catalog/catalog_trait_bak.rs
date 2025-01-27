// // Copyright 2024 Lance Developers.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
//
// use std::fmt::Debug;
// use std::sync::Arc;
// use futures::stream::BoxStream;
// use object_store::path::Path;
// use url::Url;
// use lance_core::Error;
// use lance_io::object_store::{ObjectStore, ObjectStoreExt, ObjectStoreParams, WrappingObjectStore};
// use object_store::{ObjectStore as OSObjectStore};
// use lance_table::format::{Index, Manifest};
// use lance_table::io::commit::{CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter};
//
// use crate::{Dataset, Result};
// use crate::dataset::builder::DatasetBuilder;
//
// /// Contains all the information that Lance needs to access a table
// pub struct TableReference {
//     /// Base URI of the table
//     pub uri: String,
//     /// Object store wrapper used to access the table
//     pub store_wrapper: Option<Arc<dyn WrappingObjectStore>>,
//     /// Commit handler used to handle new commits to the table
//     pub commit_handler: Arc<dyn CommitHandler>,
//     /// Parameters used to read / write to the object store
//     pub store_params: Option<ObjectStoreParams>,
// }
//
// /// Trait to be implemented by any catalog that Lance can use
// #[async_trait::async_trait]
// pub trait Catalog: std::fmt::Debug + Send + Sync {
//     /// Create a new table in the catalog
//     ///
//     /// Returns a table reference that can be used to read/write to the table
//     async fn create_table(&self, name: &str) -> Result<TableReference>;
//     /// Drop a table from the catalog
//     async fn drop_table(&self, name: &str) -> Result<()>;
//     /// Get a reference to a table in the catalog
//     ///
//     /// Returns a table reference that can be used to read/write to the table
//     async fn get_table(&self, name: &str) -> Result<TableReference>;
//     /// List all tables in the catalog
//     ///
//     /// The `start_after` parameter is an optional table name that, if provided, will
//     /// start the listing after the named table. If the named table is not found, the
//     /// listing will start after the table that would be named if it existed.
//     ///
//     /// The `limit` parameter is an optional limit on the number of tables returned.
//     async fn list_tables(
//         &self,
//         start_after: Option<String>,
//         limit: Option<u32>,
//     ) -> Result<Vec<String>>;
// }
//
// /// A simplified catalog that puts all tables in a single base directory
// ///
// /// The object store's CAS primitives are used for commit handling.
// ///
// /// This object store is simplistic but has zero dependencies (beyond an object store
// /// of some kind)
// #[derive(Debug)]
// pub struct ListingCatalog {
//     base_path: Path,
//     object_store: ObjectStore,
//     commit_handler: Arc<dyn CommitHandler>,
// }
//
// const LANCE_EXTENSION: &str = "lance";
//
// impl ListingCatalog {
//
//     pub fn new(base_path: Path, object_store: ObjectStore) -> Self {
//         Self {
//             base_path,
//             object_store: object_store.clone(),
//             commit_handler: Arc::new(
//                 ListingCommitHandler { object_store: object_store.into() }
//             ),
//         }
//     }
//
// }
//
// #[async_trait::async_trait]
// impl Catalog for ListingCatalog {
//     async fn create_table(&self, name: &str) -> Result<TableReference> {
//         let table_path = self.base_path.child(name);
//         DatasetBuilder::from_uri(table_path)
//             .with_commit_handler(self.commit_handler.clone())
//             .with_object_store(self.object_store.clone().inner, Url::parse(table_path.clone().as_ref()).unwrap(), self.commit_handler.clone())
//             .load()
//             .await?;
//         let ds = Dataset::open(table_path.as_ref()).await?;
//
//         Ok(
//             TableReference {
//                 uri: ds.uri().to_string(),
//                 store_wrapper: Some(self.object_store.into()),
//                 commit_handler: self.commit_handler.clone(),
//                 store_params: Some(ObjectStoreParams {
//                     block_size: None,
//                     object_store: None,
//                     s3_credentials_refresh_offset: Default::default(),
//                     aws_credentials: None,
//                     object_store_wrapper: None,
//                     storage_options: None,
//                     use_constant_size_upload_parts: false,
//                     list_is_lexically_ordered: None,
//                 }),
//             }
//         )
//     }
//
//     async fn drop_table(&self, name: &str) -> Result<()> {
//         todo!()
//     }
//
//     async fn get_table(&self, name: &str) -> Result<TableReference> {
//         todo!()
//     }
//
//     async fn list_tables(
//         &self,
//         start_after: Option<String>,
//         limit: Option<u32>,
//     ) -> Result<Vec<String>> {
//         let mut f = self
//             .object_store
//             .read_dir(self.base_path.clone())
//             .await?
//             .iter()
//             .map(std::path::Path::new)
//             .filter(|path| {
//                 let is_lance = path
//                     .extension()
//                     .and_then(|e| e.to_str())
//                     .map(|e| e == LANCE_EXTENSION);
//                 is_lance.unwrap_or(false)
//             })
//             .filter_map(|p| p.file_stem().and_then(|s| s.to_str().map(String::from)))
//             .collect::<Vec<String>>();
//         f.sort();
//         if let Some(start_after) = start_after {
//             let index = f
//                 .iter()
//                 .position(|name| name.as_str() > start_after.as_str())
//                 .unwrap_or(f.len());
//             f.drain(0..index);
//         }
//         if let Some(limit) = limit {
//             f.truncate(limit as usize);
//         }
//         Ok(f)
//     }
// }
//
// pub struct ListingCommitHandler {
//     object_store: ObjectStore,
// }
//
// impl Debug for ListingCommitHandler {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("ListingCommitHandler").finish()
//     }
// }
//
// impl CommitHandler for ListingCommitHandler {
//     async fn resolve_latest_location(
//         &self,
//         base_path: &Path,
//         object_store: &ObjectStore,
//     ) -> Result<ManifestLocation> {
//         let manifests = object_store
//             .read_dir(base_path.clone())
//             .await?
//             .iter()
//             .filter(|path| Path::from(path).extension().and_then(|e| e.to_str()) == Some("manifest"))
//             .collect::<Vec<_>>();
//
//         if manifests.is_empty() {
//             return Err(lance_core::Error::InvalidTableLocation {
//                 message: "No manifests found".to_string(),
//             });
//
//         }
//
//         let latest_manifest = manifests
//             .iter()
//             .max_by_key(|path| {
//                 path.file_stem()
//                     .and_then(|stem| stem.to_str())
//                     .and_then(|s| s.parse::<u64>().ok())
//             })
//             .ok_or_else(|| lance_core::Error::InvalidTableLocation {
//                 message: "Failed to find latest manifest"
//             }.to_string())?;
//
//         Ok(ManifestLocation {
//             version: 0,
//             path: Path::from(latest_manifest.clone()),
//             size: None,
//             naming_scheme: ManifestNamingScheme::V1,
//         })
//     }
//
//
//     async fn resolve_latest_version(&self, base_path: &Path, object_store: &ObjectStore) -> std::result::Result<Path, Error> {
//         let manifests = object_store
//             .read_dir(base_path.clone())
//             .await?
//             .iter()
//             .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("manifest"))
//             .collect::<Vec<_>>();
//
//         if manifests.is_empty() {
//             return Err("No manifests found".to_string());
//         }
//
//         let latest_manifest = manifests
//             .iter()
//             .max_by_key(|path| {
//                 path.file_stem()
//                     .and_then(|stem| stem.to_str())
//                     .and_then(|s| s.parse::<u64>().ok())
//             })
//             .ok_or_else(|| Err("Failed to find latest manifest".to_string()))?;
//
//         Ok(latest_manifest.clone())
//     }
//
//     async fn resolve_latest_version_id(&self, base_path: &Path, object_store: &ObjectStore) -> Result<u64> {
//         let manifests = object_store
//             .read_dir(base_path.clone())
//             .await?
//             .iter()
//             .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("manifest"))
//             .collect::<Vec<_>>();
//
//         if manifests.is_empty() {
//             return Err("No manifests found".to_string());
//         }
//
//         let latest_version = manifests
//             .iter()
//             .filter_map(|path| {
//                 path.file_stem()
//                     .and_then(|stem| stem.to_str())
//                     .and_then(|s| s.parse::<u64>().ok())
//             })
//             .max()
//             .ok_or_else(|| Err("Failed to find latest version".to_string()))?;
//
//         Ok(latest_version)
//     }
//
//     async fn resolve_version(
//         &self,
//         base_path: &Path,
//         version: u64,
//         object_store: &dyn OSObjectStore,
//     ) -> std::result::Result<Path, Error> {
//         let version_path = base_path.child(format!("{:020}.manifest", version));
//         if object_store.exists(&version_path).await? {
//             Ok(version_path)
//         } else {
//             Err(Error::Execution {
//                 message: format!(
//                     "Manifest for version {} not found",
//                     version
//                 ),
//                 location: Default::default(),
//             })
//         }
//     }
//
//
//     async fn resolve_version_location(
//         &self,
//         base_path: &Path,
//         version: u64,
//         object_store: &dyn OSObjectStore,
//     ) -> Result<ManifestLocation> {
//         let version_path = self.resolve_version(base_path, version, object_store).await?;
//         Ok(ManifestLocation { version, path: version_path, size: None, naming_scheme: ManifestNamingScheme::V1 })
//     }
//
//
//     async fn list_manifests<'a>(
//         &self,
//         base_path: &Path,
//         object_store: &'a dyn OSObjectStore,
//     ) -> Result<BoxStream<'a, Result<Path>>> {
//         let manifests = object_store
//             .read_dir_all(base_path.clone(), None)
//             .await?
//             .iter()
//             .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("manifest"))
//             .cloned()
//             .collect::<Vec<_>>();
//
//         let stream = futures::stream::iter(manifests.into_iter().map(Ok));
//         Ok(Box::pin(stream))
//     }
//
//
//     async fn commit(
//         &self,
//         manifest: &mut Manifest,
//         indices: Option<Vec<Index>>,
//         base_path: &Path,
//         object_store: &ObjectStore,
//         manifest_writer: ManifestWriter,
//         naming_scheme: ManifestNamingScheme,
//     ) -> std::result::Result<Path, CommitError> {
//         // Generate the next version number
//         let latest_version = self
//             .resolve_latest_version_id(base_path, object_store)
//             .await
//             .unwrap_or(0);
//         let next_version = latest_version + 1;
//
//         // Write the manifest to the object store
//         let manifest_path = base_path.child(format!("{:020}.manifest", next_version));
//         manifest_writer(object_store, manifest, indices, &manifest_path).await?;
//
//         Ok(manifest_path)
//     }
//
//
//     async fn delete(&self, base_path: &Path) -> Result<()> {
//         let manifests = self
//             .list_manifests(base_path, &self.object_store.inner)
//             .await?
//             .collect::<Vec<_>>()
//             .await;
//
//         for manifest in manifests {
//             if let Ok(path) = manifest {
//                 self.object_store.delete(&path).await?;
//             }
//         }
//
//         Ok(())
//     }
//
// }