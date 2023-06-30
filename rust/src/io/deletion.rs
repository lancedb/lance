use std::ops::Range;
use std::slice::Iter;
use std::sync::Mutex;
use std::{collections::HashSet, sync::Arc};

use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::ipc::writer::{FileWriter as ArrowFileWriter, IpcWriteOptions};
use arrow::ipc::CompressionType;
use arrow_array::{BooleanArray, RecordBatch, UInt32Array};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Buf;
use lru_time_cache::LruCache;
use object_store::path::Path;
use rand::Rng;
use roaring::bitmap::RoaringBitmap;
use snafu::ResultExt;

use super::ObjectStore;
use crate::dataset::DELETION_DIRS;
use crate::error::{box_error, CorruptFileSnafu};
use crate::format::{DeletionFile, DeletionFileType, Fragment, Manifest};
use crate::{Error, Result};

/// Threshold for when a DeletionVector::Set should be promoted to a DeletionVector::Bitmap.
const BITMAP_THRESDHOLD: usize = 5_000;
// TODO: Benchmark to find a better value.

/// Represents a set of deleted row ids in a single fragment.
#[derive(Debug, Clone)]
pub enum DeletionVector {
    NoDeletions,
    Set(HashSet<u32>),
    Bitmap(RoaringBitmap),
}

impl DeletionVector {
    #[allow(dead_code)] // Used in tests
    pub fn len(&self) -> usize {
        match self {
            Self::NoDeletions => 0,
            Self::Set(set) => set.len(),
            Self::Bitmap(bitmap) => bitmap.len() as usize,
        }
    }

    pub fn contains(&self, i: u32) -> bool {
        match self {
            Self::NoDeletions => false,
            Self::Set(set) => set.contains(&i),
            Self::Bitmap(bitmap) => bitmap.contains(i),
        }
    }

    pub fn contains_range(&self, mut range: Range<u32>) -> bool {
        match self {
            Self::NoDeletions => range.is_empty(),
            Self::Set(set) => range.all(|i| set.contains(&i)),
            Self::Bitmap(bitmap) => bitmap.contains_range(range),
        }
    }

    pub fn build_predicate(&self, row_ids: Iter<u64>) -> Option<BooleanArray> {
        match self {
            Self::Bitmap(bitmap) => Some(
                row_ids
                    .map(|&id| !bitmap.contains(id as u32))
                    .collect::<Vec<_>>(),
            ),
            Self::Set(set) => Some(
                row_ids
                    .map(|&id| !set.contains(&(id as u32)))
                    .collect::<Vec<_>>(),
            ),
            Self::NoDeletions => None,
        }
        .map(BooleanArray::from)
    }
}

impl Default for DeletionVector {
    fn default() -> Self {
        Self::NoDeletions
    }
}

impl PartialEq for DeletionVector {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NoDeletions, Self::NoDeletions) => true,
            (Self::Set(set1), Self::Set(set2)) => set1 == set2,
            (Self::Bitmap(bitmap1), Self::Bitmap(bitmap2)) => bitmap1 == bitmap2,
            (Self::Set(set), Self::Bitmap(bitmap)) | (Self::Bitmap(bitmap), Self::Set(set)) => {
                let set = set.iter().copied().collect::<RoaringBitmap>();
                set == *bitmap
            }
            _ => false,
        }
    }
}

impl Extend<u32> for DeletionVector {
    fn extend<T: IntoIterator<Item = u32>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        // The mem::replace allows changing the variant of Self when we only
        // have &mut Self.
        *self = match (std::mem::take(self), iter.size_hint()) {
            (Self::NoDeletions, (_, Some(0))) => Self::NoDeletions,
            (Self::NoDeletions, (lower, _)) if lower >= BITMAP_THRESDHOLD => {
                let bitmap = iter.collect::<RoaringBitmap>();
                Self::Bitmap(bitmap)
            }
            (Self::NoDeletions, (_, Some(upper))) if upper < BITMAP_THRESDHOLD => {
                let set = iter.collect::<HashSet<_>>();
                Self::Set(set)
            }
            (Self::NoDeletions, _) => {
                // We don't know the size, so just try as a set and move to bitmap
                // if it ends up being big.
                let set = iter.collect::<HashSet<_>>();
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.into_iter().collect::<RoaringBitmap>();
                    Self::Bitmap(bitmap)
                } else {
                    Self::Set(set)
                }
            }
            (Self::Set(mut set), _) => {
                set.extend(iter);
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.drain().collect::<RoaringBitmap>();
                    Self::Bitmap(bitmap)
                } else {
                    Self::Set(set)
                }
            }
            (Self::Bitmap(mut bitmap), _) => {
                bitmap.extend(iter);
                Self::Bitmap(bitmap)
            }
        };
    }
}

// TODO: impl methods for DeletionVector
/// impl DeletionVector {
///     pub fn get(i: u32) -> bool { ... }
/// }
/// impl BitAnd for DeletionVector { ... }

impl IntoIterator for DeletionVector {
    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;
    type Item = u32;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => Box::new(set.into_iter()),
            Self::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }
}

impl FromIterator<u32> for DeletionVector {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut deletion_vector = Self::default();
        deletion_vector.extend(iter);
        deletion_vector
    }
}

/// Get the Arrow schema for an Arrow deletion file.
fn deletion_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "row_id",
        DataType::UInt32,
        false,
    )]))
}

/// Get the file path for a deletion file. This is relative to the dataset root.
pub fn deletion_file_path(base: &Path, fragment_id: u64, deletion_file: &DeletionFile) -> Path {
    let DeletionFile {
        read_version,
        id,
        file_type,
    } = deletion_file;
    let suffix = file_type.suffix();
    base.child(DELETION_DIRS)
        .child(format!("{fragment_id}-{read_version}-{id}.{suffix}"))
}

/// Write a deletion file for a fragment for a given deletion vector.
///
/// Returns the deletion file if one was written. If no deletions were present,
/// returns `Ok(None)`.
pub async fn write_deletion_file(
    base: &Path,
    fragment_id: u64,
    read_version: u64,
    removed_rows: &DeletionVector,
    object_store: &ObjectStore,
) -> Result<Option<DeletionFile>> {
    match removed_rows {
        DeletionVector::NoDeletions => Ok(None),
        DeletionVector::Set(set) => {
            let id = rand::thread_rng().gen::<u64>();
            let deletion_file = DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Array,
            };
            let path = deletion_file_path(base, fragment_id, &deletion_file);

            let array = UInt32Array::from_iter(set.iter().copied());
            let array = Arc::new(array);

            let schema = deletion_arrow_schema();
            let batch = RecordBatch::try_new(schema.clone(), vec![array])?;

            let mut out: Vec<u8> = Vec::new();
            let write_options =
                IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))?;
            {
                let mut writer = ArrowFileWriter::try_new_with_options(
                    &mut out,
                    schema.as_ref(),
                    write_options,
                )?;
                writer.write(&batch)?;
                writer.finish()?;
                // Drop writer so out is no longer borrowed.
            }

            object_store.inner.put(&path, out.into()).await?;

            Ok(Some(deletion_file))
        }
        DeletionVector::Bitmap(bitmap) => {
            let id = rand::thread_rng().gen::<u64>();
            let deletion_file = DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Bitmap,
            };
            let path = deletion_file_path(base, fragment_id, &deletion_file);

            let mut out: Vec<u8> = Vec::new();
            bitmap.serialize_into(&mut out)?;

            object_store.inner.put(&path, out.into()).await?;

            Ok(Some(deletion_file))
        }
    }
}

/// Read a deletion file for a fragment.
///
/// Returns the deletion vector if one was present. Otherwise returns `Ok(None)`.
///
/// Will return an error if the file is present but invalid.
pub async fn read_deletion_file(
    base: &Path,
    fragment: &Fragment,
    object_store: &ObjectStore,
) -> Result<Option<DeletionVector>> {
    let deletion_file = if let Some(file) = &fragment.deletion_file {
        file
    } else {
        return Ok(None);
    };

    match deletion_file.file_type {
        DeletionFileType::Array => {
            let path = deletion_file_path(base, fragment.id, deletion_file);

            let data = object_store.inner.get(&path).await?.bytes().await?;
            let data = std::io::Cursor::new(data);
            let mut batches: Vec<RecordBatch> = ArrowFileReader::try_new(data, None)?
                .collect::<std::result::Result<_, ArrowError>>()
                .map_err(box_error)
                .context(CorruptFileSnafu { path: path.clone() })?;

            if batches.len() != 1 {
                return Err(Error::corrupt_file(
                    path,
                    format!(
                        "Expected exactly one batch in deletion file, got {}",
                        batches.len()
                    ),
                ));
            }

            let batch = batches.pop().unwrap();
            if batch.schema() != deletion_arrow_schema() {
                return Err(Error::corrupt_file(
                    path,
                    format!(
                        "Expected schema {:?} in deletion file, got {:?}",
                        deletion_arrow_schema(),
                        batch.schema()
                    ),
                ));
            }

            let array = batch.columns()[0]
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();

            let mut set = HashSet::with_capacity(array.len());
            for val in array.iter() {
                if let Some(val) = val {
                    set.insert(val);
                } else {
                    return Err(Error::corrupt_file(
                        path,
                        "Null values are not allowed in deletion files",
                    ));
                }
            }

            Ok(Some(DeletionVector::Set(set)))
        }
        DeletionFileType::Bitmap => {
            let path = deletion_file_path(base, fragment.id, deletion_file);

            let data = object_store.inner.get(&path).await?.bytes().await?;
            let reader = data.reader();
            let bitmap = RoaringBitmap::deserialize_from(reader)
                .map_err(box_error)
                .context(CorruptFileSnafu { path })?;

            Ok(Some(DeletionVector::Bitmap(bitmap)))
        }
    }
}

pub struct LruDeletionVectorStore {
    // can't clone mutex, so need to arc it
    cache: Arc<Mutex<LruCache<u64, Arc<DeletionVector>>>>,
    object_store: Arc<ObjectStore>,
    path: Path,
    manifest: Arc<Manifest>,
}

impl LruDeletionVectorStore {
    pub(crate) fn new(
        object_store: Arc<ObjectStore>,
        path: Path,
        manifest: Arc<Manifest>,
        cache_capacity: usize,
    ) -> Self {
        Self {
            cache: Arc::new(Mutex::new(LruCache::with_capacity(cache_capacity))),
            object_store,
            path,
            manifest,
        }
    }

    pub async fn is_deleted(&self, row_id: u64) -> Result<bool> {
        let frag_id = row_id >> 32;
        let local_row_id = row_id as u32;

        let deletion_vec = {
            let val_in_cache: Option<Arc<DeletionVector>> = {
                let mut cache = self.cache.lock().unwrap();
                cache.get(&frag_id).cloned()
            };

            // Lock is released while we do IO so we block others or poision the lock
            if val_in_cache.is_none() {
                let fragment = self
                    .manifest
                    .as_ref()
                    .fragments
                    .as_ref()
                    .iter()
                    .find(|frag| frag.id == frag_id);
                let dvec = match fragment {
                    Some(frag) => {
                        let dvec = read_deletion_file(&self.path, frag, self.object_store.as_ref())
                            .await?;
                        dvec.unwrap_or(DeletionVector::NoDeletions)
                    }
                    None => DeletionVector::NoDeletions,
                };

                // IO is done, now lock again
                let mut cache = self.cache.lock().unwrap();
                cache.insert(frag_id, Arc::new(dvec));
                cache.get(&frag_id).unwrap().clone()
            } else {
                val_in_cache.unwrap()
            }
        };

        Ok(match deletion_vec.as_ref() {
            DeletionVector::Bitmap(bitmap) => bitmap.contains(local_row_id),
            DeletionVector::Set(set) => set.contains(&local_row_id),
            DeletionVector::NoDeletions => false,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_deletion_vector() {
        let set = HashSet::from_iter(0..100);
        let bitmap = RoaringBitmap::from_iter(0..100);

        let set_dv = DeletionVector::Set(set);
        let bitmap_dv = DeletionVector::Bitmap(bitmap);

        assert_eq!(set_dv, bitmap_dv);
    }

    #[test]
    fn test_threshold() {
        let dv = DeletionVector::from_iter(0..(BITMAP_THRESDHOLD as u32));
        assert!(matches!(dv, DeletionVector::Bitmap(_)));
    }

    #[tokio::test]
    async fn test_write_no_deletions() {
        let dv = DeletionVector::NoDeletions;

        let (object_store, path) = ObjectStore::from_uri("memory:///no_deletion")
            .await
            .unwrap();
        let file = write_deletion_file(&path, 0, 0, &dv, &object_store)
            .await
            .unwrap();
        assert!(file.is_none());
    }

    #[tokio::test]
    async fn test_write_array() {
        let dv = DeletionVector::Set(HashSet::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/write");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store)
            .await
            .unwrap();

        assert!(matches!(
            file,
            Some(DeletionFile {
                file_type: DeletionFileType::Array,
                ..
            })
        ));

        let file = file.unwrap();
        assert_eq!(file.read_version, read_version);
        let path = deletion_file_path(&path, fragment_id, &file);
        assert_eq!(
            path,
            Path::from(format!("/write/_deletions/21-12-{}.arrow", file.id))
        );

        let data = object_store
            .inner
            .get(&path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let data = std::io::Cursor::new(data);
        let mut batches: Vec<RecordBatch> = ArrowFileReader::try_new(data, None)
            .unwrap()
            .collect::<std::result::Result<_, ArrowError>>()
            .unwrap();

        assert_eq!(batches.len(), 1);
        let batch = batches.pop().unwrap();
        assert_eq!(batch.schema(), deletion_arrow_schema());
        let array = batch["row_id"]
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let read_dv = DeletionVector::from_iter(array.iter().map(|v| v.unwrap()));
        assert_eq!(read_dv, dv);
    }

    #[tokio::test]
    async fn test_write_bitmap() {
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/bitmap");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store)
            .await
            .unwrap();

        assert!(matches!(
            file,
            Some(DeletionFile {
                file_type: DeletionFileType::Bitmap,
                ..
            })
        ));

        let file = file.unwrap();
        assert_eq!(file.read_version, read_version);
        let path = deletion_file_path(&path, fragment_id, &file);
        assert_eq!(
            path,
            Path::from(format!("/bitmap/_deletions/21-12-{}.bin", file.id))
        );

        let data = object_store
            .inner
            .get(&path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let reader = data.reader();
        let read_bitmap = RoaringBitmap::deserialize_from(reader).unwrap();
        assert_eq!(read_bitmap, dv.into_iter().collect::<RoaringBitmap>());
    }

    #[tokio::test]
    async fn test_roundtrip_array() {
        let dv = DeletionVector::Set(HashSet::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/roundtrip");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store)
            .await
            .unwrap();

        let mut fragment = Fragment::new(fragment_id);
        fragment.deletion_file = file;
        let read_dv = read_deletion_file(&path, &fragment, &object_store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read_dv, dv);
    }

    #[tokio::test]
    async fn test_roundtrip_bitmap() {
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter(0..100));

        let fragment_id = 21;
        let read_version = 12;

        let object_store = ObjectStore::memory();
        let path = Path::from("/bitmap");
        let file = write_deletion_file(&path, fragment_id, read_version, &dv, &object_store)
            .await
            .unwrap();

        let mut fragment = Fragment::new(fragment_id);
        fragment.deletion_file = file;
        let read_dv = read_deletion_file(&path, &fragment, &object_store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read_dv, dv);
    }
}
