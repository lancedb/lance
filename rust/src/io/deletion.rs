use std::{collections::HashSet, sync::Arc};

use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::ipc::writer::{FileWriter as ArrowFileWriter, IpcWriteOptions};
use arrow::ipc::CompressionType;
use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Buf;
use object_store::path::Path;
use rand::Rng;
use roaring::bitmap::RoaringBitmap;

use crate::format::{DeletionFile, DeletionFileType, Fragment};

use crate::{Error, Result};

use super::ObjectStore;

/// Threshold for when a DeletionVector::Set should be promoted to a DeletionVector::Bitmap.
const BITMAP_THRESDHOLD: usize = 5_000;
// TODO: Benchmark to find a better value.

/// Represents a set of deleted row ids in a single fragment.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DeletionVector {
    NoDeletions,
    Set(HashSet<u32>),
    Bitmap(RoaringBitmap),
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
            DeletionVector::NoDeletions => Box::new(std::iter::empty()),
            DeletionVector::Set(set) => Box::new(set.into_iter()),
            DeletionVector::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }
}

impl FromIterator<u32> for DeletionVector {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let iter = iter.into_iter();
        match iter.size_hint() {
            (_, Some(0)) => DeletionVector::NoDeletions,
            (lower, _) if lower > BITMAP_THRESDHOLD => {
                let bitmap = iter.collect::<RoaringBitmap>();
                DeletionVector::Bitmap(bitmap)
            }
            (_, Some(upper)) if upper < BITMAP_THRESDHOLD => {
                let set = iter.collect::<HashSet<_>>();
                DeletionVector::Set(set)
            }
            _ => {
                // We don't know the size, so just try as a set and move to bitmap
                // if it ends up being big.
                let set = iter.collect::<HashSet<_>>();
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.into_iter().collect::<RoaringBitmap>();
                    DeletionVector::Bitmap(bitmap)
                } else {
                    DeletionVector::Set(set)
                }
            }
        }
    }
}

fn deletion_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "row_id",
        DataType::UInt32,
        false,
    )]))
}

fn deletion_file_path(fragment_id: u64, read_version: u64, id: u64, suffix: &str) -> Path {
    Path::from(format!(
        "_deletions/{fragment_id}-{read_version}-{id}.{suffix}"
    ))
}

#[allow(dead_code)]
pub(crate) async fn write_deletion_file(
    fragment_id: u64,
    read_version: u64,
    removed_rows: &DeletionVector,
    object_store: &ObjectStore,
) -> Result<Option<DeletionFile>> {
    match removed_rows {
        DeletionVector::NoDeletions => Ok(None),
        DeletionVector::Set(set) => {
            let id = rand::thread_rng().gen::<u64>();
            let path = deletion_file_path(fragment_id, read_version, id, "arrow");

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

            Ok(Some(DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Array,
            }))
        }
        DeletionVector::Bitmap(bitmap) => {
            let id = rand::thread_rng().gen::<u64>();
            let path = deletion_file_path(fragment_id, read_version, id, "bin");

            let mut out: Vec<u8> = Vec::new();
            bitmap.serialize_into(&mut out)?;

            object_store.inner.put(&path, out.into()).await?;

            Ok(Some(DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Bitmap,
            }))
        }
    }
}

#[allow(dead_code)]
pub(crate) async fn read_deletion_file(
    fragment: Fragment,
    object_store: &ObjectStore,
) -> Result<Option<DeletionVector>> {
    let deletion_file = if let Some(file) = fragment.deletion_file {
        file
    } else {
        return Ok(None);
    };

    match deletion_file.file_type {
        DeletionFileType::Array => {
            let path = deletion_file_path(
                fragment.id,
                deletion_file.read_version,
                deletion_file.id,
                "arrow",
            );

            let data = object_store.inner.get(&path).await?.bytes().await?;
            let data = std::io::Cursor::new(data);
            let mut batches: Vec<RecordBatch> = ArrowFileReader::try_new(data, None)?
                .collect::<std::result::Result<_, ArrowError>>()?;

            if batches.len() != 1 {
                return Err(Error::IO(format!(
                    "Expected exactly one batch in deletion file, got {}",
                    batches.len()
                )));
            }

            let batch = batches.pop().unwrap();
            if batch.schema() != deletion_arrow_schema() {
                return Err(Error::IO(format!(
                    "Expected schema {:?} in deletion file, got {:?}",
                    deletion_arrow_schema(),
                    batch.schema()
                )));
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
                    return Err(Error::IO(
                        "Null values are not allowed in deletion files".to_string(),
                    ));
                }
            }

            Ok(Some(DeletionVector::Set(set)))
        }
        DeletionFileType::Bitmap => {
            let path = deletion_file_path(
                fragment.id,
                deletion_file.read_version,
                deletion_file.id,
                "bin",
            );

            let data = object_store.inner.get(&path).await?.bytes().await?;
            let reader = data.reader();
            let bitmap = RoaringBitmap::deserialize_from(reader)?;

            Ok(Some(DeletionVector::Bitmap(bitmap)))
        }
    }
}
