// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, sync::Arc};

use arrow_array::{RecordBatch, UInt32Array};
use arrow_ipc::reader::FileReader as ArrowFileReader;
use arrow_ipc::writer::{FileWriter as ArrowFileWriter, IpcWriteOptions};
use arrow_ipc::CompressionType;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use bytes::Buf;
use lance_core::error::{box_error, CorruptFileSnafu};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::Rng;
use roaring::bitmap::RoaringBitmap;
use snafu::{location, Location, ResultExt};
use tracing::instrument;

use crate::format::{DeletionFile, DeletionFileType, Fragment};

pub(crate) const DELETION_DIRS: &str = "_deletions";

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
        ..
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
                num_deleted_rows: Some(set.len()),
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

            object_store.put(&path, &out).await?;

            Ok(Some(deletion_file))
        }
        DeletionVector::Bitmap(bitmap) => {
            let id = rand::thread_rng().gen::<u64>();
            let deletion_file = DeletionFile {
                read_version,
                id,
                file_type: DeletionFileType::Bitmap,
                num_deleted_rows: Some(bitmap.len() as usize),
            };
            let path = deletion_file_path(base, fragment_id, &deletion_file);

            let mut out: Vec<u8> = Vec::new();
            bitmap.serialize_into(&mut out)?;

            object_store.put(&path, &out).await?;

            Ok(Some(deletion_file))
        }
    }
}

/// Read a deletion file for a fragment.
///
/// Returns the deletion vector if one was present. Otherwise returns `Ok(None)`.
///
/// Will return an error if the file is present but invalid.
#[instrument(level = "debug", skip_all)]
pub async fn read_deletion_file(
    base: &Path,
    fragment: &Fragment,
    object_store: &ObjectStore,
) -> Result<Option<DeletionVector>> {
    let Some(deletion_file) = &fragment.deletion_file else {
        return Ok(None);
    };

    match deletion_file.file_type {
        DeletionFileType::Array => {
            let path = deletion_file_path(base, fragment.id, deletion_file);

            let data = object_store.read_one_all(&path).await?;
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
                    location!(),
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
                    location!(),
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
                        location!(),
                    ));
                }
            }

            Ok(Some(DeletionVector::Set(set)))
        }
        DeletionFileType::Bitmap => {
            let path = deletion_file_path(base, fragment.id, deletion_file);

            let data = object_store.read_one_all(&path).await?;
            let reader = data.reader();
            let bitmap = RoaringBitmap::deserialize_from(reader)
                .map_err(box_error)
                .context(CorruptFileSnafu { path })?;

            Ok(Some(DeletionVector::Bitmap(bitmap)))
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

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
