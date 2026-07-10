// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use bytes::{Bytes, BytesMut};
use lance_arrow::DataTypeExt;
use lance_file::{
    previous::writer::ManifestProvider as PreviousManifestProvider, version::LanceFileVersion,
};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use std::collections::HashMap;
use std::{ops::Range, sync::Arc};
use tracing::instrument;

use lance_core::{Error, Result, datatypes::Schema};
use lance_io::{
    encodings::{Encoder, binary::BinaryEncoder, plain::PlainEncoder},
    object_store::ObjectStore,
    traits::{Reader, WriteExt, Writer},
    utils::read_message,
};

use crate::format::{
    DataStorageFormat, IndexMetadata, MAGIC, Manifest, ManifestSectionSource, Transaction, pb,
};

use super::commit::ManifestLocation;
use super::manifest_lance;

/// Returns true when footer bytes identify the single-file columnar manifest format.
pub fn is_columnar_manifest_footer(tail: &Bytes) -> Result<bool> {
    manifest_lance::is_columnar_footer(tail)
}

/// Reads a columnar manifest while reusing tail bytes already fetched by the caller.
pub async fn read_columnar_manifest_with_prefetched_tail(
    object_store: &ObjectStore,
    reader: Arc<dyn Reader>,
    tail: Bytes,
) -> Result<Manifest> {
    manifest_lance::read_with_prefetched_tail(object_store, reader, tail).await
}

/// Read Manifest on URI.
///
/// This only reads manifest files. It does not read data files.
#[instrument(level = "debug", skip(object_store))]
pub async fn read_manifest(
    object_store: &ObjectStore,
    path: &Path,
    known_size: Option<u64>,
) -> Result<Manifest> {
    let file_size = if let Some(known_size) = known_size {
        known_size
    } else {
        object_store.inner.head(path).await?.size
    };
    const PREFETCH_SIZE: u64 = 64 * 1024;
    let initial_start = file_size.saturating_sub(PREFETCH_SIZE);
    let range = Range {
        start: initial_start,
        end: file_size,
    };
    let buf = object_store.inner.get_range(path, range).await?;

    // In case of corruption, the known_size might be wrong. We can retry without
    // the size to be more robust.
    if (buf.len() < 16 || !buf.ends_with(MAGIC)) && known_size.is_some() {
        return Box::pin(read_manifest(object_store, path, None)).await;
    }

    if buf.len() < 16 {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: file size is smaller than 16 bytes".to_string(),
        ));
    }
    if !buf.ends_with(MAGIC) {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: magic number does not match".to_string(),
        ));
    }

    if is_columnar_manifest_footer(&buf)? {
        let reader: Arc<dyn Reader> = if let Some(size) = known_size {
            object_store
                .open_with_size(path, size as usize)
                .await?
                .into()
        } else {
            object_store.open(path).await?.into()
        };
        return read_columnar_manifest_with_prefetched_tail(object_store, reader, buf).await;
    }

    let manifest_pos = LittleEndian::read_i64(&buf[buf.len() - 16..buf.len() - 8]) as usize;
    let manifest_len = file_size as usize - manifest_pos;

    let buf: Bytes = if manifest_len <= buf.len() {
        // The prefetch captured the entire manifest. We just need to trim the buffer.
        buf.slice(buf.len() - manifest_len..buf.len())
    } else {
        // The prefetch only captured part of the manifest. We need to make an
        // additional range request to read the remainder.
        let mut buf2: BytesMut = object_store
            .inner
            .get_range(
                path,
                Range {
                    start: manifest_pos as u64,
                    end: file_size - PREFETCH_SIZE,
                },
            )
            .await?
            .into_iter()
            .collect();
        buf2.extend_from_slice(&buf);
        buf2.freeze()
    };

    let recorded_length = LittleEndian::read_u32(&buf[0..4]) as usize;
    // Need to trim the magic number at end and message length at beginning
    let buf = buf.slice(4..buf.len() - 16);

    if buf.len() != recorded_length {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest length does not match. Expected {}, got {}",
            recorded_length,
            buf.len()
        )));
    }

    let proto = pb::Manifest::decode(buf)?;
    Manifest::try_from(proto)
}

/// Auxiliary protobuf sections decoded opportunistically from an already-read
/// manifest tail.
#[derive(Debug)]
pub struct PrefetchedManifestSections {
    /// Index metadata when the complete legacy section was present in the tail.
    pub indices: Option<Vec<IndexMetadata>>,
    /// Inline transaction when the complete legacy section was present in the tail.
    pub transaction: Option<Transaction>,
}

/// Decodes legacy auxiliary sections only when they are fully contained in `tail`.
///
/// Columnar, unknown columnar, absent, and non-tail-resident sections return
/// `None` without I/O. Persisted offsets and message lengths are bounds checked
/// before protobuf decoding.
pub fn decode_prefetched_manifest_sections(
    manifest: &Manifest,
    tail: &Bytes,
    file_size: usize,
    path: &Path,
) -> Result<PrefetchedManifestSections> {
    if tail.len() > file_size {
        return Err(Error::corrupt_file(
            path.clone(),
            format!(
                "prefetched manifest tail has {} bytes but file size is {file_size}",
                tail.len()
            ),
        ));
    }

    let indices = match manifest.index_section_source()? {
        Some(ManifestSectionSource::ProtobufOffset(offset)) => {
            prefetched_message_data(tail, file_size, offset, path, "index section")?
                .map(pb::IndexSection::decode)
                .transpose()?
                .map(|section| {
                    section
                        .indices
                        .into_iter()
                        .map(IndexMetadata::try_from)
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
        }
        Some(
            ManifestSectionSource::ColumnarProjection | ManifestSectionSource::UnknownProjection,
        )
        | None => None,
    };
    let transaction = match manifest.transaction_section_source()? {
        Some(ManifestSectionSource::ProtobufOffset(offset)) => {
            prefetched_message_data(tail, file_size, offset, path, "transaction section")?
                .map(pb::Transaction::decode)
                .transpose()?
                .map(Transaction::from)
        }
        Some(
            ManifestSectionSource::ColumnarProjection | ManifestSectionSource::UnknownProjection,
        )
        | None => None,
    };
    Ok(PrefetchedManifestSections {
        indices,
        transaction,
    })
}

fn prefetched_message_data<'a>(
    tail: &'a Bytes,
    file_size: usize,
    offset: usize,
    path: &Path,
    section_name: &str,
) -> Result<Option<&'a [u8]>> {
    let remaining = file_size.checked_sub(offset).ok_or_else(|| {
        Error::corrupt_file(
            path.clone(),
            format!("{section_name} offset {offset} is beyond file size {file_size}"),
        )
    })?;
    if remaining > tail.len() {
        return Ok(None);
    }
    let start = tail.len() - remaining;
    let payload_start = start.checked_add(4).ok_or_else(|| {
        Error::corrupt_file(
            path.clone(),
            format!("{section_name} length offset overflows"),
        )
    })?;
    if payload_start > tail.len() {
        return Err(Error::corrupt_file(
            path.clone(),
            format!("{section_name} length prefix is truncated"),
        ));
    }
    let message_len = LittleEndian::read_u32(&tail[start..payload_start]) as usize;
    let payload_end = payload_start.checked_add(message_len).ok_or_else(|| {
        Error::corrupt_file(
            path.clone(),
            format!("{section_name} payload length overflows"),
        )
    })?;
    if payload_end > tail.len() {
        return Err(Error::corrupt_file(
            path.clone(),
            format!("{section_name} payload is truncated"),
        ));
    }
    Ok(Some(&tail[payload_start..payload_end]))
}

#[instrument(level = "debug", skip(object_store, manifest))]
pub async fn read_manifest_indexes(
    object_store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<Vec<IndexMetadata>> {
    match manifest.index_section_source()? {
        None => Ok(Vec::new()),
        Some(ManifestSectionSource::ProtobufOffset(pos)) => {
            let result = read_index_section(object_store, &location.path, location.size, pos).await;
            // A stale cached size makes the index offset fall outside the sized view,
            // so the read fails as "file size is too small". Retry once with the true
            // size; surface any other error unchanged.
            let section = match result {
                Err(error)
                    if location.size.is_some()
                        && error.to_string().contains("file size is too small") =>
                {
                    read_index_section(object_store, &location.path, None, pos).await?
                }
                other => other?,
            };
            section
                .indices
                .into_iter()
                .map(IndexMetadata::try_from)
                .collect()
        }
        Some(
            ManifestSectionSource::ColumnarProjection | ManifestSectionSource::UnknownProjection,
        ) => {
            let result =
                manifest_lance::read_indexes(object_store, &location.path, location.size).await;
            // A cached manifest size can point at an older object generation.
            // Reopen without it once so the Lance footer and section metadata are
            // resolved against the current object.
            match result {
                Err(_) if location.size.is_some() => {
                    manifest_lance::read_indexes(object_store, &location.path, None).await
                }
                other => other,
            }
        }
    }
}

/// Reads the inline transaction from its container-specific manifest section.
///
/// Returns `None` when the manifest has no inline transaction. Callers remain
/// responsible for following [`Manifest::transaction_file`] when present.
#[instrument(level = "debug", skip(object_store, manifest))]
pub async fn read_manifest_transaction(
    object_store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<Option<Transaction>> {
    match manifest.transaction_section_source()? {
        None => Ok(None),
        Some(ManifestSectionSource::ProtobufOffset(pos)) => {
            let result =
                read_transaction_section(object_store, &location.path, location.size, pos).await;
            let transaction = match result {
                Err(error)
                    if location.size.is_some()
                        && error.to_string().contains("file size is too small") =>
                {
                    read_transaction_section(object_store, &location.path, None, pos).await?
                }
                other => other?,
            };
            Ok(Some(Transaction::from(transaction)))
        }
        Some(
            ManifestSectionSource::ColumnarProjection | ManifestSectionSource::UnknownProjection,
        ) => {
            let result =
                manifest_lance::read_transaction(object_store, &location.path, location.size).await;
            match result {
                Err(_) if location.size.is_some() => {
                    manifest_lance::read_transaction(object_store, &location.path, None).await
                }
                other => other,
            }
        }
    }
}

/// Read the index section message at `pos`, opening the manifest with a known
/// size when one is provided.
async fn read_index_section(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
    pos: usize,
) -> Result<pb::IndexSection> {
    let reader = if let Some(size) = size {
        object_store.open_with_size(path, size as usize).await?
    } else {
        object_store.open(path).await?
    };
    read_message(reader.as_ref(), pos).await
}

async fn read_transaction_section(
    object_store: &ObjectStore,
    path: &Path,
    size: Option<u64>,
    pos: usize,
) -> Result<pb::Transaction> {
    let reader = if let Some(size) = size {
        object_store.open_with_size(path, size as usize).await?
    } else {
        object_store.open(path).await?
    };
    read_message(reader.as_ref(), pos).await
}

async fn do_write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    mut transaction: Option<Transaction>,
) -> Result<usize> {
    manifest.clear_section_locations();

    // Write indices if presented.
    if let Some(indices) = indices.as_ref() {
        let section = pb::IndexSection {
            indices: indices.iter().map(|i| i.into()).collect(),
        };
        let pos = writer.write_protobuf(&section).await?;
        manifest.index_section = Some(pos);
    }

    // Write inline transaction if presented.
    if let Some(tx) = transaction.take() {
        // Convert to protobuf at the write boundary to persist inline
        let pb_tx: pb::Transaction = tx.into();
        let pos = writer.write_protobuf(&pb_tx).await?;
        manifest.transaction_section = Some(pos);
    }

    writer.write_struct(manifest).await
}

/// Write manifest to an open file.
pub async fn write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    transaction: Option<Transaction>,
) -> Result<usize> {
    if manifest.data_storage_format.lance_file_version()?.resolve() == LanceFileVersion::V2_3 {
        return Err(Error::invalid_input(
            "storage version 2.3 requires the columnar manifest writer",
        ));
    }

    // Write dictionary values.
    let max_field_id = manifest.schema.max_field_id().unwrap_or(-1);
    let is_legacy_storage = manifest.should_use_legacy_format();
    for field_id in 0..max_field_id + 1 {
        if let Some(field) = manifest.schema.mut_field_by_id(field_id)
            && field.data_type().is_dictionary()
            && is_legacy_storage
        {
            let dict_info = field.dictionary.as_mut().ok_or_else(|| {
                Error::io(format!("Lance field {} misses dictionary info", field.name))
            })?;

            let value_arr = dict_info.values.as_ref().ok_or_else(|| {
                Error::io(format!(
                    "Lance field {} is dictionary type, but misses the dictionary value array",
                    field.name
                ))
            })?;

            let data_type = value_arr.data_type();
            let pos = match data_type {
                dt if dt.is_numeric() => {
                    let mut encoder = PlainEncoder::new(writer, dt);
                    encoder.encode(&[value_arr]).await?
                }
                dt if dt.is_binary_like() => {
                    let mut encoder = BinaryEncoder::new(writer);
                    encoder.encode(&[value_arr]).await?
                }
                _ => {
                    return Err(Error::schema(format!(
                        "Does not support {} as dictionary value type",
                        value_arr.data_type()
                    )));
                }
            };
            dict_info.offset = pos;
            dict_info.length = value_arr.len();
        }
    }

    let position = do_write_manifest(writer, manifest, indices, transaction).await?;
    manifest.set_protobuf_section_offsets();
    Ok(position)
}

/// Implementation of ManifestProvider that describes a Lance file by writing
/// a manifest that contains nothing but default fields and the schema
pub struct ManifestDescribing {}

#[async_trait]
impl PreviousManifestProvider for ManifestDescribing {
    async fn store_schema(
        object_writer: &mut dyn Writer,
        schema: &Schema,
    ) -> Result<Option<usize>> {
        let mut manifest = Manifest::new(
            schema.clone(),
            Arc::new(vec![]),
            DataStorageFormat::new(LanceFileVersion::Legacy),
            HashMap::new(),
        );
        let pos = do_write_manifest(object_writer, &mut manifest, None, None).await?;
        Ok(Some(pos))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatch};
    use std::collections::HashMap;

    use crate::format::SelfDescribingFileReader;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_file::format::{MAGIC, MAJOR_VERSION, MINOR_VERSION};
    use lance_file::previous::{
        reader::FileReader as PreviousFileReader, writer::FileWriter as PreviousFileWriter,
    };
    use rand::{Rng, distr::Alphanumeric};
    use tokio::io::AsyncWriteExt;

    use super::*;

    fn append_protobuf_message(message: &impl Message, bytes: &mut Vec<u8>) -> usize {
        let offset = bytes.len();
        let payload = message.encode_to_vec();
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        offset
    }

    #[test]
    fn decodes_typed_prefetched_protobuf_sections() {
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_2),
            HashMap::new(),
        );
        let mut bytes = Vec::new();
        manifest.index_section = Some(append_protobuf_message(
            &pb::IndexSection::default(),
            &mut bytes,
        ));
        manifest.transaction_section = Some(append_protobuf_message(
            &pb::Transaction::default(),
            &mut bytes,
        ));
        manifest.set_protobuf_section_offsets();
        let tail = Bytes::from(bytes);

        let prefetched = decode_prefetched_manifest_sections(
            &manifest,
            &tail,
            tail.len(),
            &Path::from("/prefetched.manifest"),
        )
        .unwrap();
        assert_eq!(prefetched.indices, Some(Vec::new()));
        assert!(prefetched.transaction.is_some());
    }

    #[test]
    fn skips_columnar_prefetch_and_checks_legacy_bounds() {
        let path = Path::from("/prefetched.manifest");
        let mut columnar = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );
        columnar.set_columnar_section_presence(true, true);
        let invalid_tail = Bytes::from_static(&[0xff]);
        let prefetched =
            decode_prefetched_manifest_sections(&columnar, &invalid_tail, 1, &path).unwrap();
        assert!(prefetched.indices.is_none());
        assert!(prefetched.transaction.is_none());

        let serialized = Manifest::try_from(pb::Manifest::from(&columnar)).unwrap();
        let prefetched =
            decode_prefetched_manifest_sections(&serialized, &invalid_tail, 1, &path).unwrap();
        assert!(prefetched.indices.is_none());
        assert!(prefetched.transaction.is_none());

        let mut legacy = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_2),
            HashMap::new(),
        );
        legacy.index_section = Some(0);
        let truncated = Bytes::from_static(&[8, 0, 0, 0, 1]);
        let error =
            decode_prefetched_manifest_sections(&legacy, &truncated, truncated.len(), &path)
                .unwrap_err();
        assert!(matches!(&error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("index section payload is truncated")
        );

        legacy.index_section = Some(8);
        let error = decode_prefetched_manifest_sections(&legacy, &truncated, 5, &path).unwrap_err();
        assert!(matches!(&error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("beyond file size"));
    }

    async fn test_roundtrip_manifest(prefix_size: usize, manifest_min_size: usize) {
        let store = ObjectStore::memory();
        let path = Path::from("/read_large_manifest");

        let mut writer = store.create(&path).await.unwrap();

        // Write prefix we should ignore
        let prefix: Vec<u8> = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(prefix_size)
            .collect();
        writer.write_all(&prefix).await.unwrap();

        let long_name: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(manifest_min_size)
            .map(char::from)
            .collect();

        let arrow_schema =
            ArrowSchema::new(vec![ArrowField::new(long_name, DataType::Int64, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let mut config = HashMap::new();
        config.insert("key".to_string(), "value".to_string());

        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        let pos = write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap();
        writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let roundtripped_manifest = read_manifest(&store, &path, None).await.unwrap();

        assert_eq!(manifest, roundtripped_manifest);

        store.inner.delete(&path).await.unwrap();
    }

    #[tokio::test]
    async fn test_protobuf_writer_rejects_storage_v2_3() {
        let store = ObjectStore::memory();
        let path = Path::from("/protobuf-v2-3.manifest");
        let mut writer = store.create(&path).await.unwrap();
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_3),
            HashMap::new(),
        );

        let error = write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("requires the columnar"));
    }

    #[tokio::test]
    async fn test_protobuf_writer_clears_stale_section_offsets() {
        let store = ObjectStore::memory();
        let path = Path::from("/protobuf-clears-sections.manifest");
        let mut writer = store.create(&path).await.unwrap();
        let mut manifest = Manifest::new(
            Schema::default(),
            Arc::new(Vec::new()),
            DataStorageFormat::new(LanceFileVersion::V2_2),
            HashMap::new(),
        );
        manifest.index_section = Some(17);
        manifest.transaction_section = Some(29);

        write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap();

        assert_eq!(manifest.index_section, None);
        assert_eq!(manifest.transaction_section, None);
        assert_eq!(manifest.index_section_source().unwrap(), None);
        assert_eq!(manifest.transaction_section_source().unwrap(), None);
    }

    #[tokio::test]
    async fn test_read_large_manifest() {
        test_roundtrip_manifest(0, 100_000).await;
        test_roundtrip_manifest(1000, 100_000).await;
        test_roundtrip_manifest(1000, 1000).await;
    }

    #[tokio::test]
    async fn test_update_schema_metadata() {
        let store = ObjectStore::memory();
        let path = Path::from("/update_schema_metadata");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = PreviousFileWriter::<ManifestDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        let array = Int32Array::from_iter_values(0..10);
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array)]).unwrap();
        file_writer
            .write(std::slice::from_ref(&batch))
            .await
            .unwrap();
        let mut metadata = HashMap::new();
        metadata.insert(String::from("lance:extra"), String::from("for_test"));
        file_writer.finish_with_metadata(&metadata).await.unwrap();

        let reader = store.open(&path).await.unwrap();
        let reader = PreviousFileReader::try_new_self_described_from_reader(reader.into(), None)
            .await
            .unwrap();
        let schema = ArrowSchema::from(reader.schema());
        assert_eq!(schema.metadata().get("lance:extra").unwrap(), "for_test");
    }
}
