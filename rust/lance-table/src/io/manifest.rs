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
    traits::{WriteExt, Writer},
    utils::read_message,
};

use crate::format::{
    DataStorageFormat, IndexMetadata, MAGIC, Manifest, ROW_ADDRESS_B_FAST, Transaction, pb,
};

use super::commit::ManifestLocation;

/// Tail bytes fetched by the manifest reader before the 2.3 row-address layout.
pub const BASE_MANIFEST_TAIL_PREFETCH_SIZE: u64 = 64 * 1024;

/// Manifest-only tail prefetch; data and index readers retain their block sizes.
///
/// If row-address-only metadata is at most [`ROW_ADDRESS_B_FAST`],
/// adding it cannot increase the manifest GET count over the previous 64-KiB
/// reader: a core manifest at most 64 KiB still needs one GET, while a larger
/// core manifest already needed two and this reader needs at most two. When a
/// second GET is needed its range ends where this tail starts, so bytes are not
/// fetched twice.
pub const MANIFEST_TAIL_PREFETCH_SIZE: u64 = BASE_MANIFEST_TAIL_PREFETCH_SIZE + ROW_ADDRESS_B_FAST;

/// A decoded manifest together with the bytes fetched by its initial tail GET.
///
/// Dataset open uses the tail to opportunistically decode an adjacent index
/// section without issuing another request. Transactions remain lazy even when
/// their bytes are present in this buffer.
pub struct ManifestReadResult {
    pub manifest: Manifest,
    pub prefetched_tail: Bytes,
    pub file_size: usize,
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
    Ok(
        read_manifest_with_prefetched_tail(object_store, path, known_size)
            .await?
            .manifest,
    )
}

/// Read a manifest and retain its initial manifest-specific tail prefetch.
#[instrument(level = "debug", skip(object_store))]
pub async fn read_manifest_with_prefetched_tail(
    object_store: &ObjectStore,
    path: &Path,
    known_size: Option<u64>,
) -> Result<ManifestReadResult> {
    let file_size = if let Some(known_size) = known_size {
        known_size
    } else {
        object_store.inner.head(path).await?.size
    };
    let initial_start = file_size.saturating_sub(MANIFEST_TAIL_PREFETCH_SIZE);
    let range = Range {
        start: initial_start,
        end: file_size,
    };
    let prefetched_tail = match object_store.inner.get_range(path, range).await {
        Ok(buf) => buf,
        // A cached size can be larger than the object, causing the range GET
        // itself to fail before a footer can be inspected. Retry once using HEAD.
        Err(_) if known_size.is_some() => {
            return Box::pin(read_manifest_with_prefetched_tail(object_store, path, None)).await;
        }
        Err(error) => return Err(error.into()),
    };

    // S3 may satisfy a range whose end is past EOF with a short response
    // instead of returning an error.  When the size came from a cache, a short
    // response means that size cannot be trusted even if the bytes happen to
    // end in a valid manifest footer.
    let expected_prefetch_len = file_size.saturating_sub(initial_start);
    if known_size.is_some() && prefetched_tail.len() as u64 != expected_prefetch_len {
        return Box::pin(read_manifest_with_prefetched_tail(object_store, path, None)).await;
    }

    // In case of corruption, the known_size might be wrong. We can retry without
    // the size to be more robust.
    if (prefetched_tail.len() < 16 || !prefetched_tail.ends_with(MAGIC)) && known_size.is_some() {
        return Box::pin(read_manifest_with_prefetched_tail(object_store, path, None)).await;
    }

    if file_size < 20 || prefetched_tail.len() < 20 {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: file size is smaller than the 4-byte length and 16-byte footer"
                .to_string(),
        ));
    }
    if !prefetched_tail.ends_with(MAGIC) {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: magic number does not match".to_string(),
        ));
    }
    let manifest_pos = LittleEndian::read_i64(
        &prefetched_tail[prefetched_tail.len() - 16..prefetched_tail.len() - 8],
    );
    if manifest_pos < 0 || manifest_pos as u64 > file_size.saturating_sub(20) {
        return Err(Error::corrupt_file(
            path.clone(),
            format!(
                "Invalid format: manifest offset {} is outside file size {}",
                manifest_pos, file_size
            ),
        ));
    }
    let manifest_pos = manifest_pos as u64;

    let manifest_bytes: Bytes = if manifest_pos >= initial_start {
        // The prefetch captured the entire manifest. We just need to trim the buffer.
        prefetched_tail.slice((manifest_pos - initial_start) as usize..)
    } else {
        // The prefetch only captured part of the manifest. We need to make an
        // additional, non-overlapping range request to read the remainder.
        let prefix = object_store
            .inner
            .get_range(
                path,
                Range {
                    start: manifest_pos,
                    end: initial_start,
                },
            )
            .await?;
        let mut combined = BytesMut::with_capacity(prefix.len() + prefetched_tail.len());
        combined.extend_from_slice(&prefix);
        combined.extend_from_slice(&prefetched_tail);
        combined.freeze()
    };

    if manifest_bytes.len() < 20 {
        return Err(Error::corrupt_file(
            path.clone(),
            "Invalid format: manifest section is smaller than its length prefix and footer"
                .to_string(),
        ));
    }
    let recorded_length = LittleEndian::read_u32(&manifest_bytes[0..4]) as usize;
    // Need to trim the magic number at end and message length at beginning
    let message = manifest_bytes.slice(4..manifest_bytes.len() - 16);

    if message.len() != recorded_length {
        return Err(Error::invalid_input(format!(
            "Invalid format: manifest length does not match. Expected {}, got {}",
            recorded_length,
            message.len()
        )));
    }

    let proto = pb::Manifest::decode(message)?;
    let manifest = Manifest::try_from(proto)?;
    let file_size = usize::try_from(file_size).map_err(|_| {
        Error::not_supported(format!(
            "manifest file size {} does not fit in usize",
            file_size
        ))
    })?;
    Ok(ManifestReadResult {
        manifest,
        prefetched_tail,
        file_size,
    })
}

#[instrument(level = "debug", skip(object_store, manifest))]
pub async fn read_manifest_indexes(
    object_store: &ObjectStore,
    location: &ManifestLocation,
    manifest: &Manifest,
) -> Result<Vec<IndexMetadata>> {
    if let Some(pos) = manifest.index_section.as_ref() {
        let result = read_index_section(object_store, &location.path, location.size, *pos).await;
        // A stale cached size makes the index offset fall outside the sized view,
        // so the read fails as "file size is too small". Retry once with the true
        // size; surface any other error unchanged.
        let section = match result {
            Err(e)
                if location.size.is_some() && e.to_string().contains("file size is too small") =>
            {
                read_index_section(object_store, &location.path, None, *pos).await?
            }
            other => other?,
        };

        let indices = section
            .indices
            .into_iter()
            .map(IndexMetadata::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(indices)
    } else {
        Ok(vec![])
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

async fn do_write_manifest(
    writer: &mut dyn Writer,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    mut transaction: Option<Transaction>,
) -> Result<usize> {
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

    do_write_manifest(writer, manifest, indices, transaction).await
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
    use rstest::rstest;
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

    const TEST_ROW_ADDRESS_DELTA_FIELD: u64 = 50_000;

    fn append_varint(mut value: u64, output: &mut Vec<u8>) {
        while value >= 0x80 {
            output.push((value as u8) | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    async fn write_sized_manifest(
        store: &ObjectStore,
        path: &Path,
        core_payload_size: usize,
        delta_size: usize,
    ) -> (Manifest, u64, u64) {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("value", DataType::Int64, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest
            .config
            .insert("core-payload".to_string(), "c".repeat(core_payload_size));

        let mut encoded = pb::Manifest::from(&manifest).encode_to_vec();
        let core_file_size = (encoded.len() + 4 + 16) as u64;
        if delta_size > 0 {
            // Append a length-delimited unknown protobuf field to model the 2.3
            // row-address layout without depending on its provisional schema.
            append_varint((TEST_ROW_ADDRESS_DELTA_FIELD << 3) | 2, &mut encoded);
            append_varint(delta_size as u64, &mut encoded);
            encoded.resize(encoded.len() + delta_size, 0);
        }

        let mut writer = store.create(path).await.unwrap();
        writer
            .write_all(&(encoded.len() as u32).to_le_bytes())
            .await
            .unwrap();
        writer.write_all(&encoded).await.unwrap();
        writer
            .write_magics(0, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let file_size = store.inner.head(path).await.unwrap().size;
        (manifest, core_file_size, file_size)
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
    async fn test_read_large_manifest() {
        test_roundtrip_manifest(0, 100_000).await;
        test_roundtrip_manifest(1000, 100_000).await;
        test_roundtrip_manifest(1000, 1000).await;
    }

    #[tokio::test]
    async fn test_small_manifest_prefetch_is_bounded_by_read_budget() {
        let store = ObjectStore::memory();
        let path = Path::from("/bounded_manifest_tail_prefetch");
        let mut writer = store.create(&path).await.unwrap();
        writer
            .write_all(&vec![0_u8; 3 * 1024 * 1024])
            .await
            .unwrap();

        let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int64,
            false,
        )]))
        .unwrap();
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        let position = write_manifest(writer.as_mut(), &mut manifest, None, None)
            .await
            .unwrap();
        writer
            .write_magics(position, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await
            .unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();
        let file_size = store.inner.head(&path).await.unwrap().size;
        store.io_stats_incremental();

        let actual = read_manifest(&store, &path, Some(file_size)).await.unwrap();
        let stats = store.io_stats_incremental();

        assert_eq!(actual, manifest);
        assert_eq!(stats.read_iops, 1);
        assert_eq!(stats.read_bytes, MANIFEST_TAIL_PREFETCH_SIZE);
    }

    #[rstest]
    #[case::small_core(32 * 1024, 1024 * 1024, 1)]
    #[case::medium_core(100 * 1024, 1024 * 1024, 1)]
    #[case::large_core(3 * 1024 * 1024, 1024 * 1024, 2)]
    #[tokio::test]
    async fn test_manifest_tail_prefetch_request_bound(
        #[case] core_payload_size: usize,
        #[case] delta_size: usize,
        #[case] expected_candidate_gets: u64,
    ) {
        let store = ObjectStore::memory();
        let path = Path::from("/manifest_tail_prefetch");
        let (expected, core_file_size, file_size) =
            write_sized_manifest(&store, &path, core_payload_size, delta_size).await;
        store.io_stats_incremental();

        let actual = read_manifest(&store, &path, Some(file_size)).await.unwrap();
        let stats = store.io_stats_incremental();

        assert_eq!(actual, expected);
        assert_eq!(stats.read_iops, expected_candidate_gets);
        assert_eq!(stats.read_bytes, file_size);

        let baseline_gets = if core_file_size <= BASE_MANIFEST_TAIL_PREFETCH_SIZE {
            1
        } else {
            2
        };
        assert!(
            stats.read_iops <= baseline_gets,
            "candidate GETs={} exceeded 64-KiB baseline GETs={} for core_file_size={} and delta_size={}",
            stats.read_iops,
            baseline_gets,
            core_file_size,
            delta_size
        );
    }

    #[tokio::test]
    async fn test_manifest_tail_prefetch_retries_stale_known_size() {
        let store = ObjectStore::memory();
        let path = Path::from("/manifest_tail_prefetch_stale_size");
        let (expected, _, file_size) =
            write_sized_manifest(&store, &path, 32 * 1024, 1024 * 1024).await;
        let stale_size = file_size - 1;
        store.io_stats_incremental();

        let actual = read_manifest_with_prefetched_tail(&store, &path, Some(stale_size))
            .await
            .unwrap();
        let stats = store.io_stats_incremental();

        assert_eq!(actual.manifest, expected);
        assert_eq!(actual.file_size as u64, file_size);
        assert_eq!(stats.read_iops, 3);
        assert_eq!(stats.read_bytes, stale_size + file_size);
    }

    #[tokio::test]
    async fn test_manifest_tail_prefetch_retries_oversized_known_size() {
        let store = ObjectStore::memory();
        let path = Path::from("/manifest_tail_prefetch_oversized_stale_size");
        let (expected, _, file_size) =
            write_sized_manifest(&store, &path, 32 * 1024, 1024 * 1024).await;
        let stale_size = file_size + 1;
        store.io_stats_incremental();

        let actual = read_manifest_with_prefetched_tail(&store, &path, Some(stale_size))
            .await
            .unwrap();

        assert_eq!(actual.manifest, expected);
        assert_eq!(actual.file_size as u64, file_size);
        assert!(store.io_stats_incremental().read_iops >= 2);
    }

    #[tokio::test]
    async fn test_manifest_tail_prefetch_rejects_footer_without_length_prefix() {
        let store = ObjectStore::memory();
        let path = Path::from("/manifest_footer_without_length_prefix");
        let mut bytes = vec![0_u8; 16];
        bytes[12..].copy_from_slice(MAGIC);
        let mut writer = store.create(&path).await.unwrap();
        writer.write_all(&bytes).await.unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let error = read_manifest(&store, &path, Some(bytes.len() as u64))
            .await
            .unwrap_err();

        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("4-byte length"));
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
