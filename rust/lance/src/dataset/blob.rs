// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, future::Future, ops::DerefMut, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::{UInt32Type, UInt64Type, UInt8Type};
use arrow_array::builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder};
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowDataType;
use lance_arrow::{FieldExt, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use object_store::path::Path;
use snafu::location;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::take::TakeBuilder;
use super::{Dataset, ProjectionRequest};
use arrow_array::StructArray;
use lance_core::datatypes::{BlobKind, BlobVersion};
use lance_core::utils::blob::blob_path;
use lance_core::{utils::address::RowAddress, Error, Result};
use lance_io::traits::{Reader, Writer};

const INLINE_MAX: usize = 64 * 1024; // 64KB inline cutoff
const DEDICATED_THRESHOLD: usize = 4 * 1024 * 1024; // 4MB dedicated cutoff
const PACK_FILE_MAX_SIZE: usize = 1024 * 1024 * 1024; // 1GiB per .pack sidecar

// Maintains rolling `.blob` sidecar files for packed blobs.
// Layout: data/{data_file_key}/{blob_id:08x}.blob where each file is an
// unframed concatenation of blob payloads; descriptors store (blob_id,
// position, size) to locate each slice. A dedicated struct keeps path state
// and rolling size separate from the per-batch preprocessor logic, so we can
// reuse the same writer across rows and close/roll files cleanly on finish.
struct PackWriter {
    object_store: ObjectStore,
    data_dir: Path,
    data_file_key: String,
    max_pack_size: usize,
    current_blob_id: Option<u32>,
    writer: Option<Box<dyn lance_io::traits::Writer>>,
    current_size: usize,
}

impl PackWriter {
    fn new(object_store: ObjectStore, data_dir: Path, data_file_key: String) -> Self {
        Self {
            object_store,
            data_dir,
            data_file_key,
            max_pack_size: PACK_FILE_MAX_SIZE,
            current_blob_id: None,
            writer: None,
            current_size: 0,
        }
    }

    async fn start_new_pack(&mut self, blob_id: u32) -> Result<()> {
        let path = blob_path(&self.data_dir, &self.data_file_key, blob_id);
        let writer = self.object_store.create(&path).await?;
        self.writer = Some(writer);
        self.current_blob_id = Some(blob_id);
        self.current_size = 0;
        Ok(())
    }

    /// Append `data` to the current `.blob` file, rolling to a new file when
    /// `max_pack_size` would be exceeded.
    ///
    /// alloc_blob_id: called only when a new pack file is opened; returns the
    /// blob_id used as the file name.
    ///
    /// Returns `(blob_id, position)` where
    /// position is the start offset of this payload in that pack file.
    async fn write_with_allocator<F>(
        &mut self,
        alloc_blob_id: &mut F,
        data: &[u8],
    ) -> Result<(u32, u64)>
    where
        F: FnMut() -> u32,
    {
        let len = data.len();
        if self
            .current_blob_id
            .map(|_| self.current_size + len > self.max_pack_size)
            .unwrap_or(true)
        {
            let blob_id = alloc_blob_id();
            self.finish().await?;
            self.start_new_pack(blob_id).await?;
        }

        let writer = self.writer.as_mut().expect("pack writer is initialized");
        let position = self.current_size as u64;
        writer.write_all(data).await?;
        self.current_size += len;
        Ok((self.current_blob_id.expect("pack blob id"), position))
    }

    async fn finish(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            Writer::shutdown(writer.as_mut()).await?;
        }
        self.current_blob_id = None;
        self.current_size = 0;
        Ok(())
    }
}

/// Preprocesses blob v2 columns on the write path so the encoder only sees lightweight descriptors:
///
/// - Spills large blobs to sidecar files before encoding, reducing memory/CPU and avoiding copying huge payloads through page builders.
/// - Emits `blob_id/blob_size` tied to the data file stem, giving readers a stable path independent of temporary fragment IDs assigned during write.
/// - Leaves small inline blobs and URI rows unchanged for compatibility.
pub struct BlobPreprocessor {
    object_store: ObjectStore,
    data_dir: Path,
    data_file_key: String,
    local_counter: u32,
    pack_writer: PackWriter,
    blob_v2_cols: Vec<bool>,
    dedicated_thresholds: Vec<usize>,
    writer_metadata: Vec<HashMap<String, String>>,
}

impl BlobPreprocessor {
    pub(crate) fn new(
        object_store: ObjectStore,
        data_dir: Path,
        data_file_key: String,
        schema: &lance_core::datatypes::Schema,
    ) -> Self {
        let pack_writer = PackWriter::new(
            object_store.clone(),
            data_dir.clone(),
            data_file_key.clone(),
        );
        let arrow_schema = arrow_schema::Schema::from(schema);
        let fields = arrow_schema.fields();
        let blob_v2_cols = fields.iter().map(|field| field.is_blob_v2()).collect();
        let dedicated_thresholds = fields
            .iter()
            .map(|field| dedicated_threshold_from_metadata(field.as_ref()))
            .collect();
        let writer_metadata = fields
            .iter()
            .map(|field| field.metadata().clone())
            .collect();
        Self {
            object_store,
            data_dir,
            data_file_key,
            // Start at 1 to avoid a potential all-zero blob_id value.
            local_counter: 1,
            pack_writer,
            blob_v2_cols,
            dedicated_thresholds,
            writer_metadata,
        }
    }

    fn next_blob_id(&mut self) -> u32 {
        let id = self.local_counter;
        self.local_counter += 1;
        id
    }

    async fn write_dedicated(&mut self, blob_id: u32, data: &[u8]) -> Result<Path> {
        let path = blob_path(&self.data_dir, &self.data_file_key, blob_id);
        let mut writer = self.object_store.create(&path).await?;
        writer.write_all(data).await?;
        Writer::shutdown(&mut writer).await?;
        Ok(path)
    }

    async fn write_packed(&mut self, data: &[u8]) -> Result<(u32, u64)> {
        let (counter, pack_writer) = (&mut self.local_counter, &mut self.pack_writer);
        pack_writer
            .write_with_allocator(
                &mut || {
                    let id = *counter;
                    *counter += 1;
                    id
                },
                data,
            )
            .await
    }
    pub(crate) async fn preprocess_batch(&mut self, batch: &RecordBatch) -> Result<RecordBatch> {
        let expected_columns = self.blob_v2_cols.len();
        if batch.num_columns() != expected_columns {
            return Err(Error::invalid_input(
                format!(
                    "Unexpected number of columns: expected {}, got {}",
                    expected_columns,
                    batch.num_columns()
                ),
                location!(),
            ));
        }

        let batch_schema = batch.schema();
        let batch_fields = batch_schema.fields();

        let mut new_columns = Vec::with_capacity(batch.num_columns());
        let mut new_fields = Vec::with_capacity(batch.num_columns());

        for idx in 0..batch.num_columns() {
            let array = batch.column(idx);
            let field = &batch_fields[idx];
            if !self.blob_v2_cols[idx] {
                new_columns.push(array.clone());
                new_fields.push(field.clone());
                continue;
            }

            let struct_arr = array
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| {
                    Error::invalid_input("Blob column was not a struct array", location!())
                })?;

            let data_col = struct_arr
                .column_by_name("data")
                .ok_or_else(|| {
                    Error::invalid_input("Blob struct missing `data` field", location!())
                })?
                .as_binary::<i64>();
            let uri_col = struct_arr
                .column_by_name("uri")
                .ok_or_else(|| {
                    Error::invalid_input("Blob struct missing `uri` field", location!())
                })?
                .as_string::<i32>();
            let position_col = struct_arr
                .column_by_name("position")
                .map(|col| col.as_primitive::<UInt64Type>());
            let size_col = struct_arr
                .column_by_name("size")
                .map(|col| col.as_primitive::<UInt64Type>());

            let mut data_builder = LargeBinaryBuilder::with_capacity(struct_arr.len(), 0);
            let mut uri_builder = StringBuilder::with_capacity(struct_arr.len(), 0);
            let mut blob_id_builder =
                PrimitiveBuilder::<arrow_array::types::UInt32Type>::with_capacity(struct_arr.len());
            let mut blob_size_builder =
                PrimitiveBuilder::<arrow_array::types::UInt64Type>::with_capacity(struct_arr.len());
            let mut kind_builder = PrimitiveBuilder::<UInt8Type>::with_capacity(struct_arr.len());
            let mut position_builder =
                PrimitiveBuilder::<arrow_array::types::UInt64Type>::with_capacity(struct_arr.len());

            let struct_nulls = struct_arr.nulls();

            for i in 0..struct_arr.len() {
                if struct_arr.is_null(i) {
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    kind_builder.append_null();
                    position_builder.append_null();
                    continue;
                }

                let has_data = !data_col.is_null(i);
                let has_uri = !uri_col.is_null(i);
                let has_position = position_col
                    .as_ref()
                    .map(|col| !col.is_null(i))
                    .unwrap_or(false);
                let has_size = size_col
                    .as_ref()
                    .map(|col| !col.is_null(i))
                    .unwrap_or(false);
                let data_len = if has_data { data_col.value(i).len() } else { 0 };

                let dedicated_threshold = self.dedicated_thresholds[idx];
                if has_data && data_len > dedicated_threshold {
                    let blob_id = self.next_blob_id();
                    self.write_dedicated(blob_id, data_col.value(i)).await?;

                    kind_builder.append_value(BlobKind::Dedicated as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(blob_id);
                    blob_size_builder.append_value(data_len as u64);
                    position_builder.append_null();
                    continue;
                }

                if has_data && data_len > INLINE_MAX {
                    let (pack_blob_id, position) = self.write_packed(data_col.value(i)).await?;

                    kind_builder.append_value(BlobKind::Packed as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(pack_blob_id);
                    blob_size_builder.append_value(data_len as u64);
                    position_builder.append_value(position);
                    continue;
                }

                if has_uri {
                    let uri_val = uri_col.value(i);
                    kind_builder.append_value(BlobKind::External as u8);
                    data_builder.append_null();
                    uri_builder.append_value(uri_val);
                    blob_id_builder.append_null();
                    if has_position && has_size {
                        let position = position_col
                            .as_ref()
                            .expect("position column must exist")
                            .value(i);
                        let size = size_col.as_ref().expect("size column must exist").value(i);
                        blob_size_builder.append_value(size);
                        position_builder.append_value(position);
                    } else {
                        blob_size_builder.append_null();
                        position_builder.append_null();
                    }
                    continue;
                }

                if has_data {
                    kind_builder.append_value(BlobKind::Inline as u8);
                    let value = data_col.value(i);
                    data_builder.append_value(value);
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    position_builder.append_null();
                } else {
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    kind_builder.append_null();
                    position_builder.append_null();
                }
            }

            let child_fields = vec![
                arrow_schema::Field::new("kind", ArrowDataType::UInt8, true),
                arrow_schema::Field::new("data", ArrowDataType::LargeBinary, true),
                arrow_schema::Field::new("uri", ArrowDataType::Utf8, true),
                arrow_schema::Field::new("blob_id", ArrowDataType::UInt32, true),
                arrow_schema::Field::new("blob_size", ArrowDataType::UInt64, true),
                arrow_schema::Field::new("position", ArrowDataType::UInt64, true),
            ];

            let struct_array = arrow_array::StructArray::try_new(
                child_fields.clone().into(),
                vec![
                    Arc::new(kind_builder.finish()),
                    Arc::new(data_builder.finish()),
                    Arc::new(uri_builder.finish()),
                    Arc::new(blob_id_builder.finish()),
                    Arc::new(blob_size_builder.finish()),
                    Arc::new(position_builder.finish()),
                ],
                struct_nulls.cloned(),
            )?;

            new_columns.push(Arc::new(struct_array));
            new_fields.push(Arc::new(
                arrow_schema::Field::new(
                    field.name(),
                    ArrowDataType::Struct(child_fields.into()),
                    field.is_nullable(),
                )
                .with_metadata(self.writer_metadata[idx].clone()),
            ));
        }

        let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            new_fields
                .iter()
                .map(|f| f.as_ref().clone())
                .collect::<Vec<_>>(),
            batch_schema.metadata().clone(),
        ));

        RecordBatch::try_new(new_schema, new_columns)
            .map_err(|e| Error::invalid_input(e.to_string(), location!()))
    }

    pub(crate) async fn finish(&mut self) -> Result<()> {
        self.pack_writer.finish().await
    }
}

fn dedicated_threshold_from_metadata(field: &arrow_schema::Field) -> usize {
    field
        .metadata()
        .get(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY)
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(DEDICATED_THRESHOLD)
}

pub async fn preprocess_blob_batches(
    batches: &[RecordBatch],
    pre: &mut BlobPreprocessor,
) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        out.push(pre.preprocess_batch(batch).await?);
    }
    Ok(out)
}

/// Current state of the reader.  Held in a mutex for easy sharing
///
/// The u64 is the cursor in the file that the reader is currently at
/// (note that seeks are allowed before the file is opened)
#[derive(Debug)]
enum ReaderState {
    Uninitialized(u64),
    Open((u64, Arc<dyn Reader>)),
    Closed,
}

/// A file-like object that represents a blob in a dataset
#[derive(Debug)]
pub struct BlobFile {
    object_store: Arc<ObjectStore>,
    path: Path,
    reader: Arc<Mutex<ReaderState>>,
    position: u64,
    size: u64,
    kind: BlobKind,
    uri: Option<String>,
}

impl BlobFile {
    /// Create a new BlobFile
    ///
    /// See [`crate::dataset::Dataset::take_blobs`]
    pub fn new_inline(
        dataset: Arc<Dataset>,
        field_id: u32,
        row_addr: u64,
        position: u64,
        size: u64,
    ) -> Self {
        let frag_id = RowAddress::from(row_addr).fragment_id();
        let frag = dataset.get_fragment(frag_id as usize).unwrap();
        let data_file = frag.data_file_for_field(field_id).unwrap();
        let data_file = dataset.data_dir().child(data_file.path.as_str());
        Self {
            object_store: dataset.object_store.clone(),
            path: data_file,
            position,
            size,
            kind: BlobKind::Inline,
            uri: None,
            reader: Arc::new(Mutex::new(ReaderState::Uninitialized(0))),
        }
    }

    pub fn new_dedicated(dataset: Arc<Dataset>, path: Path, size: u64) -> Self {
        Self {
            object_store: dataset.object_store.clone(),
            path,
            position: 0,
            size,
            kind: BlobKind::Dedicated,
            uri: None,
            reader: Arc::new(Mutex::new(ReaderState::Uninitialized(0))),
        }
    }
    pub fn new_packed(dataset: Arc<Dataset>, path: Path, position: u64, size: u64) -> Self {
        Self {
            object_store: dataset.object_store.clone(),
            path,
            position,
            size,
            kind: BlobKind::Packed,
            uri: None,
            reader: Arc::new(Mutex::new(ReaderState::Uninitialized(0))),
        }
    }
    pub async fn new_external(
        uri: String,
        position: u64,
        size: u64,
        registry: Arc<ObjectStoreRegistry>,
        params: Arc<ObjectStoreParams>,
    ) -> Result<Self> {
        let (object_store, path) =
            ObjectStore::from_uri_and_params(registry, &uri, &params).await?;
        let size = if size > 0 {
            size
        } else {
            object_store.size(&path).await?
        };
        Ok(Self {
            object_store,
            path,
            position,
            size,
            kind: BlobKind::External,
            uri: Some(uri),
            reader: Arc::new(Mutex::new(ReaderState::Uninitialized(0))),
        })
    }

    /// Close the blob file, releasing any associated resources
    pub async fn close(&self) -> Result<()> {
        let mut reader = self.reader.lock().await;
        *reader = ReaderState::Closed;
        Ok(())
    }

    /// Returns true if the blob file is closed
    pub async fn is_closed(&self) -> bool {
        matches!(*self.reader.lock().await, ReaderState::Closed)
    }

    async fn do_with_reader<
        T,
        Fut: Future<Output = Result<(u64, T)>>,
        Func: FnOnce(u64, Arc<dyn Reader>) -> Fut,
    >(
        &self,
        func: Func,
    ) -> Result<T> {
        let mut reader = self.reader.lock().await;
        if let ReaderState::Uninitialized(cursor) = *reader {
            let opened = self.object_store.open(&self.path).await?;
            let opened = Arc::<dyn Reader>::from(opened);
            *reader = ReaderState::Open((cursor, opened.clone()));
        }
        match reader.deref_mut() {
            ReaderState::Open((cursor, reader)) => {
                let (new_cursor, data) = func(*cursor, reader.clone()).await?;
                *cursor = new_cursor;
                Ok(data)
            }
            ReaderState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
                location!(),
            )),
            _ => unreachable!(),
        }
    }

    /// Read the entire blob file from the current cursor position
    /// to the end of the file
    ///
    /// After this call the cursor will be pointing to the end of
    /// the file.
    pub async fn read(&self) -> Result<bytes::Bytes> {
        let position = self.position;
        let size = self.size;
        self.do_with_reader(|cursor, reader| async move {
            let start = position as usize + cursor as usize;
            let end = (position + size) as usize;
            Ok((end as u64, reader.get_range(start..end).await?))
        })
        .await
    }

    /// Read up to `len` bytes from the current cursor position
    ///
    /// After this call the cursor will be pointing to the end of
    /// the read data.
    pub async fn read_up_to(&self, len: usize) -> Result<bytes::Bytes> {
        let position = self.position;
        let size = self.size;
        self.do_with_reader(|cursor, reader| async move {
            let start = position as usize + cursor as usize;
            let read_size = len.min((size - cursor) as usize);
            let end = start + read_size;
            let data = reader.get_range(start..end).await?;
            Ok((end as u64 - position, data))
        })
        .await
    }

    /// Seek to a new cursor position in the file
    pub async fn seek(&self, new_cursor: u64) -> Result<()> {
        let mut reader = self.reader.lock().await;
        match reader.deref_mut() {
            ReaderState::Open((cursor, _)) => {
                *cursor = new_cursor;
                Ok(())
            }
            ReaderState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
                location!(),
            )),
            ReaderState::Uninitialized(cursor) => {
                *cursor = new_cursor;
                Ok(())
            }
        }
    }

    /// Return the current cursor position in the file
    pub async fn tell(&self) -> Result<u64> {
        let reader = self.reader.lock().await;
        match *reader {
            ReaderState::Open((cursor, _)) => Ok(cursor),
            ReaderState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
                location!(),
            )),
            ReaderState::Uninitialized(cursor) => Ok(cursor),
        }
    }

    /// Return the size of the blob file in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn data_path(&self) -> &Path {
        &self.path
    }

    pub fn kind(&self) -> BlobKind {
        self.kind
    }

    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }
}

pub(super) async fn take_blobs(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    column: &str,
) -> Result<Vec<BlobFile>> {
    let projection = dataset.schema().project(&[column])?;
    let blob_field = &projection.fields[0];
    let blob_field_id = blob_field.id;
    if !projection.fields[0].is_blob() {
        return Err(Error::InvalidInput {
            location: location!(),
            source: format!("the column '{}' is not a blob column", column).into(),
        });
    }
    let description_and_addr = dataset
        .take_builder(row_ids, projection)?
        .with_row_address(true)
        .execute()
        .await?;
    let descriptions = description_and_addr.column(0).as_struct();
    let row_addrs = description_and_addr.column(1).as_primitive::<UInt64Type>();
    let blob_field_id = blob_field_id as u32;

    match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => collect_blob_files_v1(dataset, blob_field_id, descriptions, row_addrs),
        BlobVersion::V2 => {
            collect_blob_files_v2(dataset, blob_field_id, descriptions, row_addrs).await
        }
    }
}

/// Take [BlobFile] by row addresses.
///
/// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
/// Use this method when you already have row addresses, for example from
/// a scan with `with_row_address()`. For row IDs (stable identifiers), use
/// [`Dataset::take_blobs`]. For row indices (offsets), use
/// [`Dataset::take_blobs_by_indices`].
pub async fn take_blobs_by_addresses(
    dataset: &Arc<Dataset>,
    row_addrs: &[u64],
    column: &str,
) -> Result<Vec<BlobFile>> {
    let projection = dataset.schema().project(&[column])?;
    let blob_field = &projection.fields[0];
    let blob_field_id = blob_field.id;
    if !projection.fields[0].is_blob() {
        return Err(Error::InvalidInput {
            location: location!(),
            source: format!("the column '{}' is not a blob column", column).into(),
        });
    }

    // Convert Schema to ProjectionPlan
    let projection_request = ProjectionRequest::from(projection);
    let projection_plan = Arc::new(projection_request.into_projection_plan(dataset.clone())?);

    // Use try_new_from_addresses to bypass row ID index lookup.
    // This is critical when enable_stable_row_ids=true because row addresses
    // (fragment_id << 32 | row_offset) are different from row IDs (sequential integers).
    let description_and_addr =
        TakeBuilder::try_new_from_addresses(dataset.clone(), row_addrs.to_vec(), projection_plan)?
            .with_row_address(true)
            .execute()
            .await?;

    let descriptions = description_and_addr.column(0).as_struct();
    let row_addrs_result = description_and_addr.column(1).as_primitive::<UInt64Type>();
    let blob_field_id = blob_field_id as u32;

    match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => {
            collect_blob_files_v1(dataset, blob_field_id, descriptions, row_addrs_result)
        }
        BlobVersion::V2 => {
            collect_blob_files_v2(dataset, blob_field_id, descriptions, row_addrs_result).await
        }
    }
}

fn blob_version_from_descriptions(descriptions: &StructArray) -> Result<BlobVersion> {
    let fields = descriptions.fields();
    if fields.len() == 2 && fields[0].name() == "position" && fields[1].name() == "size" {
        return Ok(BlobVersion::V1);
    }
    if fields.len() == 5
        && fields[0].name() == "kind"
        && fields[1].name() == "position"
        && fields[2].name() == "size"
        && fields[3].name() == "blob_id"
        && fields[4].name() == "blob_uri"
    {
        return Ok(BlobVersion::V2);
    }
    Err(Error::InvalidInput {
        source: format!(
            "Unrecognized blob descriptions schema: expected v1 (position,size) or v2 (kind,position,size,blob_id,blob_uri) but got {:?}",
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
        )
        .into(),
        location: location!(),
    })
}

fn collect_blob_files_v1(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<BlobFile>> {
    let positions = descriptions.column(0).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(1).as_primitive::<UInt64Type>();

    Ok(row_addrs
        .values()
        .iter()
        .zip(positions.iter())
        .zip(sizes.iter())
        .filter_map(|((row_addr, position), size)| {
            let position = position?;
            let size = size?;
            Some((*row_addr, position, size))
        })
        .map(|(row_addr, position, size)| {
            BlobFile::new_inline(dataset.clone(), blob_field_id, row_addr, position, size)
        })
        .collect())
}

async fn collect_blob_files_v2(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<BlobFile>> {
    let kinds = descriptions.column(0).as_primitive::<UInt8Type>();
    let positions = descriptions.column(1).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(2).as_primitive::<UInt64Type>();
    let blob_ids = descriptions.column(3).as_primitive::<UInt32Type>();
    let blob_uris = descriptions.column(4).as_string::<i32>();

    let mut files = Vec::with_capacity(row_addrs.len());
    for (idx, row_addr) in row_addrs.values().iter().enumerate() {
        let kind = BlobKind::try_from(kinds.value(idx))?;

        // Struct is non-nullable; null rows are encoded as inline with zero position/size and empty uri
        if matches!(kind, BlobKind::Inline) && positions.value(idx) == 0 && sizes.value(idx) == 0 {
            continue;
        }

        match kind {
            BlobKind::Inline => {
                let position = positions.value(idx);
                let size = sizes.value(idx);
                files.push(BlobFile::new_inline(
                    dataset.clone(),
                    blob_field_id,
                    *row_addr,
                    position,
                    size,
                ));
            }
            BlobKind::Dedicated => {
                let blob_id = blob_ids.value(idx);
                let size = sizes.value(idx);
                let frag_id = RowAddress::from(*row_addr).fragment_id();
                let frag =
                    dataset
                        .get_fragment(frag_id as usize)
                        .ok_or_else(|| Error::Internal {
                            message: "Fragment not found".to_string(),
                            location: location!(),
                        })?;
                let data_file =
                    frag.data_file_for_field(blob_field_id)
                        .ok_or_else(|| Error::Internal {
                            message: "Data file not found for blob field".to_string(),
                            location: location!(),
                        })?;

                let data_file_key = data_file_key_from_path(data_file.path.as_str());
                let path = blob_path(&dataset.data_dir(), data_file_key, blob_id);
                files.push(BlobFile::new_dedicated(dataset.clone(), path, size));
            }
            BlobKind::Packed => {
                let blob_id = blob_ids.value(idx);
                let size = sizes.value(idx);
                let position = positions.value(idx);
                let frag_id = RowAddress::from(*row_addr).fragment_id();
                let frag =
                    dataset
                        .get_fragment(frag_id as usize)
                        .ok_or_else(|| Error::Internal {
                            message: "Fragment not found".to_string(),
                            location: location!(),
                        })?;
                let data_file =
                    frag.data_file_for_field(blob_field_id)
                        .ok_or_else(|| Error::Internal {
                            message: "Data file not found for blob field".to_string(),
                            location: location!(),
                        })?;
                let data_file_key = data_file_key_from_path(data_file.path.as_str());
                let path = blob_path(&dataset.data_dir(), data_file_key, blob_id);
                files.push(BlobFile::new_packed(dataset.clone(), path, position, size));
            }
            BlobKind::External => {
                let uri = blob_uris.value(idx).to_string();
                let position = positions.value(idx);
                let size = sizes.value(idx);
                let registry = dataset.session.store_registry();
                let params = dataset
                    .store_params
                    .as_ref()
                    .map(|p| Arc::new((**p).clone()))
                    .unwrap_or_else(|| Arc::new(ObjectStoreParams::default()));
                files.push(BlobFile::new_external(uri, position, size, registry, params).await?);
            }
        }
    }

    Ok(files)
}

fn data_file_key_from_path(path: &str) -> &str {
    let filename = path.rsplit('/').next().unwrap_or(path);
    filename.strip_suffix(".lance").unwrap_or(filename)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{array::AsArray, datatypes::UInt64Type};
    use arrow_array::RecordBatch;
    use arrow_array::{RecordBatchIterator, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use lance_arrow::{DataTypeExt, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY};
    use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
    use lance_io::stream::RecordBatchStream;

    use lance_core::{utils::tempfile::TempStrDir, Error, Result};
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_file::version::LanceFileVersion;

    use super::data_file_key_from_path;
    use crate::{
        blob::{blob_field, BlobArrayBuilder},
        dataset::WriteParams,
        utils::test::TestDatasetGenerator,
        Dataset,
    };

    struct BlobTestFixture {
        _test_dir: TempStrDir,
        dataset: Arc<Dataset>,
        data: Vec<RecordBatch>,
    }

    impl BlobTestFixture {
        async fn new() -> Self {
            let test_dir = TempStrDir::default();

            let data = lance_datagen::gen_batch()
                .col("filterme", array::step::<UInt64Type>())
                .col("blobs", array::blob())
                .into_reader_rows(RowCount::from(10), BatchCount::from(10))
                .map(|batch| Ok(batch?))
                .collect::<Result<Vec<_>>>()
                .unwrap();

            let dataset = Arc::new(
                TestDatasetGenerator::new(data.clone(), LanceFileVersion::default())
                    .make_hostile(&test_dir)
                    .await,
            );

            Self {
                _test_dir: test_dir,
                dataset,
                data,
            }
        }
    }

    #[tokio::test]
    pub async fn test_take_blobs() {
        let fixture = BlobTestFixture::new().await;

        let row_ids = fixture
            .dataset
            .scan()
            .project::<String>(&[])
            .unwrap()
            .filter("filterme >= 50")
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().values();
        let row_ids = vec![row_ids[5], row_ids[17], row_ids[33]];

        let blobs = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

        for (actual_idx, (expected_batch_idx, expected_row_idx)) in
            [(5, 5), (6, 7), (8, 3)].iter().enumerate()
        {
            let val = blobs[actual_idx].read().await.unwrap();
            let expected = fixture.data[*expected_batch_idx]
                .column(1)
                .as_binary::<i64>()
                .value(*expected_row_idx);

            assert_eq!(&val, expected);
        }
    }

    #[tokio::test]
    pub async fn test_take_blobs_by_indices() {
        let fixture = BlobTestFixture::new().await;

        let fragments = fixture.dataset.fragments();
        assert!(fragments.len() >= 2);
        let mut indices = Vec::with_capacity(fragments.len());
        let mut last = 2;

        for frag in fragments.iter() {
            indices.push(last as u64);
            last += frag.num_rows().unwrap_or(0);
        }
        indices.pop();

        // Row indices
        assert_eq!(indices, [2, 12, 22, 32, 42, 52, 62, 72, 82]);
        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&indices, "blobs")
            .await
            .unwrap();

        // Row IDs
        let row_ids = fragments
            .iter()
            .map(|frag| (frag.id << 32) + 2)
            .collect::<Vec<_>>();
        let blobs2 = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

        for (blob1, blob2) in blobs.iter().zip(blobs2.iter()) {
            assert_eq!(blob1.position(), blob2.position());
            assert_eq!(blob1.size(), blob2.size());
            assert_eq!(blob1.data_path(), blob2.data_path());
        }
    }

    #[tokio::test]
    pub async fn test_take_blob_id_not_exist() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.take_blobs(&[1000], "blobs").await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
    }

    #[tokio::test]
    pub async fn test_take_blob_not_blob_col() {
        let fixture = BlobTestFixture::new().await;

        let err = fixture.dataset.take_blobs(&[0], "filterme").await;

        assert!(matches!(err, Err(Error::InvalidInput { .. })));
        assert!(err.unwrap_err().to_string().contains("not a blob column"));
    }

    #[tokio::test]
    pub async fn test_scan_blobs() {
        let fixture = BlobTestFixture::new().await;

        // By default, scanning a blob column will load descriptions
        let batches = fixture
            .dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        let schema = batches.schema();

        assert!(schema.fields[0].data_type().is_struct());

        let batches = batches.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 10);
        for batch in batches.iter() {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column(0).data_type().is_struct());
        }

        // Should also be able to scan with filter
        let batches = fixture
            .dataset
            .scan()
            .project(&["blobs"])
            .unwrap()
            .filter("filterme = 50")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        let schema = batches.schema();

        assert!(schema.fields[0].data_type().is_struct());

        let batches = batches.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 1);
        for batch in batches.iter() {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column(0).data_type().is_struct());
        }
    }

    /// Test that take_blobs_by_indices works correctly with enable_stable_row_ids=true.
    ///
    /// This is a regression test for a bug where take_blobs_by_indices would fail
    /// with "index out of bounds" for fragment 1+ when stable row IDs are enabled.
    /// The bug was caused by passing row addresses (from row_offsets_to_row_addresses)
    /// to blob::take_blobs which expected row IDs. When stable row IDs are enabled,
    /// row addresses (fragment_id << 32 | offset) are different from row IDs
    /// (sequential integers), causing the row ID index lookup to fail for fragment 1+.
    #[tokio::test]
    pub async fn test_take_blobs_by_indices_with_stable_row_ids() {
        use crate::dataset::WriteParams;
        use arrow_array::RecordBatchIterator;

        let test_dir = TempStrDir::default();

        // Create test data with blob column
        let data = lance_datagen::gen_batch()
            .col("filterme", array::step::<UInt64Type>())
            .col("blobs", array::blob())
            .into_reader_rows(RowCount::from(6), BatchCount::from(1))
            .map(|batch| Ok(batch.unwrap()))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Write with enable_stable_row_ids=true and force multiple fragments
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 3, // Force 2 fragments with 3 rows each
            ..Default::default()
        };

        let reader = RecordBatchIterator::new(data.clone().into_iter().map(Ok), data[0].schema());
        let dataset = Arc::new(
            Dataset::write(reader, &test_dir, Some(write_params))
                .await
                .unwrap(),
        );

        // Verify we have multiple fragments
        let fragments = dataset.fragments();
        assert!(
            fragments.len() >= 2,
            "Expected at least 2 fragments, got {}",
            fragments.len()
        );

        // Test first fragment (indices 0, 1, 2) - this always worked
        let blobs = dataset
            .take_blobs_by_indices(&[0, 1, 2], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 3, "First fragment blobs should have 3 items");

        // Verify we can read the blob content
        for blob in &blobs {
            let content = blob.read().await.unwrap();
            assert!(!content.is_empty(), "Blob content should not be empty");
        }

        // Test second fragment (indices 3, 4, 5) - this was failing before the fix
        let blobs = dataset
            .take_blobs_by_indices(&[3, 4, 5], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 3, "Second fragment blobs should have 3 items");

        // Verify we can read the blob content from second fragment
        for blob in &blobs {
            let content = blob.read().await.unwrap();
            assert!(!content.is_empty(), "Blob content should not be empty");
        }

        // Test mixed indices from both fragments
        let blobs = dataset
            .take_blobs_by_indices(&[1, 4], "blobs")
            .await
            .unwrap();
        assert_eq!(blobs.len(), 2, "Mixed fragment blobs should have 2 items");
    }

    #[test]
    fn test_data_file_key_from_path() {
        assert_eq!(data_file_key_from_path("data/abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("nested/path/xyz"), "xyz");
    }

    #[tokio::test]
    async fn test_write_and_take_blobs_with_blob_array_builder() {
        let test_dir = TempStrDir::default();

        // Build a blob column with the new BlobArrayBuilder
        let mut blob_builder = BlobArrayBuilder::new(2);
        blob_builder.push_bytes(b"hello").unwrap();
        blob_builder.push_bytes(b"world").unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let id_array: arrow_array::ArrayRef = Arc::new(UInt32Array::from(vec![0, 1]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let dataset = Arc::new(
            Dataset::write(reader, &test_dir, Some(params))
                .await
                .unwrap(),
        );

        let blobs = dataset
            .take_blobs_by_indices(&[0, 1], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 2);
        let first = blobs[0].read().await.unwrap();
        let second = blobs[1].read().await.unwrap();
        assert_eq!(first.as_ref(), b"hello");
        assert_eq!(second.as_ref(), b"world");
    }

    #[tokio::test]
    async fn test_blob_v2_requires_v2_2() {
        let test_dir = TempStrDir::default();

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(b"hello").unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let id_array: arrow_array::ArrayRef = Arc::new(UInt32Array::from(vec![0]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        let result = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await;

        assert!(
            result.is_err(),
            "Blob v2 should be rejected for file version 2.1"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Blob v2 requires file version >= 2.2"));
    }

    async fn preprocess_kind_with_schema_metadata(metadata_value: &str, data_len: usize) -> u8 {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://blob_preprocessor",
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        let object_store = object_store.as_ref().clone();
        let data_dir = base_path.child("data");

        let mut field = blob_field("blob", true);
        let mut metadata = field.metadata().clone();
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            metadata_value.to_string(),
        );
        field = field.with_metadata(metadata);

        let writer_arrow_schema = Schema::new(vec![field.clone()]);
        let writer_schema = lance_core::datatypes::Schema::try_from(&writer_arrow_schema).unwrap();

        let mut preprocessor = super::BlobPreprocessor::new(
            object_store.clone(),
            data_dir,
            "data_file_key".to_string(),
            &writer_schema,
        );

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_bytes(vec![0u8; data_len]).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

        let field_without_metadata =
            Field::new("blob", field.data_type().clone(), field.is_nullable());
        let batch_schema = Arc::new(Schema::new(vec![field_without_metadata]));
        let batch = RecordBatch::try_new(batch_schema, vec![blob_array]).unwrap();

        let out = preprocessor.preprocess_batch(&batch).await.unwrap();
        let struct_arr = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(0)
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_ignores_non_positive_metadata() {
        let kind = preprocess_kind_with_schema_metadata("0", 256 * 1024).await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_smaller_metadata() {
        let kind = preprocess_kind_with_schema_metadata("131072", 256 * 1024).await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Dedicated as u8);
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_threshold_respects_larger_metadata() {
        let kind =
            preprocess_kind_with_schema_metadata("8388608", super::DEDICATED_THRESHOLD + 1024)
                .await;
        assert_eq!(kind, lance_core::datatypes::BlobKind::Packed as u8);
    }
}
