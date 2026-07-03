// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builders and file-level writer helpers for Lance blob v2 columns.
//!
//! Logical blob input uses `Struct<data: LargeBinary?, uri: Utf8?>`. File-level blob
//! descriptors use a physical writer-side struct with `kind`, `blob_id`, and range fields.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, Ordering},
};

use arrow_array::{
    Array, ArrayRef, StructArray,
    builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder},
    cast::AsArray,
    types::{UInt8Type, UInt32Type, UInt64Type},
};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, Fields};
use bytes::Bytes;
use lance_arrow::{
    ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
    BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME, FieldExt,
};
use lance_core::{
    datatypes::{BlobKind, Field as LanceField, Schema as LanceSchema},
    utils::blob::blob_path,
};
use lance_io::{
    object_store::ObjectStore,
    traits::{Reader, WriteExt, Writer},
};
use object_store::path::Path;
use tokio::io::AsyncWriteExt;

use crate::{Error, Result};

/// Construct the Arrow field for a blob v2 column.
///
/// Blob v2 expects a column shaped as `Struct<data: LargeBinary?, uri: Utf8?>` and
/// tagged with `ARROW:extension:name = "lance.blob.v2"`.
pub fn blob_field(name: &str, nullable: bool) -> Field {
    blob_field_with_options(name, nullable, BlobFieldOptions::default())
}

/// Options for constructing a blob v2 field.
#[derive(Clone, Debug, Default)]
pub struct BlobFieldOptions {
    /// Maximum payload size to keep inline in the data file before using packed blob storage.
    pub inline_size_threshold: Option<usize>,
    /// Maximum payload size to store in packed blob storage before using dedicated blob storage.
    ///
    /// A zero threshold is invalid because dedicated blob storage is selected when
    /// the payload size is greater than this value.
    pub dedicated_size_threshold: Option<NonZeroUsize>,
}

impl BlobFieldOptions {
    /// Set the maximum payload size to keep inline in the data file.
    pub fn with_inline_size_threshold(mut self, threshold: usize) -> Self {
        self.inline_size_threshold = Some(threshold);
        self
    }

    /// Set the maximum payload size to store in packed blob storage.
    pub fn with_dedicated_size_threshold(mut self, threshold: NonZeroUsize) -> Self {
        self.dedicated_size_threshold = Some(threshold);
        self
    }
}

/// Construct the Arrow field for a blob v2 column with storage layout options.
///
/// Blob v2 expects a column shaped as `Struct<data: LargeBinary?, uri: Utf8?>` and
/// tagged with `ARROW:extension:name = "lance.blob.v2"`.
///
/// ```
/// # use lance::{BlobFieldOptions, blob_field_with_options};
/// let field = blob_field_with_options(
///     "blob",
///     true,
///     BlobFieldOptions::default().with_inline_size_threshold(16 * 1024),
/// );
/// assert_eq!(
///     field
///         .metadata()
///         .get("lance-encoding:blob-inline-size-threshold")
///         .map(String::as_str),
///     Some("16384"),
/// );
/// ```
pub fn blob_field_with_options(name: &str, nullable: bool, options: BlobFieldOptions) -> Field {
    let mut metadata = [(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    if let Some(threshold) = options.inline_size_threshold {
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            threshold.to_string(),
        );
    }
    if let Some(threshold) = options.dedicated_size_threshold {
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            threshold.get().to_string(),
        );
    }
    Field::new(
        name,
        DataType::Struct(
            vec![
                Field::new("data", DataType::LargeBinary, true),
                Field::new("uri", DataType::Utf8, true),
            ]
            .into(),
        ),
        nullable,
    )
    .with_metadata(metadata)
}

fn prepared_blob_child_fields() -> Fields {
    Fields::from(vec![
        Field::new("kind", DataType::UInt8, true),
        Field::new("data", DataType::LargeBinary, true),
        Field::new("uri", DataType::Utf8, true),
        Field::new("blob_id", DataType::UInt32, true),
        Field::new("blob_size", DataType::UInt64, true),
        Field::new("position", DataType::UInt64, true),
    ])
}

fn prepared_blob_field_with_metadata(
    name: &str,
    nullable: bool,
    metadata: HashMap<String, String>,
) -> Field {
    Field::new(
        name,
        DataType::Struct(prepared_blob_child_fields()),
        nullable,
    )
    .with_metadata(metadata)
}

fn logical_blob_lance_children() -> Result<Vec<LanceField>> {
    [
        Field::new("data", DataType::LargeBinary, true),
        Field::new("uri", DataType::Utf8, true),
    ]
    .iter()
    .map(LanceField::try_from)
    .collect()
}

fn field_matches(field: &Field, name: &str, data_type: &DataType, nullable: bool) -> bool {
    field.name() == name && field.data_type() == data_type && field.is_nullable() == nullable
}

fn blob_v2_shape_error(field: &Field) -> Error {
    Error::invalid_input(format!(
        "Blob v2 field '{}' must use either logical struct<data: LargeBinary?, uri: Utf8?> \
         with optional position/size UInt64 fields or prepared struct<kind: UInt8?, data: \
         LargeBinary?, uri: Utf8?, blob_id: UInt32?, blob_size: UInt64?, position: UInt64?>",
        field.name()
    ))
}

/// Returns true when `field` is the writer-side prepared blob v2 struct.
pub(crate) fn is_prepared_blob_v2_field(field: &Field) -> bool {
    if !field.is_blob_v2() {
        return false;
    }
    let DataType::Struct(fields) = field.data_type() else {
        return false;
    };
    let expected = prepared_blob_child_fields();
    fields.len() == expected.len()
        && fields
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual.as_ref() == expected.as_ref())
}

/// Returns true when `field` is the logical blob v2 input struct.
pub(crate) fn is_logical_blob_v2_field(field: &Field) -> bool {
    if !field.is_blob_v2() {
        return false;
    }
    let DataType::Struct(fields) = field.data_type() else {
        return false;
    };
    match fields.len() {
        2 => {
            field_matches(fields[0].as_ref(), "data", &DataType::LargeBinary, true)
                && field_matches(fields[1].as_ref(), "uri", &DataType::Utf8, true)
        }
        4 => {
            field_matches(fields[0].as_ref(), "data", &DataType::LargeBinary, true)
                && field_matches(fields[1].as_ref(), "uri", &DataType::Utf8, true)
                && fields[2].name() == "position"
                && fields[2].data_type() == &DataType::UInt64
                && fields[3].name() == "size"
                && fields[3].data_type() == &DataType::UInt64
        }
        _ => false,
    }
}

fn normalize_prepared_blob_lance_field(field: &LanceField) -> Result<LanceField> {
    if field.is_blob_v2() {
        let arrow_field = Field::from(field);
        if is_prepared_blob_v2_field(&arrow_field) {
            let mut normalized = field.clone();
            let mut logical_children = logical_blob_lance_children()?;
            for (logical_child, prepared_child) in
                logical_children.iter_mut().zip(field.children.iter())
            {
                logical_child.id = prepared_child.id;
                logical_child.parent_id = field.id;
            }
            normalized.children = logical_children;
            return Ok(normalized);
        }
        if is_logical_blob_v2_field(&arrow_field) {
            return Ok(field.clone());
        }
        return Err(blob_v2_shape_error(&arrow_field));
    }

    if field.children.is_empty() {
        return Ok(field.clone());
    }

    let normalized_children = field
        .children
        .iter()
        .map(normalize_prepared_blob_lance_field)
        .collect::<Result<Vec<_>>>()?;

    Ok(LanceField {
        children: normalized_children,
        ..field.clone()
    })
}

pub(crate) fn normalize_prepared_blob_schema(schema: &LanceSchema) -> Result<LanceSchema> {
    let fields = schema
        .fields
        .iter()
        .map(normalize_prepared_blob_lance_field)
        .collect::<Result<Vec<_>>>()?;
    Ok(LanceSchema {
        fields,
        metadata: schema.metadata.clone(),
    })
}

#[derive(Clone, Debug)]
pub(crate) struct BlobIdAllocator {
    inner: Arc<BlobIdAllocatorInner>,
}

#[derive(Debug)]
struct BlobIdAllocatorInner {
    next: AtomicU32,
    used: Mutex<HashSet<u32>>,
}

impl BlobIdAllocator {
    pub(crate) fn new(start: u32) -> Self {
        Self {
            inner: Arc::new(BlobIdAllocatorInner {
                next: AtomicU32::new(start),
                used: Mutex::new(HashSet::new()),
            }),
        }
    }

    pub(crate) fn next(&self) -> Result<u32> {
        loop {
            let id = self.inner.next.load(Ordering::Relaxed);
            if id == u32::MAX {
                return Err(Error::invalid_input(
                    "Blob id allocator exhausted u32 id space",
                ));
            }
            if self
                .inner
                .next
                .compare_exchange(id, id + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let mut used =
                self.inner.used.lock().map_err(|_| {
                    Error::internal("Blob id allocator mutex was poisoned".to_string())
                })?;
            if used.insert(id) {
                return Ok(id);
            }
        }
    }
}

fn validate_blob_id(blob_id: u32) -> Result<()> {
    if blob_id == 0 {
        return Err(Error::invalid_input("Blob id 0 is reserved"));
    }
    Ok(())
}

fn validate_range(offset: u64, size: u64, object_size: u64, label: &str) -> Result<()> {
    let end = offset.checked_add(size).ok_or_else(|| {
        Error::invalid_input(format!(
            "{label} range overflows u64: offset={offset}, size={size}"
        ))
    })?;
    if end > object_size {
        return Err(Error::invalid_input(format!(
            "{label} range [{offset}, {end}) exceeds blob object size {object_size}"
        )));
    }
    Ok(())
}

fn validate_prepared_blob_value_array(field: &Field, array: &ArrayRef) -> Result<()> {
    if !is_prepared_blob_v2_field(field) {
        return Err(blob_v2_shape_error(field));
    }

    let struct_arr = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::invalid_input("Prepared blob column was not a struct array"))?;
    let kind_col = struct_arr
        .column_by_name("kind")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `kind` field"))?
        .as_primitive::<UInt8Type>();
    let data_col = struct_arr
        .column_by_name("data")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `data` field"))?
        .as_binary::<i64>();
    let uri_col = struct_arr
        .column_by_name("uri")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `uri` field"))?
        .as_string::<i32>();
    let blob_id_col = struct_arr
        .column_by_name("blob_id")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `blob_id` field"))?
        .as_primitive::<UInt32Type>();
    let blob_size_col = struct_arr
        .column_by_name("blob_size")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `blob_size` field"))?
        .as_primitive::<UInt64Type>();
    let position_col = struct_arr
        .column_by_name("position")
        .ok_or_else(|| Error::invalid_input("Prepared blob struct missing `position` field"))?
        .as_primitive::<UInt64Type>();

    for row in 0..struct_arr.len() {
        if struct_arr.is_null(row) {
            continue;
        }
        if kind_col.is_null(row) {
            return Err(Error::invalid_input(format!(
                "Prepared blob row {row} is non-null but `kind` is null"
            )));
        }

        match BlobKind::try_from(kind_col.value(row))? {
            BlobKind::Inline => {
                if data_col.is_null(row) {
                    return Err(Error::invalid_input(format!(
                        "Prepared inline blob row {row} must set `data`"
                    )));
                }
            }
            BlobKind::Packed => {
                if blob_id_col.is_null(row)
                    || blob_size_col.is_null(row)
                    || position_col.is_null(row)
                {
                    return Err(Error::invalid_input(format!(
                        "Prepared packed blob row {row} must set `blob_id`, `blob_size`, and `position`"
                    )));
                }
                validate_blob_id(blob_id_col.value(row))?;
                let offset = position_col.value(row);
                let size = blob_size_col.value(row);
                offset.checked_add(size).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Prepared packed blob row {row} range overflows u64: offset={offset}, size={size}"
                    ))
                })?;
            }
            BlobKind::Dedicated => {
                if blob_id_col.is_null(row) || blob_size_col.is_null(row) {
                    return Err(Error::invalid_input(format!(
                        "Prepared dedicated blob row {row} must set `blob_id` and `blob_size`"
                    )));
                }
                validate_blob_id(blob_id_col.value(row))?;
            }
            BlobKind::External => {
                if uri_col.is_null(row) || uri_col.value(row).is_empty() {
                    return Err(Error::invalid_input(format!(
                        "Prepared external blob row {row} must set a non-empty `uri`"
                    )));
                }
                let offset = if position_col.is_null(row) {
                    0
                } else {
                    position_col.value(row)
                };
                let size = if blob_size_col.is_null(row) {
                    0
                } else {
                    blob_size_col.value(row)
                };
                offset.checked_add(size).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Prepared external blob row {row} range overflows u64: offset={offset}, size={size}"
                    ))
                })?;
            }
        }
    }

    Ok(())
}

/// Validate a writer-side prepared blob v2 array before it reaches the encoder.
pub(crate) fn validate_prepared_blob_array(field: &Field, array: &ArrayRef) -> Result<()> {
    validate_prepared_blob_value_array(field, array)
}

fn sidecar_path_for_data_file(data_file_path: &Path, blob_id: u32) -> Result<Path> {
    validate_blob_id(blob_id)?;
    let file_name = data_file_path.filename().ok_or_else(|| {
        Error::invalid_input("Data file path must include a file name".to_string())
    })?;
    let data_file_key = file_name.strip_suffix(".lance").ok_or_else(|| {
        Error::invalid_input(format!(
            "Data file path '{}' must end with '.lance'",
            data_file_path
        ))
    })?;
    let data_dir = data_file_path.parent().unwrap_or_default();
    Ok(blob_path(&data_dir, data_file_key, blob_id))
}

/// Byte range inside a packed or external blob object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobRange {
    /// Byte offset relative to the beginning of the blob object.
    pub offset: u64,
    /// Number of bytes in this range.
    pub size: u64,
}

/// A physical blob descriptor row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlobDescriptor {
    /// A null blob row.
    Null,
    /// Payload bytes embedded into the data file by the blob encoder.
    Inline { data: Bytes },
    /// Payload bytes stored as a range in a packed sidecar blob.
    Packed {
        blob_id: u32,
        offset: u64,
        size: u64,
    },
    /// Payload bytes stored as the full contents of a dedicated sidecar blob.
    Dedicated { blob_id: u32, size: u64 },
    /// Payload bytes referenced from an external object or registered base.
    External {
        base_id: u32,
        uri: String,
        offset: u64,
        size: u64,
    },
}

/// A physical blob descriptor column ready to be included in a [`RecordBatch`](arrow_array::RecordBatch).
pub struct BlobDescriptorColumn {
    field: Field,
    array: ArrayRef,
}

impl BlobDescriptorColumn {
    /// Return the Arrow field for the descriptor column.
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Return the Arrow array for the descriptor column.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Consume this column into `(field, array)` parts.
    pub fn into_parts(self) -> (Field, ArrayRef) {
        (self.field, self.array)
    }
}

/// Builds physical blob descriptors for one blob v2 column.
///
/// This builder only produces the writer-side descriptor struct array. It does not allocate blob ids,
/// choose sidecar paths, write blob objects, or commit data files.
pub struct BlobDescriptorArrayBuilder {
    field: Field,
    values: Vec<BlobDescriptor>,
}

impl BlobDescriptorArrayBuilder {
    /// Create a descriptor array builder for one blob column.
    pub fn new(column: impl Into<String>) -> Self {
        let mut metadata = HashMap::with_capacity(1);
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        Self::new_with_metadata(column, true, metadata)
    }

    pub(crate) fn new_with_metadata(
        column: impl Into<String>,
        nullable: bool,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            field: prepared_blob_field_with_metadata(&column.into(), nullable, metadata),
            values: Vec::new(),
        }
    }

    /// Append one packed blob descriptor.
    pub fn push_packed(&mut self, blob_id: u32, range: BlobRange) -> Result<()> {
        self.push(BlobDescriptor::Packed {
            blob_id,
            offset: range.offset,
            size: range.size,
        })
    }

    /// Append multiple packed blob descriptors for the same blob object.
    pub fn extend_packed(
        &mut self,
        blob_id: u32,
        ranges: impl IntoIterator<Item = BlobRange>,
    ) -> Result<()> {
        for range in ranges {
            self.push_packed(blob_id, range)?;
        }
        Ok(())
    }

    /// Append one dedicated blob descriptor.
    pub fn push_dedicated(&mut self, blob_id: u32, size: u64) -> Result<()> {
        self.push(BlobDescriptor::Dedicated { blob_id, size })
    }

    /// Append one blob descriptor to this column.
    pub fn push(&mut self, value: BlobDescriptor) -> Result<()> {
        validate_blob_descriptor(&value)?;
        self.values.push(value);
        Ok(())
    }

    /// Append multiple blob descriptors to this column.
    pub fn extend(&mut self, values: impl IntoIterator<Item = BlobDescriptor>) -> Result<()> {
        for value in values {
            self.push(value)?;
        }
        Ok(())
    }

    /// Append an inline blob value.
    pub fn push_inline(&mut self, data: impl Into<Bytes>) -> Result<()> {
        self.push(BlobDescriptor::Inline { data: data.into() })
    }

    /// Append an external blob reference.
    pub fn push_external(
        &mut self,
        uri: impl Into<String>,
        range: Option<BlobRange>,
    ) -> Result<()> {
        let range = range.unwrap_or(BlobRange { offset: 0, size: 0 });
        self.push(BlobDescriptor::External {
            base_id: 0,
            uri: uri.into(),
            offset: range.offset,
            size: range.size,
        })
    }

    /// Append a null blob row.
    pub fn push_null(&mut self) -> Result<()> {
        self.push(BlobDescriptor::Null)
    }

    /// Return the descriptor Arrow field for this blob column.
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Finish this column and return the writer-side descriptor struct array.
    pub fn finish(self) -> Result<BlobDescriptorColumn> {
        let mut kind_builder = PrimitiveBuilder::<UInt8Type>::with_capacity(self.values.len());
        let mut data_builder = LargeBinaryBuilder::with_capacity(self.values.len(), 0);
        let mut uri_builder = StringBuilder::with_capacity(self.values.len(), 0);
        let mut blob_id_builder = PrimitiveBuilder::<UInt32Type>::with_capacity(self.values.len());
        let mut blob_size_builder =
            PrimitiveBuilder::<UInt64Type>::with_capacity(self.values.len());
        let mut position_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(self.values.len());
        let mut validity = NullBufferBuilder::new(self.values.len());

        for value in self.values {
            match value {
                BlobDescriptor::Null => {
                    validity.append_null();
                    kind_builder.append_null();
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    position_builder.append_null();
                }
                BlobDescriptor::Inline { data } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::Inline as u8);
                    data_builder.append_value(data.as_ref());
                    uri_builder.append_null();
                    blob_id_builder.append_null();
                    blob_size_builder.append_null();
                    position_builder.append_null();
                }
                BlobDescriptor::Packed {
                    blob_id,
                    offset,
                    size,
                } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::Packed as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(blob_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_value(offset);
                }
                BlobDescriptor::Dedicated { blob_id, size } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::Dedicated as u8);
                    data_builder.append_null();
                    uri_builder.append_null();
                    blob_id_builder.append_value(blob_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_null();
                }
                BlobDescriptor::External {
                    base_id,
                    uri,
                    offset,
                    size,
                } => {
                    validity.append_non_null();
                    kind_builder.append_value(BlobKind::External as u8);
                    data_builder.append_null();
                    uri_builder.append_value(uri);
                    blob_id_builder.append_value(base_id);
                    blob_size_builder.append_value(size);
                    position_builder.append_value(offset);
                }
            }
        }

        let array = Arc::new(StructArray::try_new(
            prepared_blob_child_fields(),
            vec![
                Arc::new(kind_builder.finish()),
                Arc::new(data_builder.finish()),
                Arc::new(uri_builder.finish()),
                Arc::new(blob_id_builder.finish()),
                Arc::new(blob_size_builder.finish()),
                Arc::new(position_builder.finish()),
            ],
            validity.finish(),
        )?) as ArrayRef;
        validate_prepared_blob_array(&self.field, &array)?;

        Ok(BlobDescriptorColumn {
            field: self.field,
            array,
        })
    }
}

fn validate_blob_descriptor(value: &BlobDescriptor) -> Result<()> {
    match value {
        BlobDescriptor::Null => Ok(()),
        BlobDescriptor::Inline { .. } => Ok(()),
        BlobDescriptor::Packed {
            blob_id,
            offset,
            size,
        } => {
            validate_blob_id(*blob_id)?;
            offset.checked_add(*size).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Packed blob range overflows u64: offset={offset}, size={size}"
                ))
            })?;
            Ok(())
        }
        BlobDescriptor::Dedicated { blob_id, .. } => validate_blob_id(*blob_id),
        BlobDescriptor::External {
            uri, offset, size, ..
        } => {
            if uri.is_empty() {
                return Err(Error::invalid_input("External blob URI cannot be empty"));
            }
            offset.checked_add(*size).ok_or_else(|| {
                Error::invalid_input(format!(
                    "External blob range overflows u64: offset={offset}, size={size}"
                ))
            })?;
            Ok(())
        }
    }
}

/// Writes a Lance-owned packed sidecar blob for one data file and returns descriptors.
pub struct PackedBlobWriter {
    object_store: ObjectStore,
    path: Path,
    blob_id: u32,
    writer: Box<dyn Writer>,
    offset: u64,
    values: Vec<BlobDescriptor>,
}

impl PackedBlobWriter {
    /// Create a packed blob writer for `data_file_path` and `blob_id`.
    pub async fn try_new(
        object_store: ObjectStore,
        data_file_path: Path,
        blob_id: u32,
    ) -> Result<Self> {
        let path = sidecar_path_for_data_file(&data_file_path, blob_id)?;
        let writer = object_store.create(&path).await?;
        Ok(Self {
            object_store,
            path,
            blob_id,
            writer,
            offset: 0,
            values: Vec::new(),
        })
    }

    /// Return the blob id for this packed sidecar.
    pub fn blob_id(&self) -> u32 {
        self.blob_id
    }

    /// Return the derived sidecar path for this packed blob.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one logical blob payload to the packed sidecar.
    pub async fn write_blob(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        let bytes = bytes.as_ref();
        self.write_blob_bytes(bytes).await?;
        Ok(())
    }

    pub(crate) async fn write_blob_bytes(&mut self, bytes: &[u8]) -> Result<BlobDescriptor> {
        let size = bytes.len() as u64;
        let offset = self.offset;
        self.writer.write_all(bytes).await?;
        self.record_written_blob(offset, size)
    }

    pub(crate) async fn write_blob_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<BlobDescriptor> {
        let size = range.len() as u64;
        let offset = self.offset;
        self.writer.copy_range_from_reader(reader, range).await?;
        self.record_written_blob(offset, size)
    }

    fn record_written_blob(&mut self, offset: u64, size: u64) -> Result<BlobDescriptor> {
        self.offset = self.offset.checked_add(size).ok_or_else(|| {
            Error::invalid_input(format!(
                "Packed blob writer offset overflowed: offset={offset}, size={size}"
            ))
        })?;
        let value = BlobDescriptor::Packed {
            blob_id: self.blob_id,
            offset,
            size,
        };
        self.values.push(value.clone());
        Ok(value)
    }

    /// Finish the packed sidecar and return descriptors in write order.
    pub async fn finish(mut self) -> Result<Vec<BlobDescriptor>> {
        Writer::shutdown(self.writer.as_mut()).await?;
        let object_size = self.object_store.size(&self.path).await?;
        validate_range(0, self.offset, object_size, "Packed blob")?;
        Ok(self.values)
    }
}

/// Writes a Lance-owned dedicated sidecar blob for one data file and returns its descriptor.
pub struct DedicatedBlobWriter {
    object_store: ObjectStore,
    path: Path,
    blob_id: u32,
    writer: Box<dyn Writer>,
    size: u64,
}

impl DedicatedBlobWriter {
    /// Create a dedicated blob writer for `data_file_path` and `blob_id`.
    pub async fn try_new(
        object_store: ObjectStore,
        data_file_path: Path,
        blob_id: u32,
    ) -> Result<Self> {
        let path = sidecar_path_for_data_file(&data_file_path, blob_id)?;
        let writer = object_store.create(&path).await?;
        Ok(Self {
            object_store,
            path,
            blob_id,
            writer,
            size: 0,
        })
    }

    /// Return the blob id for this dedicated sidecar.
    pub fn blob_id(&self) -> u32 {
        self.blob_id
    }

    /// Return the derived sidecar path for this dedicated blob.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append bytes to the dedicated sidecar.
    pub async fn write(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        let bytes = bytes.as_ref();
        let size = bytes.len() as u64;
        self.writer.write_all(bytes).await?;
        self.record_written_bytes(size)
    }

    pub(crate) async fn write_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<()> {
        let size = range.len() as u64;
        self.writer.copy_range_from_reader(reader, range).await?;
        self.record_written_bytes(size)
    }

    fn record_written_bytes(&mut self, size: u64) -> Result<()> {
        self.size = self.size.checked_add(size).ok_or_else(|| {
            Error::invalid_input(format!(
                "Dedicated blob writer size overflowed: current={}, append={size}",
                self.size
            ))
        })?;
        Ok(())
    }

    /// Finish the dedicated sidecar and return its descriptor.
    pub async fn finish(mut self) -> Result<BlobDescriptor> {
        Writer::shutdown(self.writer.as_mut()).await?;
        let object_size = self.object_store.size(&self.path).await?;
        if object_size != self.size {
            return Err(Error::io(format!(
                "Dedicated blob sidecar '{}' has size {}, expected {}",
                self.path, object_size, self.size
            )));
        }
        Ok(BlobDescriptor::Dedicated {
            blob_id: self.blob_id,
            size: self.size,
        })
    }
}

/// Builder for blob v2 input struct columns.
///
/// The builder enforces that each row contains exactly one of `data` or `uri` (or is null).
pub struct BlobArrayBuilder {
    data_builder: LargeBinaryBuilder,
    uri_builder: StringBuilder,
    validity: NullBufferBuilder,
    expected_len: usize,
    len: usize,
}

impl BlobArrayBuilder {
    /// Create a new builder with the given row capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data_builder: LargeBinaryBuilder::with_capacity(capacity, 0),
            uri_builder: StringBuilder::with_capacity(capacity, 0),
            validity: NullBufferBuilder::new(capacity),
            expected_len: capacity,
            len: 0,
        }
    }

    /// Append a blob backed by raw bytes.
    pub fn push_bytes(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_non_null();
        self.data_builder.append_value(bytes);
        self.uri_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Append a blob referenced by URI.
    pub fn push_uri(&mut self, uri: impl Into<String>) -> Result<()> {
        self.ensure_capacity()?;
        let uri = uri.into();
        if uri.is_empty() {
            return Err(Error::invalid_input("URI cannot be empty"));
        }
        self.validity.append_non_null();
        self.data_builder.append_null();
        self.uri_builder.append_value(uri);
        self.len += 1;
        Ok(())
    }

    /// Append an empty blob (inline, zero-length payload).
    pub fn push_empty(&mut self) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_non_null();
        self.data_builder.append_value([]);
        self.uri_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Append a null row.
    pub fn push_null(&mut self) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_null();
        self.data_builder.append_null();
        self.uri_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Finish building and return an Arrow struct array.
    pub fn finish(mut self) -> Result<ArrayRef> {
        if self.len != self.expected_len {
            return Err(Error::invalid_input(format!(
                "Expected {} rows but received {}",
                self.expected_len, self.len
            )));
        }

        let data = Arc::new(self.data_builder.finish());
        let uri = Arc::new(self.uri_builder.finish());
        let validity = self.validity.finish();

        let struct_array = StructArray::try_new(
            vec![
                Field::new("data", DataType::LargeBinary, true),
                Field::new("uri", DataType::Utf8, true),
            ]
            .into(),
            vec![data as ArrayRef, uri as ArrayRef],
            validity,
        )?;

        Ok(Arc::new(struct_array))
    }

    fn ensure_capacity(&self) -> Result<()> {
        if self.len >= self.expected_len {
            Err(Error::invalid_input("BlobArrayBuilder capacity exceeded"))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, StringArray};
    use arrow_schema::Schema as ArrowSchema;
    use lance_core::utils::tempfile::TempDir;

    #[test]
    fn test_field_metadata() {
        let field = blob_field("blob", true);
        assert!(field.metadata().get(ARROW_EXT_NAME_KEY).is_some());
        assert_eq!(
            field.metadata().get(ARROW_EXT_NAME_KEY).unwrap(),
            BLOB_V2_EXT_NAME
        );
    }

    #[test]
    fn test_field_metadata_with_options() {
        let field = blob_field_with_options(
            "blob",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(16 * 1024)
                .with_dedicated_size_threshold(NonZeroUsize::new(2 * 1024 * 1024).unwrap()),
        );
        assert_eq!(
            field
                .metadata()
                .get(BLOB_INLINE_SIZE_THRESHOLD_META_KEY)
                .unwrap(),
            "16384"
        );
        assert_eq!(
            field
                .metadata()
                .get(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY)
                .unwrap(),
            "2097152"
        );
    }

    #[test]
    fn test_builder_basic() {
        let mut b = BlobArrayBuilder::new(4);
        b.push_bytes(b"hi").unwrap();
        b.push_uri("s3://bucket/key").unwrap();
        b.push_empty().unwrap();
        b.push_null().unwrap();

        let arr = b.finish().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.null_count(), 1);

        let struct_arr = arr.as_struct();
        let data = struct_arr.column(0).as_binary::<i64>();
        let uri = struct_arr.column(1).as_string::<i32>();

        assert_eq!(data.value(0), b"hi");
        assert!(uri.is_null(0));
        assert!(data.is_null(1));
        assert_eq!(uri.value(1), "s3://bucket/key");
        assert_eq!(data.value(2).len(), 0);
        assert!(uri.is_null(2));
    }

    #[test]
    fn test_capacity_error() {
        let mut b = BlobArrayBuilder::new(1);
        b.push_bytes(b"a").unwrap();
        let err = b.push_bytes(b"b").unwrap_err();
        assert!(err.to_string().contains("capacity exceeded"));
    }

    #[test]
    fn test_empty_uri_rejected() {
        let mut b = BlobArrayBuilder::new(1);
        let err = b.push_uri("").unwrap_err();
        assert!(err.to_string().contains("URI cannot be empty"));
    }

    #[test]
    fn test_prepared_blob_column_finish() {
        let mut writer = BlobDescriptorArrayBuilder::new("blob");
        writer.push_inline(Bytes::from_static(b"hello")).unwrap();
        writer
            .push_packed(7, BlobRange { offset: 3, size: 5 })
            .unwrap();
        writer.push_dedicated(8, 9).unwrap();
        writer.push_null().unwrap();

        let column = writer.finish().unwrap();
        assert!(is_prepared_blob_v2_field(column.field()));
        let struct_arr = column.array().as_struct();
        let kinds = struct_arr
            .column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>();
        let blob_ids = struct_arr
            .column_by_name("blob_id")
            .unwrap()
            .as_primitive::<UInt32Type>();
        let sizes = struct_arr
            .column_by_name("blob_size")
            .unwrap()
            .as_primitive::<UInt64Type>();
        let positions = struct_arr
            .column_by_name("position")
            .unwrap()
            .as_primitive::<UInt64Type>();

        assert_eq!(kinds.value(0), BlobKind::Inline as u8);
        assert_eq!(kinds.value(1), BlobKind::Packed as u8);
        assert_eq!(blob_ids.value(1), 7);
        assert_eq!(positions.value(1), 3);
        assert_eq!(sizes.value(1), 5);
        assert_eq!(kinds.value(2), BlobKind::Dedicated as u8);
        assert_eq!(blob_ids.value(2), 8);
        assert_eq!(sizes.value(2), 9);
        assert!(struct_arr.is_null(3));
    }

    #[test]
    fn test_prepared_blob_array_rejects_range_overflow() {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        let field = prepared_blob_field_with_metadata("blob", true, metadata);

        for (kind, uri) in [
            (BlobKind::Packed, None),
            (BlobKind::External, Some("file:///external.bin")),
        ] {
            let array = Arc::new(
                StructArray::try_new(
                    prepared_blob_child_fields(),
                    vec![
                        Arc::new(arrow_array::UInt8Array::from(vec![kind as u8])) as ArrayRef,
                        Arc::new(arrow_array::LargeBinaryArray::from_iter([None::<&[u8]>])),
                        Arc::new(arrow_array::StringArray::from_iter([uri])),
                        Arc::new(arrow_array::UInt32Array::from(vec![1])),
                        Arc::new(arrow_array::UInt64Array::from(vec![4])),
                        Arc::new(arrow_array::UInt64Array::from(vec![u64::MAX - 1])),
                    ],
                    None,
                )
                .unwrap(),
            ) as ArrayRef;

            let err = validate_prepared_blob_array(&field, &array).unwrap_err();
            assert!(err.to_string().contains("range overflows u64"));
        }
    }

    #[test]
    fn test_normalize_prepared_blob_schema_preserves_non_blob_fields() {
        let mut metadata = HashMap::new();
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string());
        let prepared_field = prepared_blob_field_with_metadata("blob", true, metadata);
        let dict_field = Field::new(
            "dict",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        );
        let mut schema =
            LanceSchema::try_from(&ArrowSchema::new(vec![dict_field, prepared_field])).unwrap();
        schema.fields[0].id = 42;
        schema.fields[1].id = 7;

        let dictionary_values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        schema.fields[0].set_dictionary_values(&dictionary_values);

        let normalized = normalize_prepared_blob_schema(&schema).unwrap();

        assert_eq!(normalized.fields[0].id, 42);
        assert_eq!(
            normalized.fields[0]
                .dictionary
                .as_ref()
                .and_then(|dict| dict.values.as_ref())
                .map(|values| values.len()),
            Some(2)
        );
        assert_eq!(normalized.fields[1].id, 7);
        assert_eq!(normalized.fields[1].children.len(), 2);
        assert_eq!(normalized.fields[1].children[0].name, "data");
        assert_eq!(normalized.fields[1].children[1].name, "uri");
        assert!(normalized.fields[1].children[0].id >= 0);
        assert!(normalized.fields[1].children[1].id >= 0);
    }

    #[tokio::test]
    async fn test_sidecar_writers_return_prepared_values() {
        let temp_dir = TempDir::default();
        let data_dir = Path::from_absolute_path(temp_dir.std_path().join("data")).unwrap();
        let data_file_key = "data-file".to_string();
        let data_file_path = data_dir.clone().join(format!("{data_file_key}.lance"));
        let object_store = ObjectStore::local();

        let packed_id = 7;
        let packed_path = blob_path(&data_dir, &data_file_key, packed_id);
        let mut packed =
            PackedBlobWriter::try_new(object_store.clone(), data_file_path.clone(), packed_id)
                .await
                .unwrap();
        assert_eq!(packed.path(), &packed_path);
        packed.write_blob(b"abc").await.unwrap();
        packed.write_blob(b"de").await.unwrap();
        let packed_values = packed.finish().await.unwrap();
        assert_eq!(
            packed_values,
            vec![
                BlobDescriptor::Packed {
                    blob_id: packed_id,
                    offset: 0,
                    size: 3,
                },
                BlobDescriptor::Packed {
                    blob_id: packed_id,
                    offset: 3,
                    size: 2,
                },
            ]
        );

        let dedicated_id = 8;
        let dedicated_path = blob_path(&data_dir, &data_file_key, dedicated_id);
        let mut dedicated =
            DedicatedBlobWriter::try_new(object_store.clone(), data_file_path, dedicated_id)
                .await
                .unwrap();
        assert_eq!(dedicated.path(), &dedicated_path);
        dedicated.write(b"abcdef").await.unwrap();
        assert_eq!(
            dedicated.finish().await.unwrap(),
            BlobDescriptor::Dedicated {
                blob_id: dedicated_id,
                size: 6,
            }
        );

        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        builder
            .extend_packed(
                42,
                vec![
                    BlobRange { offset: 1, size: 2 },
                    BlobRange { offset: 4, size: 1 },
                ],
            )
            .unwrap();
        builder.push_dedicated(43, 3).unwrap();
        let column = builder.finish().unwrap();
        assert_eq!(column.array().len(), 3);
    }
}
