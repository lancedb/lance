// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Convenience builders for Lance blob v2 input columns.
//!
//! Blob v2 expects a struct column tagged with
//! `ARROW:extension:name = "lance.blob.v2"`. Child fields are recognized by name.
//! This module offers a type-safe builder to construct that struct without
//! manually wiring metadata.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, StructArray,
    builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder},
    types::UInt64Type,
};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field};
use lance_arrow::{
    ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
    BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME,
};

use crate::{Error, Result};

/// Construct the Arrow field for a blob v2 column.
///
/// The default Rust schema preserves the historical minimal shape:
/// `Struct<data: LargeBinary?, uri: Utf8?>`.
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
/// Blob v2 expects a struct column tagged with
/// `ARROW:extension:name = "lance.blob.v2"`. Child fields are recognized by name.
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
    blob_field_with_children(name, nullable, false, false, options)
}

fn blob_field_with_children(
    name: &str,
    nullable: bool,
    include_position_size: bool,
    include_source_id: bool,
    options: BlobFieldOptions,
) -> Field {
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
    let mut fields = vec![
        Field::new("data", DataType::LargeBinary, true),
        Field::new("uri", DataType::Utf8, true),
    ];
    if include_position_size {
        fields.push(Field::new("position", DataType::UInt64, true));
        fields.push(Field::new("size", DataType::UInt64, true));
    }
    if include_source_id {
        fields.push(Field::new("source_id", DataType::Utf8, true));
    }

    Field::new(name, DataType::Struct(fields.into()), nullable).with_metadata(metadata)
}

/// Builder for blob v2 input struct columns.
///
/// The builder enforces that each row contains exactly one of `data` or `uri` (or is null).
pub struct BlobArrayBuilder {
    data_builder: LargeBinaryBuilder,
    uri_builder: StringBuilder,
    position_builder: PrimitiveBuilder<UInt64Type>,
    size_builder: PrimitiveBuilder<UInt64Type>,
    source_id_builder: StringBuilder,
    validity: NullBufferBuilder,
    expected_len: usize,
    len: usize,
    has_position_size: bool,
    has_source_id: bool,
}

impl BlobArrayBuilder {
    /// Create a new builder with the given row capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data_builder: LargeBinaryBuilder::with_capacity(capacity, 0),
            uri_builder: StringBuilder::with_capacity(capacity, 0),
            position_builder: PrimitiveBuilder::<UInt64Type>::with_capacity(capacity),
            size_builder: PrimitiveBuilder::<UInt64Type>::with_capacity(capacity),
            source_id_builder: StringBuilder::with_capacity(capacity, 0),
            validity: NullBufferBuilder::new(capacity),
            expected_len: capacity,
            len: 0,
            has_position_size: false,
            has_source_id: false,
        }
    }

    /// Construct an Arrow field matching the shape this builder will produce.
    ///
    /// Use this instead of [`blob_field`] when constructing arrays with source
    /// identity or URI slice metadata.
    pub fn field(&self, name: &str, nullable: bool) -> Field {
        self.field_with_options(name, nullable, BlobFieldOptions::default())
    }

    /// Construct an Arrow field matching the shape this builder will produce,
    /// including storage layout options.
    pub fn field_with_options(
        &self,
        name: &str,
        nullable: bool,
        options: BlobFieldOptions,
    ) -> Field {
        blob_field_with_children(
            name,
            nullable,
            self.has_position_size || self.has_source_id,
            self.has_source_id,
            options,
        )
    }

    /// Append a blob backed by raw bytes.
    pub fn push_bytes(&mut self, bytes: impl AsRef<[u8]>) -> Result<()> {
        self.ensure_capacity()?;
        self.append_bytes(bytes.as_ref(), None)
    }

    /// Append a blob backed by raw bytes and a user-provided source identity.
    ///
    /// Rows with the same `source_id` may share the same Lance-owned packed or
    /// dedicated descriptor within one data file. Inline blobs ignore `source_id`.
    pub fn push_bytes_with_source_id(
        &mut self,
        source_id: impl Into<String>,
        bytes: impl AsRef<[u8]>,
    ) -> Result<()> {
        self.ensure_capacity()?;
        let source_id = source_id.into();
        validate_source_id(&source_id)?;
        self.append_bytes(bytes.as_ref(), Some(&source_id))
    }

    /// Append a blob referenced by URI.
    pub fn push_uri(&mut self, uri: impl Into<String>) -> Result<()> {
        self.ensure_capacity()?;
        let uri = uri.into();
        self.append_uri(&uri, None, None, None)
    }

    /// Append a sliced blob referenced by URI.
    pub fn push_uri_with_slice(
        &mut self,
        uri: impl Into<String>,
        position: u64,
        size: u64,
    ) -> Result<()> {
        self.ensure_capacity()?;
        let uri = uri.into();
        self.append_uri(&uri, Some(position), Some(size), None)
    }

    /// Append a URI blob and a user-provided source identity.
    ///
    /// In ingest mode, rows with the same `source_id` may share the same
    /// Lance-owned packed or dedicated descriptor within one data file.
    pub fn push_uri_with_source_id(
        &mut self,
        source_id: impl Into<String>,
        uri: impl Into<String>,
        position: Option<u64>,
        size: Option<u64>,
    ) -> Result<()> {
        self.ensure_capacity()?;
        let source_id = source_id.into();
        validate_source_id(&source_id)?;
        let uri = uri.into();
        self.append_uri(&uri, position, size, Some(&source_id))
    }

    /// Append an empty blob (inline, zero-length payload).
    pub fn push_empty(&mut self) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_non_null();
        self.data_builder.append_value([]);
        self.uri_builder.append_null();
        self.position_builder.append_null();
        self.size_builder.append_null();
        self.source_id_builder.append_null();
        self.len += 1;
        Ok(())
    }

    /// Append a null row.
    pub fn push_null(&mut self) -> Result<()> {
        self.ensure_capacity()?;
        self.validity.append_null();
        self.data_builder.append_null();
        self.uri_builder.append_null();
        self.position_builder.append_null();
        self.size_builder.append_null();
        self.source_id_builder.append_null();
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
        let position = Arc::new(self.position_builder.finish());
        let size = Arc::new(self.size_builder.finish());
        let source_id = Arc::new(self.source_id_builder.finish());
        let validity = self.validity.finish();
        let include_position_size = self.has_position_size || self.has_source_id;

        let mut fields = vec![
            Field::new("data", DataType::LargeBinary, true),
            Field::new("uri", DataType::Utf8, true),
        ];
        let mut arrays = vec![data as ArrayRef, uri as ArrayRef];
        if include_position_size {
            fields.push(Field::new("position", DataType::UInt64, true));
            fields.push(Field::new("size", DataType::UInt64, true));
            arrays.push(position as ArrayRef);
            arrays.push(size as ArrayRef);
        }
        if self.has_source_id {
            fields.push(Field::new("source_id", DataType::Utf8, true));
            arrays.push(source_id as ArrayRef);
        }

        let struct_array = StructArray::try_new(fields.into(), arrays, validity)?;

        Ok(Arc::new(struct_array))
    }

    fn append_bytes(&mut self, bytes: &[u8], source_id: Option<&str>) -> Result<()> {
        self.validity.append_non_null();
        self.data_builder.append_value(bytes);
        self.uri_builder.append_null();
        self.position_builder.append_null();
        self.size_builder.append_null();
        match source_id {
            Some(source_id) => {
                self.has_source_id = true;
                self.source_id_builder.append_value(source_id);
            }
            None => self.source_id_builder.append_null(),
        }
        self.len += 1;
        Ok(())
    }

    fn append_uri(
        &mut self,
        uri: &str,
        position: Option<u64>,
        size: Option<u64>,
        source_id: Option<&str>,
    ) -> Result<()> {
        if uri.is_empty() {
            return Err(Error::invalid_input("URI cannot be empty"));
        }
        if position.is_some() != size.is_some() {
            return Err(Error::invalid_input(
                "URI blob must set both position and size, or neither",
            ));
        }
        self.validity.append_non_null();
        self.data_builder.append_null();
        self.uri_builder.append_value(uri);
        match position {
            Some(position) => {
                self.has_position_size = true;
                self.position_builder.append_value(position);
            }
            None => self.position_builder.append_null(),
        }
        match size {
            Some(size) => self.size_builder.append_value(size),
            None => self.size_builder.append_null(),
        }
        match source_id {
            Some(source_id) => {
                self.has_source_id = true;
                self.source_id_builder.append_value(source_id);
            }
            None => self.source_id_builder.append_null(),
        }
        self.len += 1;
        Ok(())
    }

    fn ensure_capacity(&self) -> Result<()> {
        if self.len >= self.expected_len {
            Err(Error::invalid_input("BlobArrayBuilder capacity exceeded"))
        } else {
            Ok(())
        }
    }
}

fn validate_source_id(source_id: &str) -> Result<()> {
    if source_id.is_empty() {
        Err(Error::invalid_input("source_id cannot be empty"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use arrow_array::Array;
    use arrow_array::cast::AsArray;

    #[test]
    fn test_field_metadata() {
        let field = blob_field("blob", true);
        assert!(field.metadata().get(ARROW_EXT_NAME_KEY).is_some());
        assert_eq!(
            field.metadata().get(ARROW_EXT_NAME_KEY).unwrap(),
            BLOB_V2_EXT_NAME
        );
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected struct blob field");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "data");
        assert_eq!(fields[1].name(), "uri");
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
        assert_eq!(struct_arr.columns().len(), 2);
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
    fn test_builder_source_id() {
        let mut b = BlobArrayBuilder::new(2);
        b.push_bytes_with_source_id("image:1", b"hi").unwrap();
        b.push_uri_with_source_id("image:2", "s3://bucket/key", Some(3), Some(4))
            .unwrap();

        let field = b.field("blob", true);
        let arr = b.finish().unwrap();
        let struct_arr = arr.as_struct();
        assert_eq!(struct_arr.columns().len(), 5);
        let uri = struct_arr.column(1).as_string::<i32>();
        let position = struct_arr.column(2).as_primitive::<UInt64Type>();
        let size = struct_arr.column(3).as_primitive::<UInt64Type>();
        let source_id = struct_arr.column(4).as_string::<i32>();

        assert_eq!(source_id.value(0), "image:1");
        assert_eq!(uri.value(1), "s3://bucket/key");
        assert_eq!(position.value(1), 3);
        assert_eq!(size.value(1), 4);
        assert_eq!(source_id.value(1), "image:2");

        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected struct blob field");
        };
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            vec!["data", "uri", "position", "size", "source_id"]
        );
    }

    #[test]
    fn test_builder_slice_shape() {
        let mut b = BlobArrayBuilder::new(1);
        b.push_uri_with_slice("s3://bucket/key", 3, 4).unwrap();

        let field = b.field("blob", true);
        let arr = b.finish().unwrap();
        let struct_arr = arr.as_struct();

        assert_eq!(struct_arr.columns().len(), 4);
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected struct blob field");
        };
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            vec!["data", "uri", "position", "size"]
        );
    }

    #[test]
    fn test_builder_field_with_options() {
        let mut b = BlobArrayBuilder::new(1);
        b.push_bytes_with_source_id("image:1", b"hi").unwrap();

        let field = b.field_with_options(
            "blob",
            true,
            BlobFieldOptions::default().with_inline_size_threshold(16 * 1024),
        );

        assert_eq!(
            field
                .metadata()
                .get(BLOB_INLINE_SIZE_THRESHOLD_META_KEY)
                .map(String::as_str),
            Some("16384")
        );
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected struct blob field");
        };
        assert_eq!(
            fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
            vec!["data", "uri", "position", "size", "source_id"]
        );
    }
}
