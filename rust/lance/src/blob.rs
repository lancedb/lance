// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Convenience builders for Lance blob v2 input columns.
//!
//! Blob v2 expects a column shaped as `Struct<data: LargeBinary?, uri: Utf8?>` and
//! tagged with `ARROW:extension:name = "lance.blob.v2"`. This module offers a
//! type-safe builder to construct that struct without manually wiring metadata

use std::sync::Arc;

use arrow_array::{builder::LargeBinaryBuilder, builder::StringBuilder, ArrayRef, StructArray};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};

use crate::{Error, Result};

/// Construct the Arrow field for a blob v2 column.
///
/// Blob v2 expects a column shaped as `Struct<data: LargeBinary?, uri: Utf8?>` and
/// tagged with `ARROW:extension:name = "lance.blob.v2"`.
pub fn blob_field(name: &str, nullable: bool) -> Field {
    let metadata = [(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]
        .into_iter()
        .collect();
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
            return Err(Error::invalid_input(
                "URI cannot be empty",
                snafu::location!(),
            ));
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
            return Err(Error::invalid_input(
                format!(
                    "Expected {} rows but received {}",
                    self.expected_len, self.len
                ),
                snafu::location!(),
            ));
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
            Err(Error::invalid_input(
                "BlobArrayBuilder capacity exceeded",
                snafu::location!(),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::Array;

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
}
