// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact Lance file-version implementations.

use bytes::Bytes;
use lance_core::{Error, Result, datatypes::Schema};
use lance_encoding::encoder::EncodedBatch;
use lance_io::traits::Writer;

use crate::{
    version::ConcreteFileVersion,
    writer::{FileWriter, FileWriterOptions},
};

pub mod v1;
pub mod v2_0;
pub mod v2_1;
pub mod v2_2;
pub mod v2_3;

/// Create a current-format writer for an exact file version.
///
/// V1 uses [`v1::writer::FileWriter`] directly because its manifest provider is
/// part of the writer type.
pub fn create_writer(
    version: ConcreteFileVersion,
    object_writer: Box<dyn Writer>,
    schema: Schema,
    options: FileWriterOptions,
) -> Result<FileWriter> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported(
            "Lance v1 files must be created with versions::v1::writer::FileWriter".to_string(),
        )),
        ConcreteFileVersion::V2_0 => {
            v2_0::create_writer(object_writer, schema, options).map(Into::into)
        }
        ConcreteFileVersion::V2_1 => {
            v2_1::create_writer(object_writer, schema, options).map(Into::into)
        }
        ConcreteFileVersion::V2_2 => {
            v2_2::create_writer(object_writer, schema, options).map(Into::into)
        }
        ConcreteFileVersion::V2_3 => {
            v2_3::create_writer(object_writer, schema, options).map(Into::into)
        }
    }
}

/// Create a writer that sizes V2.0 pages from logical array slices.
///
/// This runtime policy is used for input that has already been rechunked into
/// bounded zero-copy slices. Other versions use their standard writer.
#[doc(hidden)]
pub fn create_writer_with_logical_page_sizing(
    version: ConcreteFileVersion,
    object_writer: Box<dyn Writer>,
    schema: Schema,
    options: FileWriterOptions,
) -> Result<FileWriter> {
    match version {
        ConcreteFileVersion::V2_0 => {
            v2_0::create_writer_with_logical_page_sizing(object_writer, schema, options)
                .map(Into::into)
        }
        _ => create_writer(version, object_writer, schema, options),
    }
}

/// Create a lazy current-format writer for an exact file version.
pub fn create_lazy_writer(
    version: ConcreteFileVersion,
    object_writer: Box<dyn Writer>,
    options: FileWriterOptions,
) -> Result<FileWriter> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported(
            "legacy v1 files require an explicit schema and manifest provider".to_string(),
        )),
        ConcreteFileVersion::V2_0 => Ok(v2_0::create_lazy_writer(object_writer, options).into()),
        ConcreteFileVersion::V2_1 => Ok(v2_1::create_lazy_writer(object_writer, options).into()),
        ConcreteFileVersion::V2_2 => Ok(v2_2::create_lazy_writer(object_writer, options).into()),
        ConcreteFileVersion::V2_3 => Ok(v2_3::create_lazy_writer(object_writer, options).into()),
    }
}

/// Encode a self-described batch for an exact file version.
pub fn encode_self_described_batch(
    version: ConcreteFileVersion,
    batch: &EncodedBatch,
) -> Result<Bytes> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported(
            "Lance v1 does not support self-described current-format batches".to_string(),
        )),
        ConcreteFileVersion::V2_0 => v2_0::encode_self_described_batch(batch),
        ConcreteFileVersion::V2_1 => v2_1::encode_self_described_batch(batch),
        ConcreteFileVersion::V2_2 => v2_2::encode_self_described_batch(batch),
        ConcreteFileVersion::V2_3 => v2_3::encode_self_described_batch(batch),
    }
}

/// Encode a mini-lance batch for an exact file version.
pub fn encode_mini_batch(version: ConcreteFileVersion, batch: &EncodedBatch) -> Result<Bytes> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported(
            "Lance v1 does not support mini-lance current-format batches".to_string(),
        )),
        ConcreteFileVersion::V2_0 => v2_0::encode_mini_batch(batch),
        ConcreteFileVersion::V2_1 => v2_1::encode_mini_batch(batch),
        ConcreteFileVersion::V2_2 => v2_2::encode_mini_batch(batch),
        ConcreteFileVersion::V2_3 => v2_3::encode_mini_batch(batch),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn v1_rejects_current_format_embedded_batches() {
        let batch = EncodedBatch {
            data: Bytes::new(),
            page_table: Vec::new(),
            schema: Arc::new(Schema::default()),
            top_level_columns: Vec::new(),
            num_rows: 0,
        };

        assert!(matches!(
            encode_self_described_batch(ConcreteFileVersion::V1, &batch),
            Err(Error::NotSupported { .. })
        ));
        assert!(matches!(
            encode_mini_batch(ConcreteFileVersion::V1, &batch),
            Err(Error::NotSupported { .. })
        ));
    }
}
