// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance v2.0 encoding composition.

use std::sync::Arc;

use bytes::Bytes;
use lance_core::{Result, datatypes::Schema};
use lance_encoding::{
    array_encoding::ArrayFieldEncodingStrategy,
    encoder::{EncodedBatch, FieldEncodingStrategy},
};
use lance_io::traits::Writer as ObjectWriter;

use crate::writer::FileWriterOptions;

mod writer;

pub use writer::Writer;

/// Compose the v2.0 field encoding mechanisms.
pub fn encoding_strategy() -> Arc<dyn FieldEncodingStrategy> {
    Arc::new(ArrayFieldEncodingStrategy::new())
}

/// Create a v2.0 writer with an explicit schema.
pub fn create_writer(
    object_writer: Box<dyn ObjectWriter>,
    schema: Schema,
    options: FileWriterOptions,
) -> Result<Writer> {
    Writer::try_new(object_writer, schema, options)
}

/// Create a v2.0 writer whose schema is inferred from the first batch.
pub fn create_lazy_writer(
    object_writer: Box<dyn ObjectWriter>,
    options: FileWriterOptions,
) -> Writer {
    Writer::new_lazy(object_writer, options)
}

/// Encode a self-described v2.0 batch.
pub fn encode_self_described_batch(batch: &EncodedBatch) -> Result<Bytes> {
    writer::concat_lance_footer(batch, true)
}

/// Encode a mini-lance v2.0 batch.
pub fn encode_mini_batch(batch: &EncodedBatch) -> Result<Bytes> {
    writer::concat_lance_footer(batch, false)
}
