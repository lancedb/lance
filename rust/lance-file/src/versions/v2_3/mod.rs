// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance v2.3 encoding composition.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use bytes::Bytes;
use lance_core::{
    Error, Result,
    datatypes::{Field, Schema},
};
use lance_encoding::{
    compression_config::CompressionParams,
    encoder::{
        ColumnIndexSequence, EncodedBatch, FieldEncoder, FieldEncodingContext,
        FieldEncodingStrategy,
        structural::{
            PrimitiveFieldEncoding, PrimitivePageEncoding, try_create_binary_blob, try_create_list,
            try_create_map, try_create_struct, try_create_structural_blob,
            try_create_structural_fixed_size_list,
        },
    },
};
use lance_io::traits::Writer as ObjectWriter;

use crate::writer::FileWriterOptions;

mod compression;
mod writer;

pub use writer::Writer;

static WARNED_ON_UNSTABLE_FORMAT: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
struct FieldStrategy {
    primitive: PrimitiveFieldEncoding,
}

impl FieldEncodingStrategy for FieldStrategy {
    fn create_field_encoder(
        &self,
        field: &Field,
        column_index: &mut ColumnIndexSequence,
        context: &FieldEncodingContext<'_>,
    ) -> Result<Box<dyn FieldEncoder>> {
        if let Some(encoder) =
            try_create_binary_blob(&self.primitive, field, column_index, context)?
        {
            return Ok(encoder);
        }
        if let Some(encoder) =
            try_create_structural_blob(&self.primitive, field, column_index, context)?
        {
            return Ok(encoder);
        }
        if field.is_blob() {
            return Err(Error::invalid_input_source(
                format!(
                    "Blob encoding is not available for field '{}' with data type {}",
                    field.name,
                    field.data_type()
                )
                .into(),
            ));
        }
        if let Some(encoder) = try_create_map(field, column_index, context)? {
            return Ok(encoder);
        }
        if let Some(encoder) = try_create_structural_fixed_size_list(field, column_index, context)?
        {
            return Ok(encoder);
        }
        if let Some(encoder) = self.primitive.try_create(field, column_index, context)? {
            return Ok(encoder);
        }
        if let Some(encoder) = try_create_list(field, column_index, context)? {
            return Ok(encoder);
        }
        if let Some(encoder) = try_create_struct(field, column_index, context)? {
            return Ok(encoder);
        }
        Err(Error::not_supported_source(
            format!(
                "Lance v2.3 has no field encoding for '{}' with data type {}",
                field.name,
                field.data_type()
            )
            .into(),
        ))
    }
}

/// Compose the v2.3 field encoding mechanisms.
pub fn encoding_strategy(params: CompressionParams) -> Arc<dyn FieldEncodingStrategy> {
    encoding_strategy_with_policy(params, false)
}

pub(crate) fn encoding_strategy_with_shared_dictionary_sizing(
    params: CompressionParams,
) -> Arc<dyn FieldEncodingStrategy> {
    encoding_strategy_with_policy(params, true)
}

fn encoding_strategy_with_policy(
    params: CompressionParams,
    use_shared_dictionary_sizing: bool,
) -> Arc<dyn FieldEncodingStrategy> {
    let compression = Arc::new(compression::Strategy::new(params));
    let primitive = PrimitiveFieldEncoding::new([
        PrimitivePageEncoding::sparse(compression.clone()),
        PrimitivePageEncoding::constant(),
        PrimitivePageEncoding::dense_u32(compression),
    ]);
    let primitive = if use_shared_dictionary_sizing {
        primitive.with_shared_dictionary_sizing()
    } else {
        primitive
    };
    Arc::new(FieldStrategy { primitive })
}

fn warn_unstable_format() {
    if WARNED_ON_UNSTABLE_FORMAT
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        log::warn!(
            "You have requested an unstable format version.  Files written with this format version may not be readable in the future!  This is a development feature and should only be used for experimentation and never for production data."
        );
    }
}

/// Create a v2.3 writer with an explicit schema.
pub fn create_writer(
    object_writer: Box<dyn ObjectWriter>,
    schema: Schema,
    options: FileWriterOptions,
) -> Result<Writer> {
    warn_unstable_format();
    Writer::try_new(object_writer, schema, options)
}

pub(crate) fn create_writer_with_shared_dictionary_sizing(
    object_writer: Box<dyn ObjectWriter>,
    schema: Schema,
    options: FileWriterOptions,
) -> Result<Writer> {
    Writer::try_new_with_shared_dictionary_sizing(object_writer, schema, options)
}

/// Create a v2.3 writer with explicit compression tuning.
pub fn create_writer_with_compression(
    object_writer: Box<dyn ObjectWriter>,
    schema: Schema,
    options: FileWriterOptions,
    compression: CompressionParams,
) -> Result<Writer> {
    warn_unstable_format();
    Writer::try_new_with_compression(object_writer, schema, options, compression)
}

/// Create a v2.3 writer whose schema is inferred from the first batch.
pub fn create_lazy_writer(
    object_writer: Box<dyn ObjectWriter>,
    options: FileWriterOptions,
) -> Writer {
    warn_unstable_format();
    Writer::new_lazy(object_writer, options)
}

/// Create a lazy v2.3 writer with explicit compression tuning.
pub fn create_lazy_writer_with_compression(
    object_writer: Box<dyn ObjectWriter>,
    options: FileWriterOptions,
    compression: CompressionParams,
) -> Writer {
    warn_unstable_format();
    Writer::new_lazy_with_compression(object_writer, options, compression)
}

/// Encode a self-described v2.3 batch.
pub fn encode_self_described_batch(batch: &EncodedBatch) -> Result<Bytes> {
    writer::concat_lance_footer(batch, true)
}

/// Encode a mini-lance v2.3 batch.
pub fn encode_mini_batch(batch: &EncodedBatch) -> Result<Bytes> {
    writer::concat_lance_footer(batch, false)
}
