// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance v2.1 encoding composition.

use std::sync::Arc;

use lance_core::{Error, Result, datatypes::Field};
use lance_encoding::{
    compression_config::CompressionParams,
    encoder::{
        ColumnIndexSequence, FieldEncoder, FieldEncodingContext, FieldEncodingStrategy,
        structural::{
            PrimitiveFieldEncoding, PrimitivePageEncoding, try_create_binary_blob, try_create_list,
            try_create_struct,
        },
    },
};

mod compression;

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
        if let Some(encoder) = self.primitive.try_create(field, column_index, context)? {
            return Ok(encoder);
        }
        if matches!(
            field.data_type(),
            arrow_schema::DataType::FixedSizeList(item, _)
                if matches!(item.data_type(), arrow_schema::DataType::Struct(_))
        ) {
            return Err(Error::not_supported_source(
                "FixedSizeList<Struct> is not enabled by the selected file format".into(),
            ));
        }
        if matches!(field.data_type(), arrow_schema::DataType::Map(_, _)) {
            return Err(Error::not_supported_source(
                "Map data type is not enabled by the selected file format".into(),
            ));
        }
        if let Some(encoder) = try_create_list(field, column_index, context)? {
            return Ok(encoder);
        }
        if let Some(encoder) = try_create_struct(field, column_index, context)? {
            return Ok(encoder);
        }
        Err(Error::not_supported_source(
            format!(
                "Lance v2.1 has no field encoding for '{}' with data type {}",
                field.name,
                field.data_type()
            )
            .into(),
        ))
    }
}

/// Compose the v2.1 field encoding mechanisms.
pub fn encoding_strategy(params: CompressionParams) -> Arc<dyn FieldEncodingStrategy> {
    let compression = Arc::new(compression::Strategy::new(params));
    Arc::new(FieldStrategy {
        primitive: PrimitiveFieldEncoding::new([
            PrimitivePageEncoding::reject_sparse(),
            PrimitivePageEncoding::dense_u16(compression),
        ]),
    })
}
