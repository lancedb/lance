// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::DataType;
use lance_encoding::encoder::{
    ColumnIndexSequence, CoreFieldEncodingStrategy, FieldEncodingStrategy,
};
use zone::ZoneMapsFieldEncoder;

pub mod format;
pub mod zone;

/// Wraps the core encoding strategy and adds the encoders from this
/// crate
#[derive(Debug)]
pub struct LanceDfFieldEncodingStrategy {
    core: CoreFieldEncodingStrategy,
    rows_per_map: u32,
}

impl FieldEncodingStrategy for LanceDfFieldEncodingStrategy {
    fn create_field_encoder(
        &self,
        encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &lance_core::datatypes::Field,
        column_index: &mut ColumnIndexSequence,
        cache_bytes_per_column: u64,
        keep_original_array: bool,
        config: &std::collections::HashMap<String, String>,
    ) -> lance_core::Result<Box<dyn lance_encoding::encoder::FieldEncoder>> {
        let data_type = field.data_type();
        if data_type.is_primitive()
            || matches!(
                data_type,
                DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
            )
        {
            let inner_encoder = self.core.create_field_encoder(
                // Don't collect stats on inner string fields
                &self.core,
                field,
                column_index,
                cache_bytes_per_column,
                keep_original_array,
                config,
            )?;
            Ok(Box::new(ZoneMapsFieldEncoder::try_new(
                inner_encoder,
                data_type.clone(),
                self.rows_per_map,
            )?))
        } else {
            self.core.create_field_encoder(
                encoding_strategy_root,
                field,
                column_index,
                cache_bytes_per_column,
                keep_original_array,
                config,
            )
        }
    }
}
