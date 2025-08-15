// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::{cast::AsArray, Array, ArrayRef, StructArray, UInt64Array};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field as ArrowField, Fields};
use futures::future::BoxFuture;
use lance_core::{datatypes::Field, Error, Result};
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    constants::PACKED_STRUCT_META_KEY,
    encoder::{EncodeTask, EncodedColumn, FieldEncoder, OutOfLineBuffers},
    encodings::logical::primitive::PrimitiveStructuralEncoder,
    repdef::RepDefBuilder,
};

/// Blob structural encoder - stores large binary data in external buffers
///
/// This encoder takes large binary arrays and stores them outside the normal
/// page structure. It creates a descriptor (position, size) for each blob
/// that is stored inline in the page.
pub struct BlobStructuralEncoder {
    // Encoder for the descriptors (position/size struct)
    descriptor_encoder: Box<dyn FieldEncoder>,
}

impl BlobStructuralEncoder {
    pub fn new(
        field: &Field,
        column_index: u32,
        options: &crate::encoder::EncodingOptions,
        compression_strategy: Arc<dyn crate::compression::CompressionStrategy>,
    ) -> Result<Self> {
        // Create descriptor field: struct<position: u64, size: u64>
        // Preserve the original field's metadata for packed struct
        let mut descriptor_metadata = HashMap::with_capacity(1);
        descriptor_metadata.insert(PACKED_STRUCT_META_KEY.to_string(), "true".to_string());

        let descriptor_data_type = DataType::Struct(Fields::from(vec![
            ArrowField::new("position", DataType::UInt64, false),
            ArrowField::new("size", DataType::UInt64, false),
        ]));

        // Use the original field's name for the descriptor
        let descriptor_field = Field::try_from(
            ArrowField::new(&field.name, descriptor_data_type, field.nullable)
                .with_metadata(descriptor_metadata),
        )?;

        // Use PrimitiveStructuralEncoder to handle the descriptor
        let descriptor_encoder = Box::new(PrimitiveStructuralEncoder::try_new(
            options,
            compression_strategy,
            column_index,
            descriptor_field,
            Arc::new(HashMap::new()),
        )?);

        Ok(Self { descriptor_encoder })
    }
}

impl FieldEncoder for BlobStructuralEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        // Convert input array to LargeBinary
        let binary_array = array
            .as_binary_opt::<i64>()
            .ok_or_else(|| Error::InvalidInput {
                source: format!("Expected LargeBinary array, got {}", array.data_type()).into(),
                location: location!(),
            })?;

        // Collect positions and sizes
        let mut positions = Vec::with_capacity(binary_array.len());
        let mut sizes = Vec::with_capacity(binary_array.len());

        for i in 0..binary_array.len() {
            if binary_array.is_null(i) {
                // Null values are handled in the structural layer
                // We just need placeholders here
                positions.push(0);
                sizes.push(0);
            } else {
                let value = binary_array.value(i);
                if value.is_empty() {
                    // Empty values
                    positions.push(0);
                    sizes.push(0);
                } else {
                    // Add data to external buffers
                    let position =
                        external_buffers.add_buffer(LanceBuffer::from(Buffer::from(value)));
                    positions.push(position);
                    sizes.push(value.len() as u64);
                }
            }
        }

        // Create descriptor array
        let position_array = Arc::new(UInt64Array::from(positions));
        let size_array = Arc::new(UInt64Array::from(sizes));
        let descriptor_array = Arc::new(StructArray::new(
            Fields::from(vec![
                ArrowField::new("position", DataType::UInt64, false),
                ArrowField::new("size", DataType::UInt64, false),
            ]),
            vec![position_array as ArrayRef, size_array as ArrayRef],
            binary_array.nulls().cloned(), // Pass through null buffer
        ));

        // Delegate to descriptor encoder
        self.descriptor_encoder.maybe_encode(
            descriptor_array,
            external_buffers,
            repdef,
            row_number,
            num_rows,
        )
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        self.descriptor_encoder.flush(external_buffers)
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<EncodedColumn>>> {
        self.descriptor_encoder.finish(external_buffers)
    }

    fn num_columns(&self) -> u32 {
        self.descriptor_encoder.num_columns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compression::DefaultCompressionStrategy,
        encoder::{ColumnIndexSequence, EncodingOptions},
        testing::{check_round_trip_encoding_of_data, TestCases},
    };
    use arrow_array::LargeBinaryArray;

    #[test]
    fn test_blob_encoder_creation() {
        let field =
            Field::try_from(ArrowField::new("blob_field", DataType::LargeBinary, true)).unwrap();
        let mut column_index = ColumnIndexSequence::default();
        let column_idx = column_index.next_column_index(0);
        let options = EncodingOptions::default();
        let compression = Arc::new(DefaultCompressionStrategy::new());

        let encoder = BlobStructuralEncoder::new(&field, column_idx, &options, compression);

        assert!(encoder.is_ok());
    }

    #[tokio::test]
    async fn test_blob_encoding_simple() {
        let field = Field::try_from(
            ArrowField::new("blob_field", DataType::LargeBinary, true).with_metadata(
                HashMap::from([(
                    lance_core::datatypes::BLOB_META_KEY.to_string(),
                    "true".to_string(),
                )]),
            ),
        )
        .unwrap();
        let mut column_index = ColumnIndexSequence::default();
        let column_idx = column_index.next_column_index(0);
        let options = EncodingOptions::default();
        let compression = Arc::new(DefaultCompressionStrategy::new());

        let mut encoder =
            BlobStructuralEncoder::new(&field, column_idx, &options, compression).unwrap();

        // Create test data with larger blobs
        let large_data = vec![0u8; 1024 * 100]; // 100KB blob
        let data: Vec<Option<&[u8]>> =
            vec![Some(b"hello world"), None, Some(&large_data), Some(b"")];
        let array = Arc::new(LargeBinaryArray::from(data));

        // Test encoding
        let mut external_buffers = OutOfLineBuffers::new(0, 8);
        let repdef = RepDefBuilder::default();

        let tasks = encoder
            .maybe_encode(array, &mut external_buffers, repdef, 0, 4)
            .unwrap();

        // If no tasks yet, flush to force encoding
        if tasks.is_empty() {
            let _flush_tasks = encoder.flush(&mut external_buffers).unwrap();
        }

        // Should produce encode tasks for the descriptor (or we need more data)
        // For now, just verify no errors occurred
        assert!(encoder.num_columns() > 0);

        // Verify external buffers were used for large data
        let buffers = external_buffers.take_buffers();
        assert!(
            !buffers.is_empty(),
            "Large blobs should be stored in external buffers"
        );
    }

    #[tokio::test]
    async fn test_blob_round_trip() {
        // Test round-trip encoding with blob metadata
        let blob_metadata = HashMap::from([(
            lance_core::datatypes::BLOB_META_KEY.to_string(),
            "true".to_string(),
        )]);

        // Create test data
        let val1: &[u8] = &vec![1u8; 1024]; // 1KB
        let val2: &[u8] = &vec![2u8; 10240]; // 10KB
        let val3: &[u8] = &vec![3u8; 102400]; // 100KB
        let array = Arc::new(LargeBinaryArray::from(vec![
            Some(val1),
            None,
            Some(val2),
            Some(val3),
        ]));

        // Use the standard test harness
        check_round_trip_encoding_of_data(vec![array], &TestCases::default(), blob_metadata).await;
    }
}
