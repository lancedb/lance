// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    builder::{PrimitiveBuilder, StringBuilder},
    cast::AsArray,
    types::{UInt32Type, UInt64Type, UInt8Type},
    Array, ArrayRef, StructArray, UInt64Array,
};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field as ArrowField, Fields};
use futures::{future::BoxFuture, FutureExt};
use lance_core::{
    datatypes::Field, datatypes::BLOB_V2_DESC_FIELDS, error::LanceOptionExt, Error, Result,
};
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    constants::PACKED_STRUCT_META_KEY,
    decoder::PageEncoding,
    encoder::{EncodeTask, EncodedColumn, EncodedPage, FieldEncoder, OutOfLineBuffers},
    encodings::logical::primitive::PrimitiveStructuralEncoder,
    format::ProtobufUtils21,
    repdef::{DefinitionInterpretation, RepDefBuilder},
};
use lance_core::datatypes::BlobKind;

/// Blob structural encoder - stores large binary data in external buffers
///
/// This encoder takes large binary arrays and stores them outside the normal
/// page structure. It creates a descriptor (position, size) for each blob
/// that is stored inline in the page.
pub struct BlobStructuralEncoder {
    // Encoder for the descriptors (position/size struct)
    descriptor_encoder: Box<dyn FieldEncoder>,
    // Set when we first see data
    def_meaning: Option<Arc<[DefinitionInterpretation]>>,
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

        Ok(Self {
            descriptor_encoder,
            def_meaning: None,
        })
    }

    fn wrap_tasks(
        tasks: Vec<EncodeTask>,
        def_meaning: Arc<[DefinitionInterpretation]>,
    ) -> Vec<EncodeTask> {
        tasks
            .into_iter()
            .map(|task| {
                let def_meaning = def_meaning.clone();
                task.then(|encoded_page| async move {
                    let encoded_page = encoded_page?;

                    let PageEncoding::Structural(inner_layout) = encoded_page.description else {
                        return Err(Error::Internal {
                            message: "Expected inner encoding to return structural layout"
                                .to_string(),
                            location: location!(),
                        });
                    };

                    let wrapped = ProtobufUtils21::blob_layout(inner_layout, &def_meaning);
                    Ok(EncodedPage {
                        column_idx: encoded_page.column_idx,
                        data: encoded_page.data,
                        description: PageEncoding::Structural(wrapped),
                        num_rows: encoded_page.num_rows,
                        row_number: encoded_page.row_number,
                    })
                })
                .boxed()
            })
            .collect::<Vec<_>>()
    }
}

impl FieldEncoder for BlobStructuralEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        mut repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        if let Some(validity) = array.nulls() {
            repdef.add_validity_bitmap(validity.clone());
        } else {
            repdef.add_no_null(array.len());
        }

        // Convert input array to LargeBinary
        let binary_array = array
            .as_binary_opt::<i64>()
            .ok_or_else(|| Error::InvalidInput {
                source: format!("Expected LargeBinary array, got {}", array.data_type()).into(),
                location: location!(),
            })?;

        let repdef = RepDefBuilder::serialize(vec![repdef]);

        let rep = repdef.repetition_levels.as_ref();
        let def = repdef.definition_levels.as_ref();
        let def_meaning: Arc<[DefinitionInterpretation]> = repdef.def_meaning.into();

        match self.def_meaning.as_ref() {
            None => {
                self.def_meaning = Some(def_meaning.clone());
            }
            Some(existing) => {
                debug_assert_eq!(existing, &def_meaning);
            }
        }

        // Collect positions and sizes
        let mut positions = Vec::with_capacity(binary_array.len());
        let mut sizes = Vec::with_capacity(binary_array.len());

        for i in 0..binary_array.len() {
            if binary_array.is_null(i) {
                // Null values are smuggled into the positions array

                // If we have null values we must have definition levels
                let mut repdef = (def.expect_ok()?[i] as u64) << 16;
                if let Some(rep) = rep {
                    repdef += rep[i] as u64;
                }

                debug_assert_ne!(repdef, 0);
                positions.push(repdef);
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
            None, // Descriptors are never null
        ));

        // Delegate to descriptor encoder
        let encode_tasks = self.descriptor_encoder.maybe_encode(
            descriptor_array,
            external_buffers,
            RepDefBuilder::default(),
            row_number,
            num_rows,
        )?;

        Ok(Self::wrap_tasks(encode_tasks, def_meaning))
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        let encode_tasks = self.descriptor_encoder.flush(external_buffers)?;

        // Use the cached def meaning.  If we haven't seen any data yet then we can just use a dummy
        // value (not clear there would be any encode tasks in that case)
        let def_meaning = self
            .def_meaning
            .clone()
            .unwrap_or_else(|| Arc::new([DefinitionInterpretation::AllValidItem]));

        Ok(Self::wrap_tasks(encode_tasks, def_meaning))
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

/// Blob v2 structural encoder
pub struct BlobV2StructuralEncoder {
    descriptor_encoder: Box<dyn FieldEncoder>,
}

impl BlobV2StructuralEncoder {
    pub fn new(
        field: &Field,
        column_index: u32,
        options: &crate::encoder::EncodingOptions,
        compression_strategy: Arc<dyn crate::compression::CompressionStrategy>,
    ) -> Result<Self> {
        let mut descriptor_metadata = HashMap::with_capacity(1);
        descriptor_metadata.insert(PACKED_STRUCT_META_KEY.to_string(), "true".to_string());

        let descriptor_data_type = DataType::Struct(BLOB_V2_DESC_FIELDS.clone());

        let descriptor_field = Field::try_from(
            ArrowField::new(&field.name, descriptor_data_type, field.nullable)
                .with_metadata(descriptor_metadata),
        )?;

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

impl FieldEncoder for BlobV2StructuralEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        mut repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        let struct_arr = array.as_struct();
        if let Some(validity) = struct_arr.nulls() {
            repdef.add_validity_bitmap(validity.clone());
        } else {
            repdef.add_no_null(struct_arr.len());
        }

        let kind_col = struct_arr
            .column_by_name("kind")
            .ok_or_else(|| Error::InvalidInput {
                source: "Blob v2 struct missing `kind` field".into(),
                location: location!(),
            })?
            .as_primitive::<UInt8Type>();
        let data_col = struct_arr
            .column_by_name("data")
            .ok_or_else(|| Error::InvalidInput {
                source: "Blob v2 struct missing `data` field".into(),
                location: location!(),
            })?
            .as_binary::<i64>();
        let uri_col = struct_arr
            .column_by_name("uri")
            .ok_or_else(|| Error::InvalidInput {
                source: "Blob v2 struct missing `uri` field".into(),
                location: location!(),
            })?
            .as_string::<i32>();
        let blob_id_col = struct_arr
            .column_by_name("blob_id")
            .ok_or_else(|| Error::InvalidInput {
                source: "Blob v2 struct missing `blob_id` field".into(),
                location: location!(),
            })?
            .as_primitive::<UInt32Type>();
        let blob_size_col = struct_arr
            .column_by_name("blob_size")
            .ok_or_else(|| Error::InvalidInput {
                source: "Blob v2 struct missing `blob_size` field".into(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>();
        let packed_position_col = struct_arr
            .column_by_name("position")
            .ok_or_else(|| Error::InvalidInput {
                source: "Blob v2 struct missing `position` field".into(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>();

        let row_count = struct_arr.len();

        let mut kind_builder = PrimitiveBuilder::<UInt8Type>::with_capacity(row_count);
        let mut position_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(row_count);
        let mut size_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(row_count);
        let mut blob_id_builder = PrimitiveBuilder::<UInt32Type>::with_capacity(row_count);
        let mut uri_builder = StringBuilder::with_capacity(row_count, row_count * 16);

        for i in 0..row_count {
            let (kind_value, position_value, size_value, blob_id_value, uri_value) =
                if struct_arr.is_null(i) || kind_col.is_null(i) {
                    (BlobKind::Inline as u8, 0, 0, 0, "".to_string())
                } else {
                    let kind_val = BlobKind::try_from(kind_col.value(i))?;
                    match kind_val {
                        BlobKind::Dedicated => (
                            BlobKind::Dedicated as u8,
                            0,
                            blob_size_col.value(i),
                            blob_id_col.value(i),
                            "".to_string(),
                        ),
                        BlobKind::External => {
                            let uri = uri_col.value(i).to_string();
                            let position = if packed_position_col.is_null(i) {
                                0
                            } else {
                                packed_position_col.value(i)
                            };
                            let size = if blob_size_col.is_null(i) {
                                0
                            } else {
                                blob_size_col.value(i)
                            };
                            (BlobKind::External as u8, position, size, 0, uri)
                        }
                        BlobKind::Packed => (
                            BlobKind::Packed as u8,
                            packed_position_col.value(i),
                            blob_size_col.value(i),
                            blob_id_col.value(i),
                            "".to_string(),
                        ),
                        BlobKind::Inline => {
                            let data_val = data_col.value(i);
                            let blob_len = data_val.len() as u64;
                            let position = external_buffers
                                .add_buffer(LanceBuffer::from(Buffer::from(data_val)));

                            (
                                BlobKind::Inline as u8,
                                position,
                                blob_len,
                                0,
                                "".to_string(),
                            )
                        }
                    }
                };

            kind_builder.append_value(kind_value);
            position_builder.append_value(position_value);
            size_builder.append_value(size_value);
            blob_id_builder.append_value(blob_id_value);
            uri_builder.append_value(uri_value);
        }
        let children: Vec<ArrayRef> = vec![
            Arc::new(kind_builder.finish()),
            Arc::new(position_builder.finish()),
            Arc::new(size_builder.finish()),
            Arc::new(blob_id_builder.finish()),
            Arc::new(uri_builder.finish()),
        ];

        let descriptor_array = Arc::new(StructArray::try_new(
            BLOB_V2_DESC_FIELDS.clone(),
            children,
            None,
        )?) as ArrayRef;

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
        testing::{
            check_round_trip_encoding_of_data, check_round_trip_encoding_of_data_with_expected,
            TestCases,
        },
        version::LanceFileVersion,
    };
    use arrow_array::{
        ArrayRef, LargeBinaryArray, StringArray, StructArray, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field as ArrowField};

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
                HashMap::from([(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string())]),
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
        let blob_metadata =
            HashMap::from([(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string())]);

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
        check_round_trip_encoding_of_data(
            vec![array],
            &TestCases::default().with_max_file_version(LanceFileVersion::V2_1),
            blob_metadata,
        )
        .await;
    }

    #[tokio::test]
    async fn test_blob_v2_external_round_trip() {
        let blob_metadata = HashMap::from([(
            lance_arrow::ARROW_EXT_NAME_KEY.to_string(),
            lance_arrow::BLOB_V2_EXT_NAME.to_string(),
        )]);

        let kind_field = Arc::new(ArrowField::new("kind", DataType::UInt8, true));
        let data_field = Arc::new(ArrowField::new("data", DataType::LargeBinary, true));
        let uri_field = Arc::new(ArrowField::new("uri", DataType::Utf8, true));
        let blob_id_field = Arc::new(ArrowField::new("blob_id", DataType::UInt32, true));
        let blob_size_field = Arc::new(ArrowField::new("blob_size", DataType::UInt64, true));
        let position_field = Arc::new(ArrowField::new("position", DataType::UInt64, true));

        let kind_array = UInt8Array::from(vec![
            BlobKind::Inline as u8,
            BlobKind::External as u8,
            BlobKind::External as u8,
        ]);
        let data_array = LargeBinaryArray::from(vec![Some(b"inline".as_ref()), None, None]);
        let uri_array = StringArray::from(vec![
            None,
            Some("file:///tmp/external.bin"),
            Some("s3://bucket/blob"),
        ]);
        let blob_id_array = UInt32Array::from(vec![0, 0, 0]);
        let blob_size_array = UInt64Array::from(vec![0, 0, 0]);
        let position_array = UInt64Array::from(vec![0, 0, 0]);

        let struct_array = StructArray::from(vec![
            (kind_field, Arc::new(kind_array) as ArrayRef),
            (data_field, Arc::new(data_array) as ArrayRef),
            (uri_field, Arc::new(uri_array) as ArrayRef),
            (blob_id_field, Arc::new(blob_id_array) as ArrayRef),
            (blob_size_field, Arc::new(blob_size_array) as ArrayRef),
            (position_field, Arc::new(position_array) as ArrayRef),
        ]);

        let expected_descriptor = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("kind", DataType::UInt8, false)),
                Arc::new(UInt8Array::from(vec![
                    BlobKind::Inline as u8,
                    BlobKind::External as u8,
                    BlobKind::External as u8,
                ])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("position", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0, 0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("size", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![6, 0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_id", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0, 0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_uri", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec![
                    "",
                    "file:///tmp/external.bin",
                    "s3://bucket/blob",
                ])) as ArrayRef,
            ),
        ]);

        check_round_trip_encoding_of_data_with_expected(
            vec![Arc::new(struct_array)],
            Some(Arc::new(expected_descriptor)),
            &TestCases::default().with_min_file_version(LanceFileVersion::V2_2),
            blob_metadata,
        )
        .await;
    }

    #[tokio::test]
    async fn test_blob_v2_dedicated_round_trip() {
        let blob_metadata = HashMap::from([(
            lance_arrow::ARROW_EXT_NAME_KEY.to_string(),
            lance_arrow::BLOB_V2_EXT_NAME.to_string(),
        )]);

        let kind_field = Arc::new(ArrowField::new("kind", DataType::UInt8, true));
        let data_field = Arc::new(ArrowField::new("data", DataType::LargeBinary, true));
        let uri_field = Arc::new(ArrowField::new("uri", DataType::Utf8, true));
        let blob_id_field = Arc::new(ArrowField::new("blob_id", DataType::UInt32, true));
        let blob_size_field = Arc::new(ArrowField::new("blob_size", DataType::UInt64, true));
        let position_field = Arc::new(ArrowField::new("position", DataType::UInt64, true));

        let kind_array = UInt8Array::from(vec![BlobKind::Dedicated as u8, BlobKind::Inline as u8]);
        let data_array = LargeBinaryArray::from(vec![None, Some(b"abc".as_ref())]);
        let uri_array = StringArray::from(vec![Option::<&str>::None, None]);
        let blob_id_array = UInt32Array::from(vec![42, 0]);
        let blob_size_array = UInt64Array::from(vec![12, 0]);
        let position_array = UInt64Array::from(vec![0, 0]);

        let struct_array = StructArray::from(vec![
            (kind_field, Arc::new(kind_array) as ArrayRef),
            (data_field, Arc::new(data_array) as ArrayRef),
            (uri_field, Arc::new(uri_array) as ArrayRef),
            (blob_id_field, Arc::new(blob_id_array) as ArrayRef),
            (blob_size_field, Arc::new(blob_size_array) as ArrayRef),
            (position_field, Arc::new(position_array) as ArrayRef),
        ]);

        let expected_descriptor = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("kind", DataType::UInt8, false)),
                Arc::new(UInt8Array::from(vec![
                    BlobKind::Dedicated as u8,
                    BlobKind::Inline as u8,
                ])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("position", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("size", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![12, 3])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_id", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![42, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_uri", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["", ""])) as ArrayRef,
            ),
        ]);

        check_round_trip_encoding_of_data_with_expected(
            vec![Arc::new(struct_array)],
            Some(Arc::new(expected_descriptor)),
            &TestCases::default().with_min_file_version(LanceFileVersion::V2_2),
            blob_metadata,
        )
        .await;
    }

    #[tokio::test]
    async fn test_blob_v2_external_with_range_round_trip() {
        let blob_metadata = HashMap::from([(
            lance_arrow::ARROW_EXT_NAME_KEY.to_string(),
            lance_arrow::BLOB_V2_EXT_NAME.to_string(),
        )]);

        let kind_field = Arc::new(ArrowField::new("kind", DataType::UInt8, true));
        let data_field = Arc::new(ArrowField::new("data", DataType::LargeBinary, true));
        let uri_field = Arc::new(ArrowField::new("uri", DataType::Utf8, true));
        let blob_id_field = Arc::new(ArrowField::new("blob_id", DataType::UInt32, true));
        let blob_size_field = Arc::new(ArrowField::new("blob_size", DataType::UInt64, true));
        let position_field = Arc::new(ArrowField::new("position", DataType::UInt64, true));

        let kind_array = UInt8Array::from(vec![BlobKind::External as u8]);
        let data_array = LargeBinaryArray::from(vec![None::<&[u8]>]);
        let uri_array = StringArray::from(vec![Some("memory://container.pack")]);
        let blob_id_array = UInt32Array::from(vec![0]);
        let blob_size_array = UInt64Array::from(vec![42]);
        let position_array = UInt64Array::from(vec![7]);

        let struct_array = StructArray::from(vec![
            (kind_field, Arc::new(kind_array) as ArrayRef),
            (data_field, Arc::new(data_array) as ArrayRef),
            (uri_field, Arc::new(uri_array) as ArrayRef),
            (blob_id_field, Arc::new(blob_id_array) as ArrayRef),
            (blob_size_field, Arc::new(blob_size_array) as ArrayRef),
            (position_field, Arc::new(position_array) as ArrayRef),
        ]);

        let expected_descriptor = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("kind", DataType::UInt8, false)),
                Arc::new(UInt8Array::from(vec![BlobKind::External as u8])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("position", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![7])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("size", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![42])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_id", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_uri", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["memory://container.pack"])) as ArrayRef,
            ),
        ]);

        check_round_trip_encoding_of_data_with_expected(
            vec![Arc::new(struct_array)],
            Some(Arc::new(expected_descriptor)),
            &TestCases::default().with_min_file_version(LanceFileVersion::V2_2),
            blob_metadata,
        )
        .await;
    }

    #[tokio::test]
    async fn test_blob_v2_packed_round_trip() {
        let blob_metadata = HashMap::from([(
            lance_arrow::ARROW_EXT_NAME_KEY.to_string(),
            lance_arrow::BLOB_V2_EXT_NAME.to_string(),
        )]);

        let kind_field = Arc::new(ArrowField::new("kind", DataType::UInt8, true));
        let data_field = Arc::new(ArrowField::new("data", DataType::LargeBinary, true));
        let uri_field = Arc::new(ArrowField::new("uri", DataType::Utf8, true));
        let blob_id_field = Arc::new(ArrowField::new("blob_id", DataType::UInt32, true));
        let blob_size_field = Arc::new(ArrowField::new("blob_size", DataType::UInt64, true));
        let position_field = Arc::new(ArrowField::new("position", DataType::UInt64, true));

        let kind_array = UInt8Array::from(vec![BlobKind::Packed as u8]);
        let data_array = LargeBinaryArray::from(vec![None::<&[u8]>]);
        let uri_array = StringArray::from(vec![None::<&str>]);
        let blob_id_array = UInt32Array::from(vec![7]);
        let blob_size_array = UInt64Array::from(vec![5]);
        let position_array = UInt64Array::from(vec![10]);

        let struct_array = StructArray::from(vec![
            (kind_field, Arc::new(kind_array) as ArrayRef),
            (data_field, Arc::new(data_array) as ArrayRef),
            (uri_field, Arc::new(uri_array) as ArrayRef),
            (blob_id_field, Arc::new(blob_id_array) as ArrayRef),
            (blob_size_field, Arc::new(blob_size_array) as ArrayRef),
            (position_field, Arc::new(position_array) as ArrayRef),
        ]);

        let expected_descriptor = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("kind", DataType::UInt8, false)),
                Arc::new(UInt8Array::from(vec![BlobKind::Packed as u8])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("position", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![10])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("size", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![5])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_id", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![7])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("blob_uri", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec![""])) as ArrayRef,
            ),
        ]);

        check_round_trip_encoding_of_data_with_expected(
            vec![Arc::new(struct_array)],
            Some(Arc::new(expected_descriptor)),
            &TestCases::default().with_min_file_version(LanceFileVersion::V2_2),
            blob_metadata,
        )
        .await;
    }
}
