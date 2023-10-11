// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Reading TFRecord files into Arrow data
//!
//! Use [infer_tfrecord_schema] to infer the schema of a TFRecord file, then use
//! [read_tfrecord] to read the file into an Arrow record batch stream.

use arrow::buffer::OffsetBuffer;
use arrow_array::builder::PrimitiveBuilder;
use arrow_array::{ArrayRef, FixedSizeListArray, ListArray};
use arrow_buffer::ScalarBuffer;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use half::{bf16, f16};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::io::ObjectStore;
use arrow::record_batch::RecordBatch;
use arrow_schema::{
    DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use snafu::{location, Location};
use tfrecord::protobuf::feature::Kind;
use tfrecord::protobuf::{DataType as TensorDataType, TensorProto};
use tfrecord::record_reader::RecordStream;
use tfrecord::{Example, Feature};
/// Infer the Arrow schema from a TFRecord file.
///
/// The featured named by `tensor_features` will be assumed to be binary fields
/// containing serialized tensors (TensorProto messages). Currently only
/// fixed-shape tensors are supported.
///
/// The features named by `string_features` will be assumed to be UTF-8 encoded
/// strings.
///
/// `num_rows` determines the number of rows to read from the file to infer the
/// schema. If `None`, the entire file will be read.
pub async fn infer_tfrecord_schema(
    uri: &str,
    tensor_features: &[&str],
    string_features: &[&str],
    num_rows: Option<usize>,
) -> Result<ArrowSchema> {
    let mut columns: HashMap<String, FeatureMeta> = HashMap::new();

    let (store, path) = ObjectStore::from_uri(uri).await?;
    // TODO: should we avoid reading the entire file into memory?
    let data = store
        .inner
        .get(&path)
        .await?
        .into_stream()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
        .into_async_read();
    let mut records = RecordStream::<Example, _>::from_reader(data, Default::default());
    let mut i = 0;
    while let Some(record) = records.next().await {
        let record = record.map_err(|err| Error::IO {
            message: err.to_string(),
            location: location!(),
        })?;

        if let Some(features) = record.features {
            for (name, feature) in features.feature {
                if let Some(entry) = columns.get_mut(&name) {
                    entry.try_update(&feature)?;
                } else {
                    columns.insert(
                        name.clone(),
                        FeatureMeta::try_new(
                            &feature,
                            tensor_features.contains(&name.as_str()),
                            string_features.contains(&name.as_str()),
                        )?,
                    );
                }
            }
        }

        i += 1;
        if let Some(num_rows) = num_rows {
            if i >= num_rows {
                break;
            }
        }
    }

    let mut fields = columns
        .iter()
        .map(|(name, meta)| make_field(name, meta))
        .collect::<Result<Vec<_>>>()?;

    // To guarantee some sort of deterministic order, we sort the fields by name
    fields.sort_by(|a, b| a.name().cmp(b.name()));
    Ok(ArrowSchema::new(fields))
}

/// Read a TFRecord file into an Arrow record batch stream.
///
/// Reads `batch_size` rows at a time. If `batch_size` is `None`, a default
/// batch size of 10,000 is used.
///
/// The schema may be a partial schema, in which case only the fields present in
/// the schema will be read.
pub async fn read_tfrecord(
    uri: &str,
    schema: ArrowSchemaRef,
    batch_size: Option<usize>,
) -> Result<SendableRecordBatchStream> {
    let batch_size = batch_size.unwrap_or(10_000);

    let (store, path) = ObjectStore::from_uri(uri).await?;
    let data = store
        .inner
        .get(&path)
        .await?
        .into_stream()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
        .into_async_read();
    let schema_ref = schema.clone();
    let batch_stream = RecordStream::<Example, _>::from_reader(data, Default::default())
        .try_chunks(batch_size)
        .map(move |chunk| {
            let chunk = chunk.map_err(|err| DataFusionError::External(Box::new(err.1)))?;
            let batch = convert_batch(chunk, &schema_ref)?;
            Ok(batch)
        });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        batch_stream,
    )))
}

/// Check if a feature has more than 1 value.
fn feature_is_repeated(feature: &tfrecord::Feature) -> bool {
    match feature.kind.as_ref().unwrap() {
        Kind::BytesList(bytes_list) => bytes_list.value.len() > 1,
        Kind::FloatList(float_list) => float_list.value.len() > 1,
        Kind::Int64List(int64_list) => int64_list.value.len() > 1,
    }
}

/// Simplified representation of a features data type.
#[derive(Clone, PartialEq, Debug)]
enum FeatureType {
    Integer,
    Float,
    Binary,
    String,
    Tensor {
        shape: Vec<i64>,
        dtype: TensorDataType,
    },
}

/// General type information about a single feature.
struct FeatureMeta {
    /// Whether the feature contains multiple values per example. Ones that do
    /// will be converted to Arrow lists. Otherwise they will be primitive arrays.
    repeated: bool,
    feature_type: FeatureType,
}

impl FeatureMeta {
    /// Create a new FeatureMeta from a single example.
    pub fn try_new(feature: &Feature, is_tensor: bool, is_string: bool) -> Result<Self> {
        let feature_type = match feature.kind.as_ref().unwrap() {
            Kind::BytesList(data) => {
                if is_tensor {
                    Self::extract_tensor(data.value[0].as_slice())?
                } else if is_string {
                    FeatureType::String
                } else {
                    FeatureType::Binary
                }
            }
            Kind::FloatList(_) => FeatureType::Float,
            Kind::Int64List(_) => FeatureType::Integer,
        };
        Ok(Self {
            repeated: feature_is_repeated(feature),
            feature_type,
        })
    }

    /// Update the FeatureMeta with a new example, or return an error if the
    /// example is inconsistent with the existing FeatureMeta.
    pub fn try_update(&mut self, feature: &Feature) -> Result<()> {
        let feature_type = match feature.kind.as_ref().unwrap() {
            Kind::BytesList(data) => match self.feature_type {
                FeatureType::String => FeatureType::String,
                FeatureType::Binary => FeatureType::Binary,
                FeatureType::Tensor { .. } => Self::extract_tensor(data.value[0].as_slice())?,
                _ => {
                    return Err(Error::IO {
                        message: format!(
                            "Data type mismatch: expected {:?}, got {:?}",
                            self.feature_type,
                            feature.kind.as_ref().unwrap()
                        ),
                        location: location!(),
                    })
                }
            },
            Kind::FloatList(_) => FeatureType::Float,
            Kind::Int64List(_) => FeatureType::Integer,
        };
        if self.feature_type != feature_type {
            return Err(Error::IO {
                message: format!("inconsistent feature type for field {:?}", feature_type),
                location: location!(),
            });
        }
        if feature_is_repeated(feature) {
            self.repeated = true;
        }
        Ok(())
    }

    fn extract_tensor(data: &[u8]) -> Result<FeatureType> {
        let tensor_proto = TensorProto::decode(data)?;
        Ok(FeatureType::Tensor {
            shape: tensor_proto
                .tensor_shape
                .as_ref()
                .unwrap()
                .dim
                .iter()
                .map(|d| d.size)
                .collect(),
            dtype: tensor_proto.dtype(),
        })
    }
}

/// Metadata for a fixed-shape tensor.
#[derive(serde::Serialize)]
struct ArrowTensorMetadata {
    shape: Vec<i64>,
}

fn tensor_dtype_to_arrow(tensor_dtype: &TensorDataType) -> Result<DataType> {
    Ok(match tensor_dtype {
        TensorDataType::DtBfloat16 => DataType::FixedSizeBinary(2),
        TensorDataType::DtHalf => DataType::Float16,
        TensorDataType::DtFloat => DataType::Float32,
        TensorDataType::DtDouble => DataType::Float64,
        _ => {
            return Err(Error::IO {
                message: format!("unsupported tensor data type {:?}", tensor_dtype),
                location: location!(),
            });
        }
    })
}

fn make_field(name: &str, feature_meta: &FeatureMeta) -> Result<ArrowField> {
    let data_type = match &feature_meta.feature_type {
        FeatureType::Integer => DataType::Int64,
        FeatureType::Float => DataType::Float32,
        FeatureType::Binary => DataType::Binary,
        FeatureType::String => DataType::Utf8,
        FeatureType::Tensor { shape, dtype } => {
            let list_size = shape.iter().map(|x| *x as i32).product();
            let inner_type = tensor_dtype_to_arrow(dtype)?;

            let inner_meta = match dtype {
                TensorDataType::DtBfloat16 => Some(
                    [("ARROW:extension:name", "lance.bfloat16")]
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect::<HashMap<String, String>>(),
                ),
                _ => None,
            };
            let mut inner_field = ArrowField::new("item", inner_type, true);
            if let Some(metadata) = inner_meta {
                inner_field.set_metadata(metadata);
            }

            DataType::FixedSizeList(Arc::new(inner_field), list_size)
        }
    };

    // This metadata marks the field as a tensor column, which PyArrow can
    // recognize.
    let metadata = match &feature_meta.feature_type {
        FeatureType::Tensor { shape, dtype: _ } => {
            let mut metadata = HashMap::new();
            let tensor_metadata = ArrowTensorMetadata {
                shape: shape.clone(),
            };
            metadata.insert(
                "ARROW:extension:name".to_string(),
                "arrow.fixed_shape_tensor".to_string(),
            );
            metadata.insert(
                "ARROW:extension:metadata".to_string(),
                serde_json::to_string(&tensor_metadata)?,
            );
            Some(metadata)
        }
        _ => None,
    };

    let mut field = if feature_meta.repeated {
        ArrowField::new("item", data_type, true)
    } else {
        ArrowField::new(name, data_type, true)
    };
    if let Some(metadata) = metadata {
        field.set_metadata(metadata);
    }

    let field = if feature_meta.repeated {
        ArrowField::new(name, DataType::List(Arc::new(field)), true)
    } else {
        field
    };

    Ok(field)
}

/// Convert a vector of TFRecord examples into an Arrow record batch.
fn convert_batch(records: Vec<Example>, schema: &ArrowSchema) -> Result<RecordBatch> {
    // TODO: do this in parallel?
    let columns = schema
        .fields
        .iter()
        .map(|field| convert_column(&records, field))
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;
    Ok(batch)
}

/// Convert a single column of TFRecord examples into an Arrow array.
fn convert_column(records: &[Example], field: &ArrowField) -> Result<ArrayRef> {
    let type_info = parse_type(field.data_type());
    // Make leaf type
    let (mut column, offsets) = convert_leaf(records, field.name(), &type_info)?;

    if let Some(fsl_size) = &type_info.fsl_size {
        let mut field = ArrowField::new("item", type_info.leaf_type.clone(), true);
        if matches!(&type_info.leaf_type, DataType::FixedSizeBinary(2)) {
            field.set_metadata(
                [
                    (
                        "ARROW:extension:name".to_string(),
                        "lance.bfloat16".to_string(),
                    ),
                    ("ARROW:extension:metadata".to_string(), "".to_string()),
                ]
                .into_iter()
                .collect(),
            );
        }
        // Wrap in a FSL
        column = Arc::new(FixedSizeListArray::try_new(
            Arc::new(field),
            *fsl_size,
            column,
            None,
        )?);
    }

    if type_info.in_list {
        column = Arc::new(ListArray::try_new(
            Arc::new(ArrowField::new("item", column.data_type().clone(), true)),
            offsets.unwrap(),
            column,
            None,
        )?);
    }

    Ok(column)
}

/// Representation of a field in the TFRecord file. It can be a leaf type, a
/// tensor, or a list of either.
struct TypeInfo {
    leaf_type: DataType,
    fsl_size: Option<i32>,
    in_list: bool,
}

fn parse_type(data_type: &DataType) -> TypeInfo {
    match data_type {
        DataType::FixedSizeList(inner_field, list_size) => {
            let inner_type = parse_type(inner_field.data_type());
            TypeInfo {
                leaf_type: inner_type.leaf_type,
                fsl_size: Some(*list_size),
                in_list: false,
            }
        }
        DataType::List(inner_field) => {
            let inner_type = parse_type(inner_field.data_type());
            TypeInfo {
                leaf_type: inner_type.leaf_type,
                fsl_size: inner_type.fsl_size,
                in_list: true,
            }
        }
        _ => TypeInfo {
            leaf_type: data_type.clone(),
            fsl_size: None,
            in_list: false,
        },
    }
}

fn convert_leaf(
    records: &[Example],
    name: &str,
    type_info: &TypeInfo,
) -> Result<(ArrayRef, Option<OffsetBuffer<i32>>)> {
    use arrow::array::*;
    let features: Vec<Option<&Feature>> = records
        .iter()
        .map(|record| {
            let features = record.features.as_ref().unwrap();
            features.feature.get(name)
        })
        .collect();
    let (values, offsets): (ArrayRef, Option<OffsetBuffer<i32>>) = match type_info {
        // First, the Non-tensor leaf types
        TypeInfo {
            leaf_type: DataType::Int64,
            fsl_size: None,
            in_list,
        } => {
            let mut values = Int64Builder::with_capacity(features.len());
            for feature in features.iter() {
                if let Some(Feature {
                    kind: Some(Kind::Int64List(list)),
                }) = feature
                {
                    values.append_slice(&list.value);
                } else if !type_info.in_list {
                    values.append_null();
                }
            }
            let offsets = if *in_list {
                Some(compute_offsets(&features, type_info))
            } else {
                None
            };
            (Arc::new(values.finish()), offsets)
        }
        TypeInfo {
            leaf_type: DataType::Float32,
            fsl_size: None,
            in_list,
        } => {
            let mut values = Float32Builder::with_capacity(features.len());
            for feature in features.iter() {
                if let Some(Feature {
                    kind: Some(Kind::FloatList(list)),
                }) = feature
                {
                    values.append_slice(&list.value);
                } else if !type_info.in_list {
                    values.append_null();
                }
            }
            let offsets = if *in_list {
                Some(compute_offsets(&features, type_info))
            } else {
                None
            };
            (Arc::new(values.finish()), offsets)
        }
        TypeInfo {
            leaf_type: DataType::Binary,
            fsl_size: None,
            in_list,
        } => {
            let mut values = BinaryBuilder::with_capacity(features.len(), 1024);
            for feature in features.iter() {
                if let Some(Feature {
                    kind: Some(Kind::BytesList(list)),
                }) = feature
                {
                    for value in &list.value {
                        values.append_value(value);
                    }
                } else if !type_info.in_list {
                    values.append_null();
                }
            }
            let offsets = if *in_list {
                Some(compute_offsets(&features, type_info))
            } else {
                None
            };
            (Arc::new(values.finish()), offsets)
        }
        TypeInfo {
            leaf_type: DataType::Utf8,
            fsl_size: None,
            in_list,
        } => {
            let mut values = StringBuilder::with_capacity(features.len(), 1024);
            for feature in features.iter() {
                if let Some(Feature {
                    kind: Some(Kind::BytesList(list)),
                }) = feature
                {
                    for value in &list.value {
                        values.append_value(String::from_utf8_lossy(value));
                    }
                } else if !type_info.in_list {
                    values.append_null();
                }
            }
            let offsets = if *in_list {
                Some(compute_offsets(&features, type_info))
            } else {
                None
            };
            (Arc::new(values.finish()), offsets)
        }
        // Now, handle tensors
        TypeInfo {
            fsl_size: Some(_), ..
        } => convert_fixedshape_tensor(&features, type_info)?,
        _ => Err(Error::IO {
            message: format!("unsupported type {:?}", type_info.leaf_type),
            location: location!(),
        })?,
    };

    Ok((values, offsets))
}

fn compute_offsets(features: &Vec<Option<&Feature>>, type_info: &TypeInfo) -> OffsetBuffer<i32> {
    let mut offsets: Vec<i32> = Vec::with_capacity(features.len() + 1);
    offsets.push(0);

    let mut current = 0;
    for feature in features.iter() {
        if let Some(feature) = feature {
            match (
                type_info.fsl_size.is_some(),
                &type_info.leaf_type,
                feature.kind.as_ref().unwrap(),
            ) {
                (true, _, Kind::BytesList(list)) => {
                    current += list.value.len() as i32;
                }
                (false, DataType::Binary, Kind::BytesList(list)) => {
                    current += list.value.len() as i32;
                }
                (false, DataType::Utf8, Kind::BytesList(list)) => {
                    current += list.value.len() as i32;
                }
                (false, DataType::Float32, Kind::FloatList(list)) => {
                    current += list.value.len() as i32;
                }
                (false, DataType::Int64, Kind::Int64List(list)) => {
                    current += list.value.len() as i32;
                }
                _ => {} // Ignore mismatched types
            }
        }
        offsets.push(current);
    }

    OffsetBuffer::new(ScalarBuffer::from(offsets))
}

// /// Convert TensorProto message into an element of a FixedShapeTensor array and
// /// append it to the builder.
// ///
// /// TensorProto definition:
// /// https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor.proto
// ///
// /// FixedShapeTensor definition:
// /// https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
fn convert_fixedshape_tensor(
    features: &Vec<Option<&Feature>>,
    type_info: &TypeInfo,
) -> Result<(ArrayRef, Option<OffsetBuffer<i32>>)> {
    use arrow::array::*;
    let tensor_iter = features.iter().map(|maybe_feature| {
        if let Some(feature) = maybe_feature {
            if let Some(Kind::BytesList(list)) = &feature.kind {
                list.value
                    .iter()
                    .map(|val| TensorProto::decode(val.as_slice()))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map(Some)
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    });

    let offsets = if type_info.in_list {
        Some(compute_offsets(features, type_info))
    } else {
        None
    };

    let list_size = type_info.fsl_size.unwrap() as usize;

    let values: ArrayRef = match type_info.leaf_type {
        DataType::Float16 => {
            let mut values = Float16Builder::with_capacity(features.len());
            for tensors in tensor_iter {
                if let Some(tensors) = tensors? {
                    for tensor in tensors {
                        validate_tensor(&tensor, type_info)?;
                        if tensor.half_val.is_empty() {
                            append_primitive_from_slice(
                                &mut values,
                                tensor.tensor_content.as_slice(),
                                |bytes| f16::from_le_bytes(bytes.try_into().unwrap()),
                            )
                        } else {
                            // The individual values have padding (they are stored as i32)
                            // because protobuf has no 2-byte type
                            for value in tensor.half_val {
                                values.append_value(f16::from_bits(value as u16));
                            }
                        }
                    }
                } else {
                    values.append_nulls(list_size);
                }
            }
            Arc::new(values.finish())
        }
        // BFloat16
        DataType::FixedSizeBinary(2) => {
            let mut values = FixedSizeBinaryBuilder::with_capacity(features.len(), 2);

            for tensors in tensor_iter {
                if let Some(tensors) = tensors? {
                    for tensor in tensors {
                        validate_tensor(&tensor, type_info)?;
                        if tensor.half_val.is_empty() {
                            // Just directly move the bytes
                            for bytes in tensor.tensor_content.as_slice().chunks_exact(2) {
                                values.append_value(bytes)?;
                            }
                        } else {
                            // The individual values have padding (they are stored as i32)
                            // because protobuf has no 2-byte type
                            for value in tensor.half_val {
                                let bf16_value = bf16::from_bits(value as u16);
                                values.append_value(bf16_value.to_le_bytes())?;
                            }
                        }
                    }
                } else {
                    for _ in 0..list_size {
                        values.append_null();
                    }
                }
            }
            Arc::new(values.finish())
        }
        DataType::Float32 => {
            let mut values = Float32Builder::with_capacity(features.len());
            for tensors in tensor_iter {
                if let Some(tensors) = tensors? {
                    for tensor in tensors {
                        validate_tensor(&tensor, type_info)?;
                        if tensor.float_val.is_empty() {
                            append_primitive_from_slice(
                                &mut values,
                                tensor.tensor_content.as_slice(),
                                |bytes| f32::from_le_bytes(bytes.try_into().unwrap()),
                            )
                        } else {
                            values.append_slice(tensor.float_val.as_slice());
                        }
                    }
                } else {
                    values.append_nulls(list_size);
                }
            }
            Arc::new(values.finish())
        }
        DataType::Float64 => {
            let mut values = Float64Builder::with_capacity(features.len());
            for tensors in tensor_iter {
                if let Some(tensors) = tensors? {
                    for tensor in tensors {
                        validate_tensor(&tensor, type_info)?;
                        if tensor.float_val.is_empty() {
                            append_primitive_from_slice(
                                &mut values,
                                tensor.tensor_content.as_slice(),
                                |bytes| f64::from_le_bytes(bytes.try_into().unwrap()),
                            )
                        } else {
                            values.append_slice(tensor.double_val.as_slice())
                        };
                    }
                } else {
                    values.append_nulls(list_size);
                }
            }
            Arc::new(values.finish())
        }
        _ => Err(Error::IO {
            message: format!("unsupported type {:?}", type_info.leaf_type),
            location: location!(),
        })?,
    };

    Ok((values, offsets))
}

fn validate_tensor(tensor: &TensorProto, type_info: &TypeInfo) -> Result<()> {
    let tensor_shape = tensor.tensor_shape.as_ref().unwrap();
    let length = tensor_shape.dim.iter().map(|d| d.size as i32).product();
    if type_info.fsl_size != Some(length) {
        return Err(Error::IO {
            message: format!(
                "tensor length mismatch: expected {}, got {}",
                type_info.fsl_size.unwrap(),
                length
            ),
            location: location!(),
        });
    }

    let data_type = tensor_dtype_to_arrow(&tensor.dtype())?;
    if data_type != type_info.leaf_type {
        return Err(Error::IO {
            message: format!(
                "tensor type mismatch: expected {:?}, got {:?}",
                type_info.leaf_type,
                tensor.dtype()
            ),
            location: location!(),
        });
    }

    Ok(())
}

/// Given a potentially unaligned slice, append the slice to the builder.
fn append_primitive_from_slice<T>(
    builder: &mut PrimitiveBuilder<T>,
    slice: &[u8],
    parse_val: impl Fn(&[u8]) -> T::Native,
) where
    T: arrow::datatypes::ArrowPrimitiveType,
{
    // Safety: we are trusting that the data in the buffer are valid for the
    // datatype T::Native, as claimed by the file. There isn't anywhere for
    // TensorProto to tell us the original endianness, so it's possible there
    // could be a mismatch here.
    let (prefix, middle, suffix) = unsafe { slice.align_to::<T::Native>() };
    for val in prefix.chunks_exact(T::get_byte_width()) {
        builder.append_value(parse_val(val));
    }

    builder.append_slice(middle);

    for val in suffix.chunks_exact(T::get_byte_width()) {
        builder.append_value(parse_val(val));
    }
}
