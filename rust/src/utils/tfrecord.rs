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

use arrow::buffer::OffsetBuffer;
use arrow_array::{FixedSizeListArray, ListArray};
use arrow_buffer::ScalarBuffer;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;

use crate::arrow::FixedSizeListArrayExt;
use crate::error::{Error, Result};
use crate::io::ObjectStore;
use arrow::record_batch::RecordBatch;
use arrow_array::builder::{
    make_builder, ArrayBuilder, BinaryBuilder, Int32BufferBuilder, Int32Builder, ListBuilder,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use tfrecord::protobuf::feature::Kind;
use tfrecord::protobuf::{DataType as TensorDataType, TensorProto};
use tfrecord::record_reader::RecordStream;
use tfrecord::{Example, Feature};

fn feature_is_repeated(feature: &tfrecord::Feature) -> bool {
    match feature.kind.as_ref().unwrap() {
        Kind::BytesList(bytes_list) => bytes_list.value.len() > 1,
        Kind::FloatList(float_list) => float_list.value.len() > 1,
        Kind::Int64List(int64_list) => int64_list.value.len() > 1,
    }
}

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

struct FeatureMeta {
    repeated: bool,
    feature_type: FeatureType,
}

impl FeatureMeta {
    pub fn new(feature: &Feature, is_tensor: bool, is_string: bool) -> Self {
        let feature_type = match feature.kind.as_ref().unwrap() {
            Kind::BytesList(data) => {
                if is_tensor {
                    let val = &data.value[0];
                    let tensor_proto = TensorProto::decode(val.as_slice()).unwrap();
                    FeatureType::Tensor {
                        shape: tensor_proto
                            .tensor_shape
                            .as_ref()
                            .unwrap()
                            .dim
                            .iter()
                            .map(|d| d.size)
                            .collect(),
                        dtype: tensor_proto.dtype(),
                    }
                } else if is_string {
                    FeatureType::String
                } else {
                    FeatureType::Binary
                }
            }
            Kind::FloatList(_) => FeatureType::Float,
            Kind::Int64List(_) => FeatureType::Integer,
        };
        Self {
            repeated: feature_is_repeated(&feature),
            feature_type,
        }
    }

    pub fn try_update(&mut self, feature: &Feature) -> Result<()> {
        let feature_type = match feature.kind.as_ref().unwrap() {
            Kind::BytesList(data) => {
                let val = &data.value[0];
                let tensor_proto = TensorProto::decode(val.as_slice()).unwrap();
                FeatureType::Tensor {
                    shape: tensor_proto
                        .tensor_shape
                        .as_ref()
                        .unwrap()
                        .dim
                        .iter()
                        .map(|d| d.size)
                        .collect(),
                    dtype: tensor_proto.dtype(),
                }
            }
            Kind::FloatList(_) => FeatureType::Float,
            Kind::Int64List(_) => FeatureType::Integer,
        };
        if self.feature_type != feature_type {
            return Err(Error::IO {
                message: format!("inconsistent feature type for field {:?}", feature_type),
            });
        }
        if feature_is_repeated(&feature) {
            self.repeated = true;
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
struct ArrowTensorMetadata {
    shape: Vec<i64>,
}

fn make_field(name: &str, feature_meta: &FeatureMeta) -> Result<ArrowField> {
    let data_type = match &feature_meta.feature_type {
        FeatureType::Integer => DataType::Int64,
        FeatureType::Float => DataType::Float32,
        FeatureType::Binary => DataType::Binary,
        FeatureType::String => DataType::Utf8,
        FeatureType::Tensor { shape, dtype } => {
            let list_size = shape.iter().map(|x| *x as i32).product();
            let inner_type = match dtype {
                TensorDataType::DtBfloat16 => DataType::FixedSizeBinary(2),
                TensorDataType::DtHalf => DataType::Float16,
                TensorDataType::DtFloat => DataType::Float32,
                TensorDataType::DtDouble => DataType::Float64,
                _ => {
                    return Err(Error::IO {
                        message: format!("unsupported tensor data type {:?}", dtype),
                    });
                }
            };

            let inner_meta = match dtype {
                TensorDataType::DtBfloat16 => Some(
                    [("ARROW:extension:name", "lance.bfloat16")]
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect::<HashMap<String, String>>(),
                ),
                _ => None,
            };
            let mut inner_field = ArrowField::new("value", inner_type, false);
            if let Some(metadata) = inner_meta {
                inner_field.set_metadata(metadata);
            }

            DataType::FixedSizeList(Arc::new(inner_field), list_size)
        }
    };

    let metadata = match &feature_meta.feature_type {
        FeatureType::Tensor { shape, dtype } => {
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

/// Infer the Arrow schema from a TFRecord file.
///
/// The featured named by `tensor_features` will be assumed to be binary fields
/// containing serialized tensors (TensorProto messages). Currently only
/// fixed-shape tensors are supported.
///
/// The features named by `string_features` will be assumed to be UTF-8 encoded
/// strings.
pub async fn infer_tfrecord_schema(
    uri: &str,
    tensor_features: &[&str],
    string_features: &[&str],
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
    while let Some(record) = records.next().await {
        let record = record.map_err(|err| Error::IO {
            message: err.to_string(),
        })?;

        if let Some(features) = record.features {
            for (name, feature) in features.feature {
                if let Some(entry) = columns.get_mut(&name) {
                    entry.try_update(&feature)?;
                } else {
                    columns.insert(
                        name.clone(),
                        FeatureMeta::new(
                            &feature,
                            tensor_features.contains(&name.as_str()),
                            string_features.contains(&name.as_str()),
                        ),
                    );
                }
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
/// Reads 10k rows at a time.
///
/// The schema may be a partial schema, in which case only the fields present in
/// the schema will be read.
pub async fn read_tfrecord(
    uri: &str,
    schema: Arc<ArrowSchema>,
) -> Result<SendableRecordBatchStream> {
    let batch_size = 10_000;

    let (store, path) = ObjectStore::from_uri(uri).await?;
    let data = store
        .inner
        .get(&path)
        .await?
        .into_stream()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
        .into_async_read();
    let records = RecordStream::<Example, _>::from_reader(data, Default::default());

    let builder = TFRecordBatchBuilder::new(schema.clone());
    let batch_stream = futures::stream::try_unfold(
        (records, builder),
        move |(mut records, mut builder)| async move {
            if let Some(record) = records.next().await {
                let record = record.map_err(|err| Error::IO {
                    message: err.to_string(),
                })?;
                builder.append(record)?;
                if builder.num_rows() == batch_size {
                    // A batch is ready
                    let batch = builder.finish()?;
                    Ok(Some((Some(batch), (records, builder))))
                } else {
                    Ok(Some((None, (records, builder))))
                }
            } else {
                if builder.num_rows() > 0 {
                    // output the last (partial) batch
                    let batch = builder.finish()?;
                    Ok(Some((Some(batch), (records, builder))))
                } else {
                    Ok(None)
                }
            }
        },
    )
    .filter_map(|x| async { x.transpose() });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        batch_stream,
    )))
}

struct TFRecordBatchBuilder {
    schema: Arc<ArrowSchema>,
    builders: Vec<Box<dyn ArrayBuilder>>,
    /// For fields that are lists, we also have offset builders
    offset_builders: Vec<Option<Vec<i32>>>,
    num_rows: usize,
}

impl TFRecordBatchBuilder {
    pub fn new(schema: Arc<ArrowSchema>) -> Self {
        // make_builder is not implemented for list types yet, so we handle
        // offsets manually.
        let mut builders = Vec::with_capacity(schema.fields.len());
        let mut offset_builders = Vec::with_capacity(schema.fields.len());

        for field in schema.fields() {
            let leaf_type = match field.data_type() {
                DataType::List(inner_field) => {
                    offset_builders.push(Some(vec![0]));
                    match inner_field.data_type() {
                        DataType::FixedSizeList(inner_field, _) => inner_field.data_type(),
                        _ => inner_field.data_type(),
                    }
                }
                DataType::FixedSizeList(inner_field, _) => {
                    offset_builders.push(None);
                    inner_field.data_type()
                }
                _ => {
                    offset_builders.push(None);
                    field.data_type()
                }
            };
            builders.push(make_builder(leaf_type, 1024));
        }

        Self {
            schema,
            builders,
            offset_builders,
            num_rows: 0,
        }
    }

    pub fn append(&mut self, record: Example) -> Result<()> {
        let features = record.features.unwrap().feature;

        for (i, field) in self.schema.fields().iter().enumerate() {
            // TODO: right now we have to downcast the builder for every cell, which
            // might be inefficient.
            if let Some(feature) = features.get(field.name()) {
                Self::append_values(field, &mut self.builders[i], feature)?;
                Self::append_offsets(field, &mut self.offset_builders[i], feature)?;
            } else {
                Self::append_nulls(field, &mut self.builders[i], &mut self.offset_builders[i])?;
            }
        }

        self.num_rows += 1;
        Ok(())
    }

    fn append_values(
        field: &ArrowField,
        builder: &mut dyn ArrayBuilder,
        feature: &tfrecord::Feature,
    ) -> Result<()> {
        todo!()
    }

    fn append_offsets(
        field: &ArrowField,
        builder: &mut Option<Vec<i32>>,
        feature: &tfrecord::Feature,
    ) -> Result<()> {
        todo!()
    }

    fn append_nulls(
        field: &ArrowField,
        builder: &mut dyn ArrayBuilder,
        offset_builder: &mut Option<Vec<i32>>,
    ) -> Result<()> {
        let num_null_values = match field.data_type() {
            DataType::List(inner_field) => match inner_field.data_type() {
                DataType::FixedSizeList(_, size) => *size as usize,
                _ => 1,
            },
            DataType::FixedSizeList(_, size) => *size as usize,
            _ => 1,
        };
        // TODO: append nulls
        builder.append_nulls(num_null_values);
        if let Some(offset_builder) = offset_builder {
            offset_builder.push(*offset_builder.last().unwrap());
        }
        Ok(())
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        // Handle the list and fixed-size list types
        let mut columns = self
            .builders
            .iter_mut()
            .map(|builder| builder.finish())
            .collect::<Vec<_>>();
        let offsets = self
            .offset_builders
            .iter_mut()
            .map(|builder| {
                let data = builder.take();
                data.map(|builder| OffsetBuffer::new(ScalarBuffer::from(builder)))
            })
            .collect::<Vec<_>>();

        for (column, offsets) in columns.iter_mut().zip(offsets.into_iter()) {
            match column.data_type() {
                DataType::List(inner_field) => match inner_field.data_type() {
                    DataType::FixedSizeList(fsl_field, size) => {
                        let fsl_column = FixedSizeListArray::try_new(
                            fsl_field.clone(),
                            *size,
                            column.clone(),
                            None,
                        )?;
                        *column = Arc::new(ListArray::try_new(
                            inner_field.clone(),
                            offsets.unwrap(),
                            Arc::new(fsl_column),
                            None,
                        )?);
                    }
                    _ => {
                        *column = Arc::new(ListArray::try_new(
                            inner_field.clone(),
                            offsets.unwrap(),
                            column.clone(),
                            None,
                        )?);
                    }
                },
                DataType::FixedSizeList(fsl_field, size) => {
                    *column = Arc::new(FixedSizeListArray::try_new(
                        fsl_field.clone(),
                        *size,
                        column.clone(),
                        None,
                    )?);
                }
                _ => {} // Leave column as-is
            }
        }
        let batch = RecordBatch::try_new(self.schema.clone(), columns)?;
        self.num_rows = 0;
        Ok(batch)
    }
}

fn append_feature(
    field: &ArrowField,
    builder: &mut dyn ArrayBuilder,
    feature: &tfrecord::Feature,
) -> Result<()> {
    match (field.data_type(), feature.kind.as_ref().unwrap()) {
        (DataType::Binary, tfrecord::protobuf::feature::Kind::BytesList(bytes_list)) => {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<BinaryBuilder>()
                .unwrap();
            builder.append_value(&bytes_list.value[0]);
        }
        (_, _) => {
            return Err(Error::IO {
                message: format!("invalid feature type for field {}", field.name()),
            });
        }
    }

    Ok(())
}

/// Convert TensorProto message into an element of a FixedShapeTensor array and
/// append it to the builder.
///
/// TensorProto definition:
/// https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor.proto
///
/// FixedShapeTensor definition:
/// https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
pub fn append_fixedshape_tensor(
    field: &ArrowField,
    builder: &dyn ArrayBuilder,
    tensor: &TensorProto,
) -> Result<()> {
    todo!()
}
