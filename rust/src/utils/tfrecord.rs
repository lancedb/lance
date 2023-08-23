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

use crate::error::{Error, Result};
use crate::io::ObjectStore;
use arrow::record_batch::RecordBatch;
use arrow_array::builder::{make_builder, ArrayBuilder, BinaryBuilder};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use tfrecord::protobuf::TensorProto;
use tfrecord::record_reader::RecordIter;
use tfrecord::Example;

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
    todo!()
}

pub async fn read_tfrecord(uri: &str, schema: ArrowSchema) -> Result<RecordBatch> {
    let mut builders = schema
        .fields
        .iter()
        .map(|field| make_builder(field.data_type(), 10))
        .collect::<Vec<_>>();

    let (store, path) = ObjectStore::from_uri(uri).await?;
    // TODO: should we avoid reading the entire file into memory?
    let data = store.inner.get(&path).await?.bytes().await?;
    let records = RecordIter::<Example, _>::from_reader(data.as_ref(), Default::default());

    for record in records {
        let record = record.map_err(|err| Error::IO {
            message: err.to_string(),
        })?;
        if let Some(features) = record.features {
            for (i, field) in schema.fields.iter().enumerate() {
                let builder = builders.get_mut(i).unwrap();
                if let Some(feature) = features.feature.get(field.name()) {
                    todo!("append feature to appropriate builder")
                } else {
                    // TODO: do we care about nulls / missing values?
                    return Err(Error::IO {
                        message: format!("missing feature {}", field.name()),
                    });
                }
            }
        }
    }

    // Start with:
    // * string
    // * fixed shape tensor
    // * int64
    // * float32
    // * binary list

    // TODO: support reading:
    // Binary
    // Binary list
    // int64
    // int64 list
    // float32
    // float32 list
    // Then also tensors based on https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor.proto

    todo!()
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
