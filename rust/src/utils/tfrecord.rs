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
use arrow_array::builder::ArrayBuilder;
use arrow_schema::Schema as ArrowSchema;
use object_store::path::Path;
use tfrecord::protobuf::TensorProto;
use tfrecord::record_reader::RecordIter;
use tfrecord::Example;

pub async fn read_tfrecord(uri: &str, schema: ArrowSchema) -> Result<RecordBatch> {
    // TODO: Use schema to create builders

    let (store, path) = ObjectStore::from_uri(uri).await?;
    // TODO: should we avoid reading the entire file into memory?
    let data = store.inner.get(&path).await?.bytes().await?;
    let records = RecordIter::<Example, _>::from_reader(data.as_ref(), Default::default());

    for record in records {
        let record = record.map_err(|err| Error::IO {
            message: err.to_string(),
        })?;
        if let Some(features) = record.features {
            for field in &schema.fields {
                if let Some(feature) = features.feature.get(field.name()) {
                    todo!("append feature to appropriate builder")
                }
            }
        }
    }

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

/// Infer the Arrow schema from a TFRecord file.
///
/// The featured named by `tensor_features` will be assumed to be binary fields
/// containing serialized tensors (TensorProto messages). Currently only
/// fixed-shape tensors are supported.
pub async fn infer_tfrecord_schema(uri: &str, tensor_features: &[&str]) -> Result<ArrowSchema> {
    todo!()
}

/// Convert TensorProto message into an element of a FixedShapeTensor array and
/// append it to the builder.
///
/// TensorProto definition:
/// https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor.proto
///
/// FixedShapeTensor definition:
/// https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor
pub fn append_fixedshape_tensor(builder: &dyn ArrayBuilder, tensor: &TensorProto) -> Result<()> {
    todo!()
}
