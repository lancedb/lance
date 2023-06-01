// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Testing utilities

use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::iter::repeat_with;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, RecordBatchReader};

use crate::arrow::{FixedSizeListArrayExt, RecordBatchBuffer};
use crate::dataset::{Dataset, WriteMode, WriteParams};
use crate::index::vector::ivf::IvfBuildParams;
use crate::index::vector::pq::PQBuildParams;
use crate::index::vector::{MetricType, VectorIndexParams};
use crate::index::{DatasetIndexExt, IndexType};

/// Create a random float32 array.
pub fn generate_random_array_with_seed(n: usize, seed: [u8; 32]) -> Float32Array {
    let mut rng = StdRng::from_seed(seed);
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}

/// Create a random float32 array.
pub fn generate_random_array(n: usize) -> Float32Array {
    let mut rng = rand::thread_rng();
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}

pub async fn create_file(
    path: &std::path::Path,
    mode: WriteMode,
    dims: i32,
    num_rows: i32,
    batch_size: i32,
    seed: [u8; 32],
) {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            dims,
        ),
        false,
    )]));

    let batches = RecordBatchBuffer::new(
        (0..(num_rows / batch_size) as i32)
            .map(|_| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(
                        FixedSizeListArray::try_new(
                            create_float32_array_with_seed(num_rows * dims, seed),
                            dims,
                        )
                        .unwrap(),
                    )],
                )
                .unwrap()
            })
            .collect(),
    );

    let test_uri = path.to_str().unwrap();
    std::fs::remove_dir_all(test_uri).map_or_else(|_| println!("{} not exists", test_uri), |_| {});
    let mut write_params = WriteParams::default();
    write_params.max_rows_per_file = num_rows as usize;
    write_params.max_rows_per_group = batch_size as usize;
    write_params.mode = mode;
    let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
    Dataset::write(&mut reader, test_uri, Some(write_params))
        .await
        .unwrap();
}

pub async fn create_vector_index(
    path: &str,
    column_name: &str,
    index_name: &str,
    num_partitions: usize,
    num_bits: usize,
    num_sub_vectors: usize,
    use_opq: bool,
    m_type: MetricType,
) {
    let dataset = Dataset::open(path).await.unwrap();
    let mut ivf_params = IvfBuildParams::default();
    ivf_params.num_partitions = num_partitions;
    let mut pq_params = PQBuildParams::default();
    pq_params.num_bits = num_bits;
    pq_params.num_sub_vectors = num_sub_vectors;
    pq_params.use_opq = use_opq;
    let m_type = m_type;
    let params = VectorIndexParams::with_ivf_pq_params(m_type, ivf_params, pq_params);
    dataset
        .create_index(
            vec![column_name].as_slice(),
            IndexType::Vector,
            Some(index_name.to_string()),
            &params,
        )
        .await
        .unwrap();
}

fn create_float32_array_with_seed(num_elements: i32, seed: [u8; 32]) -> Float32Array {
    // generate an Arrow Float32Array with 10000*128 elements randomly
    let mut rng = rand::rngs::StdRng::from_seed(seed);
    let mut values = Vec::with_capacity(num_elements as usize);
    for _ in 0..num_elements {
        values.push(rng.gen_range(0.0..1.0));
    }
    Float32Array::from(values)
}
