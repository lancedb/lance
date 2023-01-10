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

use std::vec;

use arrow_array::{cast::as_primitive_array, Array, FixedSizeListArray, Float32Array, UInt32Array};
use arrow_schema::DataType;

use crate::arrow::*;
use crate::index::ann::distance::euclidean_distance;
use crate::Result;

use faiss::cluster::kmeans_clustering;

pub fn train_kmean_model_from_array(
    array: &Float32Array,
    dimension: usize,
    nclusters: u32,
    max_iterations: u32,
) -> Result<Vec<f32>> {
    let results = kmeans_clustering(
        dimension as u32,
        nclusters,
        array.data_ref().buffers()[0].typed_data(),
    )
    .unwrap();

    Ok(results.centroids)
}

pub struct KMean {
}

impl KMean {
    /// Fit (train) a KMean model and transform the input data into cluster IDs.
    pub fn fit_and_transform(
        data: &FixedSizeListArray,
        num_clusters: usize,
        max_iterations: usize,
    ) -> Result<(Vec<f32>, UInt32Array)> {
        assert_eq!(data.value_type(), DataType::Float32);
        let dimension = data.value_length() as usize;

        let value_arr = data.values();
        let centroids = train_kmean_model_from_array(
            as_primitive_array(&value_arr),
            dimension,
            num_clusters as u32,
            max_iterations as u32,
        )?;
        let f32_values = Float32Array::from(centroids.clone());
        let cent = FixedSizeListArray::try_new(&f32_values, dimension as i32).unwrap();

        let partition_ids = UInt32Array::from(vec![0_u32]);
        for i in 0..data.len() {
            let vector = data.value(i);

            let id = argmin(
                euclidean_distance(as_primitive_array(&vector), &cent)
                    .unwrap()
                    .as_ref(),
            )
            .unwrap() as u32;
        }
        Ok((centroids, partition_ids))
    }
}
