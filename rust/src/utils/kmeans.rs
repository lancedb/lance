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

use std::collections::HashSet;
use std::sync::Arc;

use arrow_arith::arithmetic::{add, divide_scalar};
use arrow_array::{
    cast::as_primitive_array, new_empty_array, Array, FixedSizeListArray, Float32Array,
};
use arrow_array::{ArrayAccessor, PrimitiveArray};
use arrow_schema::{DataType, Field as ArrowField};
use arrow_select::concat::concat;
use futures::stream::{self, StreamExt};
use rand::prelude::*;
use rand::{distributions::WeightedIndex, Rng, RngCore};

use super::distance::l2_distance;
use crate::arrow::*;
use crate::{Error, Result};

#[derive(Debug)]
pub struct KMeansParams {
    /// Max number of iterations.
    max_iters: u32,

    /// When the difference of mean distance to the centroids is less than this `tolerance`
    /// threshold, stop the training.
    tolerance: f32,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-2,
        }
    }
}

/// KMeans implementation for Apache Arrow Arrays.
#[derive(Debug)]
pub struct KMeans {
    /// Centroids for each of the k clusters.
    ///
    /// k * dimension.
    pub centroids: Arc<FixedSizeListArray>,

    /// The number of clusters
    pub k: u32,
}

/// Initialize using kmean++, and returns the centroids of k clusters.
fn kmean_plusplus(
    data: Arc<FixedSizeListArray>,
    k: u32, /* dist_fn, rand_seed */
    rng: &mut dyn RngCore,
) -> KMeans {
    assert!(data.len() > k as usize);
    let mut kmeans = KMeans {
        centroids: new_empty_array(&DataType::FixedSizeList(
            Box::new(ArrowField::new("item", DataType::Float32, false)),
            data.value_length(),
        ))
        .as_any()
        .downcast_ref::<Arc<FixedSizeListArray>>()
        .unwrap()
        .clone(),
        k,
    };

    let dimension = data.value_length();
    let first = rng.gen_range(0..data.len());

    let vector = data.value(first);
    kmeans.centroids = FixedSizeListArray::try_new(vector, dimension)
        .unwrap()
        .into();

    let mut seen = HashSet::new();
    seen.insert(first);
    for _ in 1..k {
        let membership = kmeans.compute_membership(data.clone());
        let weights = WeightedIndex::new(&membership.distances).unwrap();
        let mut chosen;
        loop {
            chosen = weights.sample(rng);
            if !seen.contains(&chosen) {
                seen.insert(chosen);
                break;
            }
        }
        let vector = data.value(chosen);
        let new_vector: &Float32Array = as_primitive_array(&vector);

        let centroids_array = kmeans.centroids.values();
        let centroids_f32_array: &Float32Array = as_primitive_array(&centroids_array);
        let new_centroid_values = Float32Array::from_iter_values(
            centroids_f32_array
                .values()
                .iter()
                .copied()
                .chain(new_vector.values().iter().copied()),
        );
        kmeans.centroids =
            Arc::new(FixedSizeListArray::try_new(new_centroid_values, dimension).unwrap());
    }
    kmeans
}

struct KMeanMembership {
    /// Reference to the input vectors.
    data: Arc<FixedSizeListArray>,

    /// Cluster Id for each vector.
    cluster_ids: Vec<u32>,

    /// Distance between each vector, to its corresponding centroids.
    distances: Vec<f32>,

    k: u32,
}

impl TryFrom<&KMeanMembership> for KMeans {
    type Error = Error;

    fn try_from(membership: &KMeanMembership) -> Result<Self> {
        let dimension = membership.data.value_length();
        let means = tokio::runtime::Runtime::new().unwrap().block_on(async {
            stream::iter(0..membership.k)
                .map(|cluster| {
                    membership
                        .cluster_ids
                        .iter()
                        .filter(|id| **id == cluster)
                        .fold(
                            (
                                Float32Array::from_iter_values(
                                    (0..dimension).map(|_| 0.0).collect::<Vec<_>>(),
                                ),
                                0.0,
                            ),
                            |(arr, total), idx| {
                                (
                                    add(
                                        &arr,
                                        as_primitive_array(
                                            membership.data.value(*idx as usize).as_ref(),
                                        ),
                                    )
                                    .unwrap(),
                                    total + 1.0,
                                )
                            },
                        )
                })
                .map(|(arr, total)| divide_scalar(&arr, total).unwrap())
                .collect::<Vec<_>>()
                .await
        });

        let mut mean_refs: Vec<&dyn Array> = vec![];
        for m in means.iter() {
            mean_refs.push(m);
        }
        let centroids = concat(&mean_refs).unwrap();
        Ok(KMeans {
            centroids: Arc::new(FixedSizeListArray::try_new(centroids, dimension)?),
            k: membership.k,
        })
    }
}

impl KMeans {
    /// Train a KMean on data with `k` clusters.
    pub fn new(data: Arc<FixedSizeListArray>, k: u32, max_iters: u32) -> Self {
        let mut params = KMeansParams::default();
        params.max_iters = max_iters;
        KMeans::new_with_params(data, k, &params)
    }

    pub fn new_with_params(data: Arc<FixedSizeListArray>, k: u32, params: &KMeansParams) -> Self {
        let mut kmeans = kmean_plusplus(data.clone(), k, &mut rand::thread_rng());

        let mut last_membership = kmeans.compute_membership(data.clone());
        for _ in 0..params.max_iters {
            let new_kmeans: KMeans = (&last_membership).try_into().unwrap();
            let new_membership = new_kmeans.compute_membership(data.clone());
            if (new_membership.distances.iter().sum::<f32>()
                - last_membership.distances.iter().sum::<f32>())
                / last_membership.distances.iter().sum::<f32>()
                < params.tolerance
            {
                break;
            }
            kmeans = new_kmeans;
            last_membership = new_membership;
        }
        kmeans
    }

    fn compute_membership(&self, data: Arc<FixedSizeListArray>) -> KMeanMembership {
        let cluster_with_distances = (0..data.len())
            .map(|idx| {
                let value_arr = data.value(idx);
                let vector: &Float32Array = as_primitive_array(&value_arr);
                let distances = l2_distance(vector, self.centroids.as_ref()).unwrap();
                let cluster_id = argmin(distances.as_ref()).unwrap();
                let distance = distances.value(cluster_id as usize);
                (cluster_id, distance)
            })
            .collect::<Vec<_>>();
        let k = self.k;
        KMeanMembership {
            data: data.clone(),
            cluster_ids: cluster_with_distances.iter().map(|(c, _)| *c).collect(),
            distances: cluster_with_distances.iter().map(|(_, d)| *d).collect(),
            k,
        }
    }
}

#[cfg(test)]
mod tests {

}