// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! KMeans spill-tree implementation for Apache Arrow Arrays.
//!
//! Support ``cosine`` distances only for now

use crate::vector::flat::storage::FlatStorage;
use crate::vector::graph::OrderedFloat;
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::VECTOR_RESULT_SCHEMA;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use core::cmp::min;
use lance_core::{Error, Result};
use snafu::{location, Location};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use {
    lance_linalg::distance::DistanceType,
    lance_linalg::kmeans::{compute_partitions_arrow_array, KMeans},
};

#[derive(Debug, Clone)]
pub struct KMeansTreeParams {
    /// (Max) number of clusters per node
    pub k: usize,

    /// Number of layers to generate
    pub num_layers: usize,

    /// Amount of spilling to do
    pub spill_count: usize,

    /// Maximum number of iterations for each k-means
    pub max_iters: u32,

    /// The distance metric used for clustering (currently only Cosine supported).
    pub distance_type: DistanceType,
}

/// KMeansTree implementation for hierarchical k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansTree {
    /// TODO: Temporary for development & testing: All the data stored in memory
    pub data: Vec<f32>,
    pub dimension: usize,

    /// Parameters for indexing/inserting
    pub params: KMeansTreeParams,

    /// Layers of clusterings, each referring to the next layer
    pub clusterings_per_layer: Vec<Vec<ClusteringNode>>,

    /// Bottom-level clusters, referred to by the last layer of clusterings
    pub leaf_clusters: Vec<LeafCluster>,
}

/// A non-leaf node in the k-means tree.
#[derive(Debug, Clone)]
pub struct ClusteringNode {
    /// Indices of the child nodes in the next layer
    /// During initial build, this does not include spilled nodes
    pub children: Vec<usize>,
    /// For each child, also record the centroid
    pub centroids: FixedSizeListArray,
}

/// A leaf node in the k-means tree.
#[derive(Debug, Clone)]
pub struct LeafCluster {
    /// The offset in the data array for this node
    pub offset: usize,
    /// How many data points are assigned to this node (incl spills)
    pub size: usize,
}

fn chunk_to_fixed_size_list_array(
    float_array: Arc<Float32Array>,
    dimension: i32,
) -> FixedSizeListArray {
    // Define the field for the list elements
    let field = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        true,
    ));

    // Create the FixedSizeListArray
    FixedSizeListArray::try_new(field, dimension, float_array as ArrayRef, None)
        .expect("Unable to create FixedSizeListArray")
}

impl KMeansTree {
    fn search_to_layer(&self, query: ArrayRef, k: usize, max_layer: usize) -> Result<RecordBatch> {
        // let dist_calc = storage.dist_calculator(query);
        let mut best_candidates = vec![(OrderedFloat(0.0), 0)]; // the first layer only ever contains one tree node, whether it's a leaf or not
        for layer in 0..min(max_layer + 1, self.clusterings_per_layer.len()) {
            best_candidates = best_candidates
                .iter()
                // TODO: later, check if putting par_flat_map here nets any perf gains over
                // basic query-wise parallelism
                .flat_map(|&(_centroid_distance, index)| {
                    let clustering = &self.clusterings_per_layer[layer][index];
                    // TODO this could avoid copying with better structs
                    let flat_storage =
                        FlatStorage::new(clustering.centroids.clone(), self.params.distance_type);
                    let dist_calc = flat_storage.dist_calculator(query.clone());
                    (0..clustering.centroids.len())
                        .map(|local_index| {
                            (
                                OrderedFloat(dist_calc.distance(local_index as u32)),
                                clustering.children[local_index],
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .map(Reverse)
                .collect::<BinaryHeap<_>>()
                .into_sorted_vec()
                .into_iter()
                .take(k) // TODO maybe this should be a different query-time parameter, similar to nprobes
                .map(|Reverse(candidate)| candidate)
                .collect();
        }
        if max_layer == self.clusterings_per_layer.len() {
            // TODO update best_candidates with leaves
        }
        // TODO below is a stub, need to format output and complete
        let schema = VECTOR_RESULT_SCHEMA.clone();
        return Ok(RecordBatch::new_empty(schema));
    }
    pub fn search(&self, query: ArrayRef, k: usize) -> Result<RecordBatch> {
        self.search_to_layer(query, k, self.clusterings_per_layer.len())
    }
    fn spill_centroids(&mut self) -> Result<()> {
        // TODO for each layer, spill centroids into other clusterings at the same layer
        // for each centroid, should get a top k candidates for secondary spilling via ANN
        Ok(())
    }
    fn spill_data(&mut self) -> Result<()> {
        // TODO spill data into other leaf clusters
        // - requires updating data array, and editing offsets/sizes of each leaf cluster
        // - to get secondary assignments
        Ok(())
    }
    /// Convert every leaf cluster into a clustering layer
    fn split_leaves(&mut self) -> Result<()> {
        let old_leaf_clusters = self.leaf_clusters.clone();
        self.leaf_clusters = Vec::new();
        self.clusterings_per_layer.push(
            old_leaf_clusters
                .into_iter()
                .map(|leaf: LeafCluster| {
                    let data_slice = chunk_to_fixed_size_list_array(
                        Arc::new(Float32Array::from(
                            self.data[leaf.offset..leaf.offset + leaf.size].to_vec(),
                        )),
                        self.dimension as i32,
                    );
                    // let data_slice = self.data.slice(leaf.offset, leaf.offset + leaf.size);
                    let k = min(leaf.size, self.params.k); // don't make empty clusters
                    let kmeans = KMeans::new(&data_slice, k, self.params.max_iters).unwrap();

                    let centroids = chunk_to_fixed_size_list_array(
                        Arc::new(
                            kmeans
                                .centroids
                                .as_any()
                                .downcast_ref::<Float32Array>()
                                .unwrap()
                                .clone(),
                        ),
                        self.dimension as i32,
                    );
                    let partitions = compute_partitions_arrow_array(
                        &centroids,
                        &data_slice,
                        self.params.distance_type,
                    )
                    .unwrap();

                    let mut children = Vec::with_capacity(k);

                    let mut partitioned_indices: Vec<Vec<usize>> = vec![Vec::new(); k];

                    for (i, partition) in partitions.iter().enumerate() {
                        if let Some(cluster_idx) = partition {
                            partitioned_indices[*cluster_idx as usize].push(i);
                        }
                    }

                    let mut new_data = Vec::with_capacity(data_slice.len());
                    let mut current_offset = leaf.offset;
                    for (_cluster_idx, indices) in partitioned_indices.iter().enumerate() {
                        if !indices.is_empty() {
                            let partition_size = indices.len();

                            // Copy the corresponding data points into the new_data vector
                            for &index in indices {
                                new_data.push(data_slice.value(index).clone());
                            }

                            self.leaf_clusters.push(LeafCluster {
                                offset: current_offset,
                                size: partition_size,
                            });
                            children.push(self.leaf_clusters.len() - 1);
                            current_offset += partition_size;
                        }
                    }
                    for (i, value) in new_data.iter().enumerate() {
                        self.data[(i + leaf.offset) * self.dimension
                            ..(i + leaf.offset + 1) * self.dimension]
                            .copy_from_slice(
                                value
                                    .as_any()
                                    .downcast_ref::<Float32Array>()
                                    .unwrap()
                                    .values(),
                            );
                    }

                    ClusteringNode {
                        children,
                        centroids,
                    }
                })
                .collect(),
        );
        Ok(())
    }
    fn build(&mut self) -> Result<()> {
        for _layer in 0..self.params.num_layers {
            self.split_leaves()?;
        }
        self.spill_centroids()?;
        self.spill_data()?;
        Ok(())
    }
    /// Create a new KMeansTree with the specified parameters.
    pub fn new(data: &FixedSizeListArray, params: KMeansTreeParams) -> Result<Self> {
        // Verify that only Cosine distance is supported for now
        if params.distance_type != DistanceType::Cosine {
            return Err(Error::Index {
                message: "Only DistanceType::Cosine is supported at this time.".to_string(),
                location: location!(),
            });
        }

        // Create the tree with one root node
        let mut tree = KMeansTree {
            data: data
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .to_vec(),
            dimension: data.value_length() as usize,
            params,
            clusterings_per_layer: vec![],
            leaf_clusters: vec![LeafCluster {
                offset: 0,
                size: data.len(),
            }],
        };

        // Split recursively
        tree.build()?;

        Ok(tree)
    }
}

// TODO add unit tests
// -- first layer should contain =1 node
// TODO add IvfSubIndex impl, incl reading/writing
// TODO add VectorIndex impl
