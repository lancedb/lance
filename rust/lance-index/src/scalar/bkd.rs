// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! BKD Tree (Block K-Dimensional Tree) implementation
//!
//! A BKD tree is a spatial index structure for efficiently indexing and querying
//! multi-dimensional points. It's similar to a KD-tree but optimized for disk storage
//! by grouping multiple points into leaf blocks.
//!
//! ## Algorithm
//!
//! Based on Lucene's BKD tree implementation:
//! - Recursively splits points by alternating dimensions (X, Y)
//! - Splits at median to create balanced tree
//! - Groups points into leaves of configurable size
//! - Stores tree structure separately from leaf data for lazy loading

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Field, Schema};
use deepsize::DeepSizeOf;
use lance_core::{Result, ROW_ID};
use std::sync::Arc;

/// BKD Tree node representing either an inner node or a leaf
#[derive(Debug, Clone, DeepSizeOf)]
pub struct BKDNode {
    /// Bounding box: [min_x, min_y, max_x, max_y]
    pub bounds: [f64; 4],
    /// Split dimension: 0=X, 1=Y
    pub split_dim: u8,
    /// Split value along split_dim
    pub split_value: f64,
    /// Left child node index (for inner nodes)
    pub left_child: Option<u32>,
    /// Right child node index (for inner nodes)
    pub right_child: Option<u32>,
    /// Leaf ID - corresponds to leaf_{id}.lance file (for leaf nodes)
    pub leaf_id: Option<u32>,
}

/// In-memory BKD tree structure for efficient spatial queries
#[derive(Debug, DeepSizeOf)]
pub struct BKDTreeLookup {
    pub nodes: Vec<BKDNode>,
    pub root_id: u32,
    pub num_leaves: u32,
}

impl BKDTreeLookup {
    pub fn new(nodes: Vec<BKDNode>, root_id: u32, num_leaves: u32) -> Self {
        Self {
            nodes,
            root_id,
            num_leaves,
        }
    }

    /// Find all leaf IDs that intersect with the query bounding box
    pub fn find_intersecting_leaves(&self, query_bbox: [f64; 4]) -> Vec<u32> {
        let mut leaf_ids = Vec::new();
        let mut stack = vec![self.root_id];
        let mut nodes_visited = 0;

        while let Some(node_id) = stack.pop() {
            if node_id as usize >= self.nodes.len() {
                continue;
            }

            let node = &self.nodes[node_id as usize];
            nodes_visited += 1;

            // Check if node's bounding box intersects with query bbox
            if !bboxes_intersect(&node.bounds, &query_bbox) {
                continue;
            }

            // If this is a leaf node, add its leaf_id
            if let Some(leaf_id) = node.leaf_id {
                println!(
                    "  🍃 Found intersecting leaf_{}.lance: bounds {:?}",
                    leaf_id,
                    node.bounds
                );
                leaf_ids.push(leaf_id);
            } else {
                // Inner node - traverse children
                if let Some(left) = node.left_child {
                    stack.push(left);
                }
                if let Some(right) = node.right_child {
                    stack.push(right);
                }
            }
        }

        println!(
            "🌲 Tree traversal: visited {} nodes, found {} intersecting leaves",
            nodes_visited,
            leaf_ids.len()
        );

        leaf_ids
    }

    /// Deserialize from RecordBatch
    pub fn from_record_batch(batch: RecordBatch) -> Result<Self> {
        if batch.num_rows() == 0 {
            return Ok(Self::new(vec![], 0, 0));
        }

        let min_x = batch
            .column(0)
            .as_primitive::<arrow_array::types::Float64Type>();
        let min_y = batch
            .column(1)
            .as_primitive::<arrow_array::types::Float64Type>();
        let max_x = batch
            .column(2)
            .as_primitive::<arrow_array::types::Float64Type>();
        let max_y = batch
            .column(3)
            .as_primitive::<arrow_array::types::Float64Type>();
        let split_dim = batch
            .column(4)
            .as_primitive::<arrow_array::types::UInt8Type>();
        let split_value = batch
            .column(5)
            .as_primitive::<arrow_array::types::Float64Type>();
        let left_child = batch
            .column(6)
            .as_primitive::<arrow_array::types::UInt32Type>();
        let right_child = batch
            .column(7)
            .as_primitive::<arrow_array::types::UInt32Type>();
        let leaf_id = batch
            .column(8)
            .as_primitive::<arrow_array::types::UInt32Type>();

        let mut nodes = Vec::with_capacity(batch.num_rows());
        let mut num_leaves = 0;

        for i in 0..batch.num_rows() {
            let leaf_id_val = if leaf_id.is_null(i) {
                None
            } else {
                num_leaves += 1;
                Some(leaf_id.value(i))
            };

            nodes.push(BKDNode {
                bounds: [
                    min_x.value(i),
                    min_y.value(i),
                    max_x.value(i),
                    max_y.value(i),
                ],
                split_dim: split_dim.value(i),
                split_value: split_value.value(i),
                left_child: if left_child.is_null(i) {
                    None
                } else {
                    Some(left_child.value(i))
                },
                right_child: if right_child.is_null(i) {
                    None
                } else {
                    Some(right_child.value(i))
                },
                leaf_id: leaf_id_val,
            });
        }

        Ok(Self::new(nodes, 0, num_leaves))
    }
}

/// Check if two bounding boxes intersect
pub fn bboxes_intersect(bbox1: &[f64; 4], bbox2: &[f64; 4]) -> bool {
    // bbox format: [min_x, min_y, max_x, max_y]
    !(bbox1[2] < bbox2[0] || bbox1[0] > bbox2[2] || bbox1[3] < bbox2[1] || bbox1[1] > bbox2[3])
}

/// Check if a point is within a bounding box
pub fn point_in_bbox(x: f64, y: f64, bbox: &[f64; 4]) -> bool {
    x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3]
}

/// BKD Tree builder following Lucene's bulk-loading algorithm
pub struct BKDTreeBuilder {
    leaf_size: usize,
}

impl BKDTreeBuilder {
    pub fn new(leaf_size: usize) -> Self {
        Self { leaf_size }
    }

    /// Build a BKD tree from points
    /// Returns (tree_nodes, leaf_batches)
    pub fn build(&self, points: &mut [(f64, f64, u64)]) -> Result<(Vec<BKDNode>, Vec<RecordBatch>)> {
        if points.is_empty() {
            return Ok((vec![], vec![]));
        }

        println!(
            "\n🏗️  Building BKD tree for {} points with leaf size {}",
            points.len(),
            self.leaf_size
        );

        // Log first few points for debugging
        println!("📍 First 5 points:");
        for i in 0..std::cmp::min(5, points.len()) {
            println!("  Point {}: x={}, y={}, row_id={}", i, points[i].0, points[i].1, points[i].2);
        }

        let mut leaf_counter = 0u32;
        let mut all_nodes = Vec::new();
        let mut all_leaf_batches = Vec::new();

        self.build_recursive(
            points,
            0, // depth
            &mut leaf_counter,
            &mut all_nodes,
            &mut all_leaf_batches,
        )?;

        println!(
            "✅ Built BKD tree: {} nodes ({} leaves)\n",
            all_nodes.len(),
            leaf_counter
        );

        Ok((all_nodes, all_leaf_batches))
    }

    /// Recursively build BKD tree following Lucene's algorithm
    fn build_recursive(
        &self,
        points: &mut [(f64, f64, u64)],
        depth: u32,
        leaf_counter: &mut u32,
        all_nodes: &mut Vec<BKDNode>,
        all_leaf_batches: &mut Vec<RecordBatch>,
    ) -> Result<u32> {
        // Base case: create leaf node
        if points.len() <= self.leaf_size {
            let node_id = all_nodes.len() as u32;
            let leaf_id = *leaf_counter;
            *leaf_counter += 1;

            // Calculate bounding box for this leaf
            let (min_x, min_y, max_x, max_y) = calculate_bounds(points);

            // Create leaf batch
            let leaf_batch = create_leaf_batch(points)?;
            all_leaf_batches.push(leaf_batch);

            // Create leaf node
            all_nodes.push(BKDNode {
                bounds: [min_x, min_y, max_x, max_y],
                split_dim: 0,
                split_value: 0.0,
                left_child: None,
                right_child: None,
                leaf_id: Some(leaf_id),
            });

            return Ok(node_id);
        }

        // Recursive case: split and build subtrees
        let split_dim = (depth % 2) as u8; // Alternate between X (0) and Y (1)

        // Sort points by the split dimension
        if split_dim == 0 {
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Split at median
        let mid = points.len() / 2;
        let split_value = if split_dim == 0 {
            points[mid].0
        } else {
            points[mid].1
        };

        // Calculate bounds for this node (before splitting the slice)
        let (min_x, min_y, max_x, max_y) = calculate_bounds(points);

        // Reserve space for this inner node
        let node_id = all_nodes.len() as u32;
        all_nodes.push(BKDNode {
            bounds: [min_x, min_y, max_x, max_y],
            split_dim,
            split_value,
            left_child: None,
            right_child: None,
            leaf_id: None,
        });

        // Recursively build left and right subtrees
        let (left_points, right_points) = points.split_at_mut(mid);

        let left_child_id = self.build_recursive(
            left_points,
            depth + 1,
            leaf_counter,
            all_nodes,
            all_leaf_batches,
        )?;

        let right_child_id = self.build_recursive(
            right_points,
            depth + 1,
            leaf_counter,
            all_nodes,
            all_leaf_batches,
        )?;

        // Update the inner node with child pointers
        all_nodes[node_id as usize].left_child = Some(left_child_id);
        all_nodes[node_id as usize].right_child = Some(right_child_id);

        Ok(node_id)
    }
}

/// Calculate bounding box for a set of points
fn calculate_bounds(points: &[(f64, f64, u64)]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for (x, y, _) in points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    (min_x, min_y, max_x, max_y)
}

/// Create a leaf batch from points
fn create_leaf_batch(points: &[(f64, f64, u64)]) -> Result<RecordBatch> {
    let x_vals: Vec<f64> = points.iter().map(|(x, _, _)| *x).collect();
    let y_vals: Vec<f64> = points.iter().map(|(_, y, _)| *y).collect();
    let row_ids: Vec<u64> = points.iter().map(|(_, _, r)| *r).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new(ROW_ID, DataType::UInt64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(x_vals)) as ArrayRef,
            Arc::new(Float64Array::from(y_vals)) as ArrayRef,
            Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

