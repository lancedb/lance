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

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array, UInt32Array, UInt8Array};
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Field, Schema};
use deepsize::DeepSizeOf;
use lance_core::{Result, ROW_ID};
use std::sync::Arc;
use snafu::location;

// Schema field names
const NODE_ID: &str = "node_id";
const MIN_X: &str = "min_x";
const MIN_Y: &str = "min_y";
const MAX_X: &str = "max_x";
const MAX_Y: &str = "max_y";
const SPLIT_DIM: &str = "split_dim";
const SPLIT_VALUE: &str = "split_value";
const LEFT_CHILD: &str = "left_child";
const RIGHT_CHILD: &str = "right_child";
const FILE_ID: &str = "file_id";
const ROW_OFFSET: &str = "row_offset";
const NUM_ROWS: &str = "num_rows";

/// Schema for inner node metadata
pub fn inner_node_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(NODE_ID, DataType::UInt32, false),
        Field::new(MIN_X, DataType::Float64, false),
        Field::new(MIN_Y, DataType::Float64, false),
        Field::new(MAX_X, DataType::Float64, false),
        Field::new(MAX_Y, DataType::Float64, false),
        Field::new(SPLIT_DIM, DataType::UInt8, false),
        Field::new(SPLIT_VALUE, DataType::Float64, false),
        Field::new(LEFT_CHILD, DataType::UInt32, false),
        Field::new(RIGHT_CHILD, DataType::UInt32, false),
    ]))
}

/// Schema for leaf node metadata
pub fn leaf_node_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(NODE_ID, DataType::UInt32, false),
        Field::new(MIN_X, DataType::Float64, false),
        Field::new(MIN_Y, DataType::Float64, false),
        Field::new(MAX_X, DataType::Float64, false),
        Field::new(MAX_Y, DataType::Float64, false),
        Field::new(FILE_ID, DataType::UInt32, false),
        Field::new(ROW_OFFSET, DataType::UInt64, false),
        Field::new(NUM_ROWS, DataType::UInt64, false),
    ]))
}

/// BKD Tree node - either an inner node (with children) or a leaf node (with data location)
#[derive(Debug, Clone, DeepSizeOf)]
pub enum BKDNode {
    Inner(BKDInnerNode),
    Leaf(BKDLeafNode),
}

impl BKDNode {
    pub fn bounds(&self) -> [f64; 4] {
        match self {
            BKDNode::Inner(inner) => inner.bounds,
            BKDNode::Leaf(leaf) => leaf.bounds,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, BKDNode::Leaf(_))
    }

    pub fn as_inner(&self) -> Option<&BKDInnerNode> {
        match self {
            BKDNode::Inner(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn as_leaf(&self) -> Option<&BKDLeafNode> {
        match self {
            BKDNode::Leaf(leaf) => Some(leaf),
            _ => None,
        }
    }
}

/// Inner node in BKD tree - contains split information and child pointers
#[derive(Debug, Clone, DeepSizeOf)]
pub struct BKDInnerNode {
    /// Bounding box: [min_x, min_y, max_x, max_y]
    pub bounds: [f64; 4],
    /// Split dimension: 0=X, 1=Y
    pub split_dim: u8,
    /// Split value along split_dim
    pub split_value: f64,
    /// Left child node index
    pub left_child: u32,
    /// Right child node index
    pub right_child: u32,
}

/// Leaf node in BKD tree - contains location of actual point data
#[derive(Debug, Clone, DeepSizeOf)]
pub struct BKDLeafNode {
    /// Bounding box: [min_x, min_y, max_x, max_y]
    pub bounds: [f64; 4],
    /// Which leaf group file this leaf is in
    /// Corresponds to leaf_group_{file_id}.lance
    pub file_id: u32,
    /// Row offset within the leaf group file
    pub row_offset: u64,
    /// Number of rows in this leaf batch
    pub num_rows: u64,
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

    /// Find all leaf nodes that intersect with the query bounding box
    /// Returns references to the intersecting leaf nodes
    pub fn find_intersecting_leaves(&self, query_bbox: [f64; 4]) -> Result<Vec<&BKDLeafNode>> {
        let mut leaves = Vec::new();
        let mut stack = vec![self.root_id];
        let mut nodes_visited = 0;

        while let Some(node_id) = stack.pop() {
            if node_id as usize >= self.nodes.len() {
                continue;
            }

            let node = &self.nodes[node_id as usize];
            nodes_visited += 1;

            // Check if node's bounding box intersects with query bbox
            let intersects = bboxes_intersect(&node.bounds(), &query_bbox);
            if !intersects {
                continue;
            }

            match node {
                BKDNode::Leaf(leaf) => {
                    // Leaf node - add to results
                    leaves.push(leaf);
                }
                BKDNode::Inner(inner) => {
                    // Inner node - traverse children
                    stack.push(inner.left_child);
                    stack.push(inner.right_child);
                }
            }
        }

        println!(
            "🌲 Tree traversal: visited {} nodes, found {} intersecting leaves",
            nodes_visited,
            leaves.len()
        );

        Ok(leaves)
    }

    /// Deserialize from separate inner and leaf RecordBatches  
    /// Both batches include node_id field to preserve tree structure
    pub fn from_record_batches(inner_batch: RecordBatch, leaf_batch: RecordBatch) -> Result<Self> {
        if inner_batch.num_rows() == 0 && leaf_batch.num_rows() == 0 {
            return Ok(Self::new(vec![], 0, 0));
        }

        // Helper to get column by name
        let get_col = |batch: &RecordBatch, name: &str| -> Result<usize> {
            batch.schema().column_with_name(name)
                .map(|(idx, _)| idx)
                .ok_or_else(|| lance_core::Error::Internal {
                    message: format!("Missing column '{}' in BKD tree batch", name),
                    location: location!(),
                })
        };
        
        // Determine total number of nodes (max node_id + 1)
        let max_node_id = {
            let mut max_id = 0u32;
            
            if inner_batch.num_rows() > 0 {
                let col_idx = get_col(&inner_batch, NODE_ID)?;
                let node_ids = inner_batch.column(col_idx).as_primitive::<arrow_array::types::UInt32Type>();
                for i in 0..inner_batch.num_rows() {
                    max_id = max_id.max(node_ids.value(i));
                }
            }
            
            if leaf_batch.num_rows() > 0 {
                let col_idx = get_col(&leaf_batch, NODE_ID)?;
                let node_ids = leaf_batch.column(col_idx).as_primitive::<arrow_array::types::UInt32Type>();
                for i in 0..leaf_batch.num_rows() {
                    max_id = max_id.max(node_ids.value(i));
                }
            }
            
            max_id
        };
        
        // Create sparse array of nodes (filled with dummy data initially)
        let mut nodes = vec![
            BKDNode::Leaf(BKDLeafNode {
                bounds: [0.0, 0.0, 0.0, 0.0],
                file_id: 0,
                row_offset: 0,
                num_rows: 0,
            });
            (max_node_id + 1) as usize
        ];
        
        let mut num_leaves = 0;
        
        // Fill in inner nodes
        if inner_batch.num_rows() > 0 {
            let node_ids = inner_batch.column(get_col(&inner_batch, NODE_ID)?).as_primitive::<arrow_array::types::UInt32Type>();
            let min_x = inner_batch.column(get_col(&inner_batch, MIN_X)?).as_primitive::<arrow_array::types::Float64Type>();
            let min_y = inner_batch.column(get_col(&inner_batch, MIN_Y)?).as_primitive::<arrow_array::types::Float64Type>();
            let max_x = inner_batch.column(get_col(&inner_batch, MAX_X)?).as_primitive::<arrow_array::types::Float64Type>();
            let max_y = inner_batch.column(get_col(&inner_batch, MAX_Y)?).as_primitive::<arrow_array::types::Float64Type>();
            let split_dim = inner_batch.column(get_col(&inner_batch, SPLIT_DIM)?).as_primitive::<arrow_array::types::UInt8Type>();
            let split_value = inner_batch.column(get_col(&inner_batch, SPLIT_VALUE)?).as_primitive::<arrow_array::types::Float64Type>();
            let left_child = inner_batch.column(get_col(&inner_batch, LEFT_CHILD)?).as_primitive::<arrow_array::types::UInt32Type>();
            let right_child = inner_batch.column(get_col(&inner_batch, RIGHT_CHILD)?).as_primitive::<arrow_array::types::UInt32Type>();
            
            for i in 0..inner_batch.num_rows() {
                let node_id = node_ids.value(i) as usize;
                nodes[node_id] = BKDNode::Inner(BKDInnerNode {
                    bounds: [
                        min_x.value(i),
                        min_y.value(i),
                        max_x.value(i),
                        max_y.value(i),
                    ],
                    split_dim: split_dim.value(i),
                    split_value: split_value.value(i),
                    left_child: left_child.value(i),
                    right_child: right_child.value(i),
                });
            }
        }
        
        // Fill in leaf nodes
        if leaf_batch.num_rows() > 0 {
            let node_ids = leaf_batch.column(get_col(&leaf_batch, NODE_ID)?).as_primitive::<arrow_array::types::UInt32Type>();
            let min_x = leaf_batch.column(get_col(&leaf_batch, MIN_X)?).as_primitive::<arrow_array::types::Float64Type>();
            let min_y = leaf_batch.column(get_col(&leaf_batch, MIN_Y)?).as_primitive::<arrow_array::types::Float64Type>();
            let max_x = leaf_batch.column(get_col(&leaf_batch, MAX_X)?).as_primitive::<arrow_array::types::Float64Type>();
            let max_y = leaf_batch.column(get_col(&leaf_batch, MAX_Y)?).as_primitive::<arrow_array::types::Float64Type>();
            let file_id = leaf_batch.column(get_col(&leaf_batch, FILE_ID)?).as_primitive::<arrow_array::types::UInt32Type>();
            let row_offset = leaf_batch.column(get_col(&leaf_batch, ROW_OFFSET)?).as_primitive::<arrow_array::types::UInt64Type>();
            let num_rows = leaf_batch.column(get_col(&leaf_batch, NUM_ROWS)?).as_primitive::<arrow_array::types::UInt64Type>();
            
            for i in 0..leaf_batch.num_rows() {
                let node_id = node_ids.value(i) as usize;
                nodes[node_id] = BKDNode::Leaf(BKDLeafNode {
                    bounds: [
                        min_x.value(i),
                        min_y.value(i),
                        max_x.value(i),
                        max_y.value(i),
                    ],
                    file_id: file_id.value(i),
                    row_offset: row_offset.value(i),
                    num_rows: num_rows.value(i),
                });
                num_leaves += 1;
            }
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
    pub fn build(&self, points: &mut [(f64, f64, u64)], batches_per_file: u32) -> Result<(Vec<BKDNode>, Vec<RecordBatch>)> {
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
            batches_per_file,
            &mut all_nodes,
            &mut all_leaf_batches,
        )?;

        // Post-process: Update leaf nodes with correct row offsets
        // Leaves are created in order during build_recursive, so we can
        // calculate cumulative offsets based on actual batch sizes
        let mut current_file_id = 0u32;
        let mut row_offset_in_file = 0u64;
        let mut batches_in_current_file = 0u32;
        let mut leaf_idx = 0;
        
        for node in all_nodes.iter_mut() {
            if let BKDNode::Leaf(leaf) = node {
                if leaf_idx < all_leaf_batches.len() {
                    let batch_num_rows = all_leaf_batches[leaf_idx].num_rows() as u64;
                    
                    // Check if we need to move to next file
                    if batches_in_current_file >= batches_per_file && batches_per_file > 0 {
                        current_file_id += 1;
                        row_offset_in_file = 0;
                        batches_in_current_file = 0;
                    }
                    
                    // Update leaf with correct metadata
                    leaf.file_id = current_file_id;
                    leaf.row_offset = row_offset_in_file;
                    leaf.num_rows = batch_num_rows;
                    
                    // Advance for next leaf
                    row_offset_in_file += batch_num_rows;
                    batches_in_current_file += 1;
                    leaf_idx += 1;
                }
            }
        }

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
        batches_per_file: u32,
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
            let num_rows = leaf_batch.num_rows() as u64;
            all_leaf_batches.push(leaf_batch);

            // Create leaf node (file_id, row_offset will be set in post-processing)
            all_nodes.push(BKDNode::Leaf(BKDLeafNode {
                bounds: [min_x, min_y, max_x, max_y],
                file_id: 0,  // Will be updated in post-processing
                row_offset: 0,  // Will be updated in post-processing
                num_rows,
            }));

            // Debug: Check if SF (row_id=0) is in this leaf
            if points.iter().any(|(_, _, rid)| *rid == 0) {
                println!("🎯 SF (row_id=0) in leaf node_id={}, leaf_id={}, num_rows={}, bounds=[{}, {}, {}, {}]",
                         node_id, leaf_id, num_rows, min_x, min_y, max_x, max_y);
            }

            return Ok(node_id);
        }

        // Recursive case: split and build subtrees
        let split_dim = (depth % 2) as u8; // Alternate between X (0) and Y (1)

        // TODO: Replace with radix selection for O(n) median finding (Lucene's approach)
        // Current: O(n log n) sorting at each level = O(n log² n) total
        // Target: O(n) radix select at each level = O(n log n) total
        // See: https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/bkd/BKDRadixSelector.java
        
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
        
        // Debug: Log first inner node to verify bounds
        if all_nodes.is_empty() {
            println!("🔍 Root node bounds: [{}, {}, {}, {}]", min_x, min_y, max_x, max_y);
            println!("   Split on dim {} at value {}", split_dim, split_value);
            println!("   Contains SF (-122.4194, 37.7749)? x_ok={}, y_ok={}", 
                     min_x <= -122.4194 && -122.4194 <= max_x,
                     min_y <= 37.7749 && 37.7749 <= max_y);
        }

        // Reserve space for this inner node (placeholder - we'll update it after building children)
        let node_id = all_nodes.len() as u32;
        all_nodes.push(BKDNode::Inner(BKDInnerNode {
            bounds: [min_x, min_y, max_x, max_y],
            split_dim,
            split_value,
            left_child: 0,  // Placeholder
            right_child: 0, // Placeholder
        }));

        // Recursively build left and right subtrees
        let (left_points, right_points) = points.split_at_mut(mid);

        let left_child_id = self.build_recursive(
            left_points,
            depth + 1,
            leaf_counter,
            batches_per_file,
            all_nodes,
            all_leaf_batches,
        )?;

        let right_child_id = self.build_recursive(
            right_points,
            depth + 1,
            leaf_counter,
            batches_per_file,
            all_nodes,
            all_leaf_batches,
        )?;

        // Update the inner node with actual child pointers
        if let BKDNode::Inner(inner) = &mut all_nodes[node_id as usize] {
            inner.left_child = left_child_id;
            inner.right_child = right_child_id;
        }

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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use arrow_array::{UInt32Array, UInt8Array};

//     /// Helper to serialize nodes (mirrors logic from geoindex.rs)
//     fn serialize_nodes(nodes: &[BKDNode]) -> Result<RecordBatch> {
//         let mut min_x_vals = Vec::with_capacity(nodes.len());
//         let mut min_y_vals = Vec::with_capacity(nodes.len());
//         let mut max_x_vals = Vec::with_capacity(nodes.len());
//         let mut max_y_vals = Vec::with_capacity(nodes.len());
//         let mut split_dim_vals = Vec::with_capacity(nodes.len());
//         let mut split_value_vals = Vec::with_capacity(nodes.len());
//         let mut left_child_vals = Vec::with_capacity(nodes.len());
//         let mut right_child_vals = Vec::with_capacity(nodes.len());
//         let mut leaf_id_vals = Vec::with_capacity(nodes.len());

//         for node in nodes {
//             min_x_vals.push(node.bounds[0]);
//             min_y_vals.push(node.bounds[1]);
//             max_x_vals.push(node.bounds[2]);
//             max_y_vals.push(node.bounds[3]);
//             split_dim_vals.push(node.split_dim);
//             split_value_vals.push(node.split_value);
//             left_child_vals.push(node.left_child);
//             right_child_vals.push(node.right_child);
//             leaf_id_vals.push(node.leaf_id);
//         }

//         let schema = Arc::new(Schema::new(vec![
//             Field::new("min_x", DataType::Float64, false),
//             Field::new("min_y", DataType::Float64, false),
//             Field::new("max_x", DataType::Float64, false),
//             Field::new("max_y", DataType::Float64, false),
//             Field::new("split_dim", DataType::UInt8, false),
//             Field::new("split_value", DataType::Float64, false),
//             Field::new("left_child", DataType::UInt32, true),
//             Field::new("right_child", DataType::UInt32, true),
//             Field::new("leaf_id", DataType::UInt32, true),
//         ]));

//         let columns: Vec<ArrayRef> = vec![
//             Arc::new(Float64Array::from(min_x_vals)),
//             Arc::new(Float64Array::from(min_y_vals)),
//             Arc::new(Float64Array::from(max_x_vals)),
//             Arc::new(Float64Array::from(max_y_vals)),
//             Arc::new(UInt8Array::from(split_dim_vals)),
//             Arc::new(Float64Array::from(split_value_vals)),
//             Arc::new(UInt32Array::from(left_child_vals)),
//             Arc::new(UInt32Array::from(right_child_vals)),
//             Arc::new(UInt32Array::from(leaf_id_vals)),
//         ];

//         Ok(RecordBatch::try_new(schema, columns)?)
//     }

//     #[test]
//     fn test_empty_tree_roundtrip() {
//         // Create empty tree
//         let tree = BKDTreeLookup::new(vec![], 0, 0);
        
//         // Serialize
//         let batch = serialize_nodes(&tree.nodes).unwrap();
//         assert_eq!(batch.num_rows(), 0);
        
//         // Deserialize
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         // Verify
//         assert_eq!(deserialized.nodes.len(), 0);
//         assert_eq!(deserialized.num_leaves, 0);
//         assert_eq!(deserialized.root_id, 0);
//     }

//     #[test]
//     fn test_single_leaf_roundtrip() {
//         // Create single leaf node
//         let nodes = vec![BKDNode {
//             bounds: [1.0, 2.0, 3.0, 4.0],
//             split_dim: 0,
//             split_value: 0.0,
//             left_child: None,
//             right_child: None,
//             leaf_id: Some(0),
//         }];
        
//         let tree = BKDTreeLookup::new(nodes.clone(), 0, 1);
        
//         // Serialize
//         let batch = serialize_nodes(&tree.nodes).unwrap();
//         assert_eq!(batch.num_rows(), 1);
        
//         // Deserialize
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         // Verify structure
//         assert_eq!(deserialized.nodes.len(), 1);
//         assert_eq!(deserialized.num_leaves, 1);
//         assert_eq!(deserialized.root_id, 0);
        
//         // Verify node fields
//         let node = &deserialized.nodes[0];
//         assert_eq!(node.bounds, [1.0, 2.0, 3.0, 4.0]);
//         assert_eq!(node.split_dim, 0);
//         assert_eq!(node.split_value, 0.0);
//         assert_eq!(node.left_child, None);
//         assert_eq!(node.right_child, None);
//         assert_eq!(node.leaf_id, Some(0));
//     }

//     #[test]
//     fn test_simple_tree_roundtrip() {
//         // Create tree: 1 root (inner) + 2 leaves
//         let nodes = vec![
//             // Node 0: Root (inner node)
//             BKDNode {
//                 bounds: [0.0, 0.0, 10.0, 10.0],
//                 split_dim: 0, // Split on X
//                 split_value: 5.0,
//                 left_child: Some(1),
//                 right_child: Some(2),
//                 leaf_id: None,
//             },
//             // Node 1: Left leaf
//             BKDNode {
//                 bounds: [0.0, 0.0, 5.0, 10.0],
//                 split_dim: 0,
//                 split_value: 0.0,
//                 left_child: None,
//                 right_child: None,
//                 leaf_id: Some(0),
//             },
//             // Node 2: Right leaf
//             BKDNode {
//                 bounds: [5.0, 0.0, 10.0, 10.0],
//                 split_dim: 0,
//                 split_value: 0.0,
//                 left_child: None,
//                 right_child: None,
//                 leaf_id: Some(1),
//             },
//         ];
        
//         let tree = BKDTreeLookup::new(nodes.clone(), 0, 2);
        
//         // Serialize
//         let batch = serialize_nodes(&tree.nodes).unwrap();
//         assert_eq!(batch.num_rows(), 3);
        
//         // Deserialize
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         // Verify structure
//         assert_eq!(deserialized.nodes.len(), 3);
//         assert_eq!(deserialized.num_leaves, 2);
//         assert_eq!(deserialized.root_id, 0);
        
//         // Verify root node (inner)
//         let root = &deserialized.nodes[0];
//         assert_eq!(root.bounds, [0.0, 0.0, 10.0, 10.0]);
//         assert_eq!(root.split_dim, 0);
//         assert_eq!(root.split_value, 5.0);
//         assert_eq!(root.left_child, Some(1));
//         assert_eq!(root.right_child, Some(2));
//         assert_eq!(root.leaf_id, None);
        
//         // Verify left leaf
//         let left = &deserialized.nodes[1];
//         assert_eq!(left.bounds, [0.0, 0.0, 5.0, 10.0]);
//         assert_eq!(left.leaf_id, Some(0));
//         assert_eq!(left.left_child, None);
//         assert_eq!(left.right_child, None);
        
//         // Verify right leaf
//         let right = &deserialized.nodes[2];
//         assert_eq!(right.bounds, [5.0, 0.0, 10.0, 10.0]);
//         assert_eq!(right.leaf_id, Some(1));
//         assert_eq!(right.left_child, None);
//         assert_eq!(right.right_child, None);
//     }

//     #[test]
//     fn test_multi_level_tree_roundtrip() {
//         // Build a real tree from points
//         let mut points = vec![
//             (1.0, 1.0, 0),
//             (2.0, 2.0, 1),
//             (3.0, 3.0, 2),
//             (4.0, 4.0, 3),
//             (5.0, 5.0, 4),
//             (6.0, 6.0, 5),
//             (7.0, 7.0, 6),
//             (8.0, 8.0, 7),
//             (9.0, 9.0, 8),
//             (10.0, 10.0, 9),
//         ];
        
//         let builder = BKDTreeBuilder::new(3); // leaf_size = 3
//         let (nodes, _leaf_batches) = builder.build(&mut points).unwrap();
        
//         let original_tree = BKDTreeLookup::new(nodes.clone(), 0, 4);
        
//         // Serialize
//         let batch = serialize_nodes(&original_tree.nodes).unwrap();
        
//         // Deserialize
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         // Verify counts
//         assert_eq!(deserialized.nodes.len(), original_tree.nodes.len());
//         assert_eq!(deserialized.num_leaves, original_tree.num_leaves);
//         assert_eq!(deserialized.root_id, 0);
        
//         // Verify each node
//         for (i, (orig, deser)) in original_tree.nodes.iter().zip(deserialized.nodes.iter()).enumerate() {
//             assert_eq!(deser.bounds, orig.bounds, "Node {} bounds mismatch", i);
//             assert_eq!(deser.split_dim, orig.split_dim, "Node {} split_dim mismatch", i);
//             assert_eq!(deser.split_value, orig.split_value, "Node {} split_value mismatch", i);
//             assert_eq!(deser.left_child, orig.left_child, "Node {} left_child mismatch", i);
//             assert_eq!(deser.right_child, orig.right_child, "Node {} right_child mismatch", i);
//             assert_eq!(deser.leaf_id, orig.leaf_id, "Node {} leaf_id mismatch", i);
//         }
//     }

//     #[test]
//     fn test_field_precision() {
//         // Test edge values and precision
//         let nodes = vec![
//             BKDNode {
//                 bounds: [f64::MIN, f64::MAX, -1e-10, 1e10],
//                 split_dim: 1,
//                 split_value: std::f64::consts::PI,
//                 left_child: None,
//                 right_child: None,
//                 leaf_id: Some(42),
//             },
//         ];
        
//         let tree = BKDTreeLookup::new(nodes.clone(), 0, 1);
        
//         // Serialize and deserialize
//         let batch = serialize_nodes(&tree.nodes).unwrap();
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         // Verify exact values (no precision loss)
//         let node = &deserialized.nodes[0];
//         assert_eq!(node.bounds[0], f64::MIN);
//         assert_eq!(node.bounds[1], f64::MAX);
//         assert_eq!(node.bounds[2], -1e-10);
//         assert_eq!(node.bounds[3], 1e10);
//         assert_eq!(node.split_value, std::f64::consts::PI);
//         assert_eq!(node.leaf_id, Some(42));
//     }

//     #[test]
//     fn test_tree_structure_validation() {
//         // Create tree with invalid structure (child pointer out of bounds)
//         // This should be caught during traversal, not deserialization
//         let nodes = vec![
//             BKDNode {
//                 bounds: [0.0, 0.0, 10.0, 10.0],
//                 split_dim: 0,
//                 split_value: 5.0,
//                 left_child: Some(1),
//                 right_child: Some(2),
//                 leaf_id: None,
//             },
//             BKDNode {
//                 bounds: [0.0, 0.0, 5.0, 10.0],
//                 split_dim: 0,
//                 split_value: 0.0,
//                 left_child: None,
//                 right_child: None,
//                 leaf_id: Some(0),
//             },
//         ];
        
//         let tree = BKDTreeLookup::new(nodes.clone(), 0, 1);
        
//         // Serialize and deserialize
//         let batch = serialize_nodes(&tree.nodes).unwrap();
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         // Deserialization succeeds
//         assert_eq!(deserialized.nodes.len(), 2);
        
//         // But traversal would fail because right_child=2 is out of bounds
//         // This is expected - validation happens at query time
//         let query_bbox = [0.0, 0.0, 10.0, 10.0];
//         let leaves = deserialized.find_intersecting_leaves(query_bbox);
        
//         // Should only find the valid left leaf
//         assert_eq!(leaves.len(), 1);
//         assert_eq!(leaves[0], 0);
//     }

//     #[test]
//     fn test_nullable_fields() {
//         // Test that nullable fields work correctly
//         let nodes = vec![
//             // Inner node: has children, no leaf_id
//             BKDNode {
//                 bounds: [0.0, 0.0, 10.0, 10.0],
//                 split_dim: 0,
//                 split_value: 5.0,
//                 left_child: Some(1),
//                 right_child: Some(2),
//                 leaf_id: None, // NULL
//             },
//             // Leaf node: has leaf_id, no children
//             BKDNode {
//                 bounds: [0.0, 0.0, 5.0, 10.0],
//                 split_dim: 0,
//                 split_value: 0.0,
//                 left_child: None, // NULL
//                 right_child: None, // NULL
//                 leaf_id: Some(0),
//             },
//             // Another leaf
//             BKDNode {
//                 bounds: [5.0, 0.0, 10.0, 10.0],
//                 split_dim: 0,
//                 split_value: 0.0,
//                 left_child: None,
//                 right_child: None,
//                 leaf_id: Some(1),
//             },
//         ];
        
//         let tree = BKDTreeLookup::new(nodes.clone(), 0, 2);
        
//         // Serialize
//         let batch = serialize_nodes(&tree.nodes).unwrap();
        
//         // Check that nulls are properly represented
//         let left_child_col = batch.column(6).as_primitive::<arrow_array::types::UInt32Type>();
//         let right_child_col = batch.column(7).as_primitive::<arrow_array::types::UInt32Type>();
//         let leaf_id_col = batch.column(8).as_primitive::<arrow_array::types::UInt32Type>();
        
//         // Row 0 (inner): children NOT null, leaf_id IS null
//         assert!(!left_child_col.is_null(0));
//         assert!(!right_child_col.is_null(0));
//         assert!(leaf_id_col.is_null(0));
        
//         // Row 1 (leaf): children ARE null, leaf_id NOT null
//         assert!(left_child_col.is_null(1));
//         assert!(right_child_col.is_null(1));
//         assert!(!leaf_id_col.is_null(1));
        
//         // Deserialize and verify
//         let deserialized = BKDTreeLookup::from_record_batch(batch).unwrap();
        
//         assert_eq!(deserialized.nodes[0].left_child, Some(1));
//         assert_eq!(deserialized.nodes[0].right_child, Some(2));
//         assert_eq!(deserialized.nodes[0].leaf_id, None);
        
//         assert_eq!(deserialized.nodes[1].left_child, None);
//         assert_eq!(deserialized.nodes[1].right_child, None);
//         assert_eq!(deserialized.nodes[1].leaf_id, Some(0));
//     }
// }

