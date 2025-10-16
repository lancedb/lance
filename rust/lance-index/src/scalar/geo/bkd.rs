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

use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt64Array};
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
        let mut _nodes_visited = 0;

        while let Some(node_id) = stack.pop() {
            if node_id as usize >= self.nodes.len() {
                continue;
            }

            let node = &self.nodes[node_id as usize];
            _nodes_visited += 1;

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
            let _leaf_id = *leaf_counter;
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



#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, UInt32Array, UInt64Array, UInt8Array};

    /// Test serialization and deserialization of a simple BKD tree
    #[test]
    fn test_serialize_deserialize_simple_tree() {
        // Create a simple tree: 1 inner node with 2 leaf children
        let nodes = vec![
            BKDNode::Inner(BKDInnerNode {
                bounds: [-10.0, -10.0, 10.0, 10.0],
                split_dim: 0,
                split_value: 0.0,
                left_child: 1,
                right_child: 2,
            }),
            BKDNode::Leaf(BKDLeafNode {
                bounds: [-10.0, -10.0, 0.0, 10.0],
                file_id: 0,
                row_offset: 0,
                num_rows: 100,
            }),
            BKDNode::Leaf(BKDLeafNode {
                bounds: [0.0, -10.0, 10.0, 10.0],
                file_id: 0,
                row_offset: 100,
                num_rows: 100,
            }),
        ];

        // Separate inner and leaf nodes
        let inner_nodes: Vec<(u32, &BKDNode)> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, BKDNode::Inner(_)))
            .map(|(i, n)| (i as u32, n))
            .collect();

        let leaf_nodes: Vec<(u32, &BKDNode)> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, BKDNode::Leaf(_)))
            .map(|(i, n)| (i as u32, n))
            .collect();

        // Serialize inner nodes
        let mut inner_node_ids = Vec::new();
        let mut inner_min_x = Vec::new();
        let mut inner_min_y = Vec::new();
        let mut inner_max_x = Vec::new();
        let mut inner_max_y = Vec::new();
        let mut inner_split_dim = Vec::new();
        let mut inner_split_value = Vec::new();
        let mut inner_left_child = Vec::new();
        let mut inner_right_child = Vec::new();

        for (idx, node) in &inner_nodes {
            if let BKDNode::Inner(inner) = node {
                inner_node_ids.push(*idx);
                inner_min_x.push(inner.bounds[0]);
                inner_min_y.push(inner.bounds[1]);
                inner_max_x.push(inner.bounds[2]);
                inner_max_y.push(inner.bounds[3]);
                inner_split_dim.push(inner.split_dim);
                inner_split_value.push(inner.split_value);
                inner_left_child.push(inner.left_child);
                inner_right_child.push(inner.right_child);
            }
        }

        let inner_batch = RecordBatch::try_new(
            inner_node_schema(),
            vec![
                Arc::new(UInt32Array::from(inner_node_ids)),
                Arc::new(Float64Array::from(inner_min_x)),
                Arc::new(Float64Array::from(inner_min_y)),
                Arc::new(Float64Array::from(inner_max_x)),
                Arc::new(Float64Array::from(inner_max_y)),
                Arc::new(UInt8Array::from(inner_split_dim)),
                Arc::new(Float64Array::from(inner_split_value)),
                Arc::new(UInt32Array::from(inner_left_child)),
                Arc::new(UInt32Array::from(inner_right_child)),
            ],
        )
        .unwrap();

        // Serialize leaf nodes
        let mut leaf_node_ids = Vec::new();
        let mut leaf_min_x = Vec::new();
        let mut leaf_min_y = Vec::new();
        let mut leaf_max_x = Vec::new();
        let mut leaf_max_y = Vec::new();
        let mut leaf_file_ids = Vec::new();
        let mut leaf_row_offsets = Vec::new();
        let mut leaf_num_rows = Vec::new();

        for (idx, node) in &leaf_nodes {
            if let BKDNode::Leaf(leaf) = node {
                leaf_node_ids.push(*idx);
                leaf_min_x.push(leaf.bounds[0]);
                leaf_min_y.push(leaf.bounds[1]);
                leaf_max_x.push(leaf.bounds[2]);
                leaf_max_y.push(leaf.bounds[3]);
                leaf_file_ids.push(leaf.file_id);
                leaf_row_offsets.push(leaf.row_offset);
                leaf_num_rows.push(leaf.num_rows);
            }
        }

        let leaf_batch = RecordBatch::try_new(
            leaf_node_schema(),
            vec![
                Arc::new(UInt32Array::from(leaf_node_ids)),
                Arc::new(Float64Array::from(leaf_min_x)),
                Arc::new(Float64Array::from(leaf_min_y)),
                Arc::new(Float64Array::from(leaf_max_x)),
                Arc::new(Float64Array::from(leaf_max_y)),
                Arc::new(UInt32Array::from(leaf_file_ids)),
                Arc::new(UInt64Array::from(leaf_row_offsets)),
                Arc::new(UInt64Array::from(leaf_num_rows)),
            ],
        )
        .unwrap();

        // Deserialize
        let tree = BKDTreeLookup::from_record_batches(inner_batch, leaf_batch).unwrap();

        // Verify structure
        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.root_id, 0);
        assert_eq!(tree.num_leaves, 2);

        // Verify root (inner node)
        match &tree.nodes[0] {
            BKDNode::Inner(inner) => {
                assert_eq!(inner.bounds, [-10.0, -10.0, 10.0, 10.0]);
                assert_eq!(inner.split_dim, 0);
                assert_eq!(inner.split_value, 0.0);
                assert_eq!(inner.left_child, 1);
                assert_eq!(inner.right_child, 2);
            }
            _ => panic!("Expected inner node at index 0"),
        }

        // Verify left leaf
        match &tree.nodes[1] {
            BKDNode::Leaf(leaf) => {
                assert_eq!(leaf.bounds, [-10.0, -10.0, 0.0, 10.0]);
                assert_eq!(leaf.file_id, 0);
                assert_eq!(leaf.row_offset, 0);
                assert_eq!(leaf.num_rows, 100);
            }
            _ => panic!("Expected leaf node at index 1"),
        }

        // Verify right leaf
        match &tree.nodes[2] {
            BKDNode::Leaf(leaf) => {
                assert_eq!(leaf.bounds, [0.0, -10.0, 10.0, 10.0]);
                assert_eq!(leaf.file_id, 0);
                assert_eq!(leaf.row_offset, 100);
                assert_eq!(leaf.num_rows, 100);
            }
            _ => panic!("Expected leaf node at index 2"),
        }
    }

    /// Test serialization of empty tree
    #[test]
    fn test_serialize_deserialize_empty_tree() {
        let inner_batch = RecordBatch::new_empty(inner_node_schema());
        let leaf_batch = RecordBatch::new_empty(leaf_node_schema());

        let tree = BKDTreeLookup::from_record_batches(inner_batch, leaf_batch).unwrap();

        assert_eq!(tree.nodes.len(), 0);
        assert_eq!(tree.num_leaves, 0);
    }

    /// Test bbox intersection logic
    #[test]
    fn test_bboxes_intersect() {
        // Overlapping boxes
        assert!(bboxes_intersect(
            &[0.0, 0.0, 10.0, 10.0],
            &[5.0, 5.0, 15.0, 15.0]
        ));

        // Touching boxes
        assert!(bboxes_intersect(
            &[0.0, 0.0, 10.0, 10.0],
            &[10.0, 0.0, 20.0, 10.0]
        ));

        // Fully contained
        assert!(bboxes_intersect(
            &[0.0, 0.0, 10.0, 10.0],
            &[2.0, 2.0, 8.0, 8.0]
        ));

        // Non-overlapping (left)
        assert!(!bboxes_intersect(
            &[0.0, 0.0, 10.0, 10.0],
            &[-20.0, 0.0, -10.0, 10.0]
        ));

        // Non-overlapping (above)
        assert!(!bboxes_intersect(
            &[0.0, 0.0, 10.0, 10.0],
            &[0.0, 20.0, 10.0, 30.0]
        ));
    }

    /// Test point in bbox logic
    #[test]
    fn test_point_in_bbox() {
        let bbox = [0.0, 0.0, 10.0, 10.0];

        // Inside
        assert!(point_in_bbox(5.0, 5.0, &bbox));

        // On boundary
        assert!(point_in_bbox(0.0, 0.0, &bbox));
        assert!(point_in_bbox(10.0, 10.0, &bbox));

        // Outside
        assert!(!point_in_bbox(-1.0, 5.0, &bbox));
        assert!(!point_in_bbox(11.0, 5.0, &bbox));
        assert!(!point_in_bbox(5.0, -1.0, &bbox));
        assert!(!point_in_bbox(5.0, 11.0, &bbox));
    }

    /// Test find_intersecting_leaves with simple tree
    #[test]
    fn test_find_intersecting_leaves() {
        // Create a simple tree: 1 inner node with 2 leaf children
        let nodes = vec![
            BKDNode::Inner(BKDInnerNode {
                bounds: [-10.0, -10.0, 10.0, 10.0],
                split_dim: 0,
                split_value: 0.0,
                left_child: 1,
                right_child: 2,
            }),
            BKDNode::Leaf(BKDLeafNode {
                bounds: [-10.0, -10.0, 0.0, 10.0],
                file_id: 0,
                row_offset: 0,
                num_rows: 100,
            }),
            BKDNode::Leaf(BKDLeafNode {
                bounds: [0.0, -10.0, 10.0, 10.0],
                file_id: 0,
                row_offset: 100,
                num_rows: 100,
            }),
        ];

        let tree = BKDTreeLookup::new(nodes, 0, 2);

        // Query that intersects only left leaf
        let leaves = tree
            .find_intersecting_leaves([-10.0, -10.0, -5.0, 10.0])
            .unwrap();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].file_id, 0);
        assert_eq!(leaves[0].row_offset, 0);

        // Query that intersects only right leaf
        let leaves = tree
            .find_intersecting_leaves([5.0, -10.0, 10.0, 10.0])
            .unwrap();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].file_id, 0);
        assert_eq!(leaves[0].row_offset, 100);

        // Query that intersects both leaves
        let leaves = tree
            .find_intersecting_leaves([-5.0, -10.0, 5.0, 10.0])
            .unwrap();
        assert_eq!(leaves.len(), 2);

        // Query that intersects no leaves
        let leaves = tree
            .find_intersecting_leaves([20.0, 20.0, 30.0, 30.0])
            .unwrap();
        assert_eq!(leaves.len(), 0);
    }

    /// Test tree building with small dataset
    #[test]
    fn test_build_tree_small() {
        let mut points = vec![
            (-5.0, -5.0, 0),
            (-4.0, -4.0, 1),
            (4.0, 4.0, 2),
            (5.0, 5.0, 3),
        ];

        let builder = BKDTreeBuilder::new(2); // leaf_size = 2
        let (nodes, batches) = builder.build(&mut points, 5).unwrap();

        // Should have: 1 root + 2 leaves = 3 nodes
        assert_eq!(nodes.len(), 3);
        assert_eq!(batches.len(), 2);

        // Verify root is inner node
        assert!(matches!(nodes[0], BKDNode::Inner(_)));

        // Verify we have 2 leaf nodes
        let leaf_count = nodes.iter().filter(|n| n.is_leaf()).count();
        assert_eq!(leaf_count, 2);

        // Verify each batch has correct size
        assert_eq!(batches[0].num_rows(), 2);
        assert_eq!(batches[1].num_rows(), 2);
    }
}

