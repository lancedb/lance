//! Spatial index implementations
//! 
//! This module provides two R-tree implementations:
//! - Simple R-tree: Single rstar tree, good for smaller datasets
//! - Paged Leaf R-tree: Complex paged approach, better for very large datasets

// Re-export both implementations for flexibility

// Simple R-tree (default - recommended for most use cases)
pub use crate::rtree::simple_rtree::{
    RTreeEntry as SimpleRTreeEntry,
    RTreeIndex as SimpleRTreeIndex,
};
pub use crate::rtree::simple_builder::train_geo_index as train_simple_rtree_index;

// Paged Leaf R-tree (for very large datasets)
pub use crate::rtree::paged_leaf_rtree::{
    SpatialDataEntry as PagedSpatialEntry,
    PagedLeafRTreeIndex as PagedRTreeIndex,
    PagedLeafConfig,
};
pub use crate::rtree::builder::train_rtree_index as train_paged_rtree_index;

// Default exports (using Simple R-tree as the default)
pub type SpatialEntry = SimpleRTreeEntry;
pub type SpatialIndex = SimpleRTreeIndex;
pub use train_simple_rtree_index as train_rtree_index;

// Backward compatibility
pub type RTreeEntry = SpatialEntry;
pub type RTreeIndex = SpatialIndex;
