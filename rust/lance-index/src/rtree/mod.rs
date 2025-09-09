//! Spatial indices for GeoArrow geometries

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator};
use deepsize::DeepSizeOf;
use rstar::AABB;
use serde::{Deserialize, Serialize};

use crate::metrics::MetricsCollector;
use crate::scalar::{AnyQuery, IndexStore, SearchResult};
use crate::{Index, IndexParams, IndexType};
use lance_core::Result;

pub mod builder;
pub mod rtree;
pub mod paged_leaf_rtree;
pub mod simple_rtree;
pub mod simple_builder;

pub const LANCE_RTREE_INDEX: &str = "__lance_rtree_index";

/// Spatial query types that can be performed against a GeoIndex
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialQuery {
    /// Find all geometries that intersect with the given bounding box
    Intersects(BoundingBox),
    /// Find all geometries that are contained within the given bounding box
    Within(BoundingBox),
    /// Find all geometries that contain the given point
    Contains(Point),
    /// Find all geometries within a given distance of a point
    DWithin(Point, f64),
    /// Find all geometries that touch the given bounding box
    Touches(BoundingBox),
    /// Find all geometries that are disjoint from the given bounding box
    Disjoint(BoundingBox),
}

impl AnyQuery for SpatialQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::Intersects(bbox) => format!("ST_Intersects({}, {})", col, bbox.format()),
            Self::Within(bbox) => format!("ST_Within({}, {})", col, bbox.format()),
            Self::Contains(point) => format!("ST_Contains({}, {})", col, point.format()),
            Self::DWithin(point, distance) => format!("ST_DWithin({}, {}, {})", col, point.format(), distance),
            Self::Touches(bbox) => format!("ST_Touches({}, {})", col, bbox.format()),
            Self::Disjoint(bbox) => format!("ST_Disjoint({}, {})", col, bbox.format()),
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        // For now, return a placeholder expression
        // In a full implementation, this would use spatial functions from DataFusion
        match self {
            Self::Intersects(_) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Expr::Column(datafusion_common::Column::new_unqualified(col))),
                op: Operator::Eq,
                right: Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
            }),
            _ => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Expr::Column(datafusion_common::Column::new_unqualified(col))),
                op: Operator::Eq,
                right: Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
            }),
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }

    fn needs_recheck(&self) -> bool {
        // Some spatial queries may need rechecking depending on the precision of the index
        false
    }
}

/// A 2D bounding box for spatial queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub fn format(&self) -> String {
        format!("BBOX({}, {}, {}, {})", self.min_x, self.min_y, self.max_x, self.max_y)
    }

    pub fn to_aabb(&self) -> AABB<[f64; 2]> {
        AABB::from_corners([self.min_x, self.min_y], [self.max_x, self.max_y])
    }

    /// Check if this bounding box intersects with another bounding box
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.max_x < other.min_x || 
          self.min_x > other.max_x || 
          self.max_y < other.min_y || 
          self.min_y > other.max_y)
    }

    /// Check if this bounding box is entirely within another bounding box
    pub fn within(&self, other: &BoundingBox) -> bool {
        self.min_x >= other.min_x &&
        self.max_x <= other.max_x &&
        self.min_y >= other.min_y &&
        self.max_y <= other.max_y
    }

    /// Check if this bounding box contains a point
    pub fn contains_point(&self, point: &Point) -> bool {
        point.x >= self.min_x &&
        point.x <= self.max_x &&
        point.y >= self.min_y &&
        point.y <= self.max_y
    }
}

/// A 2D point for spatial queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn format(&self) -> String {
        format!("POINT({} {})", self.x, self.y)
    }

    pub fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}

/// Spatial geometry types supported by the index
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub enum SpatialGeometry {
    /// A single point geometry
    Point(Point),
}

impl SpatialGeometry {
    /// Compute the minimum bounding box that encloses this geometry
    pub fn envelope(&self) -> BoundingBox {
        match self {
            Self::Point(p) => BoundingBox::new(p.x, p.y, p.x, p.y),
        }
    }

    /// Check if this geometry matches a spatial query (geometry-specific optimization)
    pub fn matches_query(&self, query: &SpatialQuery) -> bool {
        match (self, query) {
            (Self::Point(p), SpatialQuery::Intersects(bbox)) => bbox.contains_point(p),
            (Self::Point(p), SpatialQuery::Within(bbox)) => bbox.contains_point(p),
            (Self::Point(p), SpatialQuery::Contains(point)) => p == point,
            (Self::Point(p), SpatialQuery::DWithin(point, distance)) => {
                let dx = p.x - point.x;
                let dy = p.y - point.y;
                (dx * dx + dy * dy).sqrt() <= *distance
            },
            (Self::Point(p), SpatialQuery::Touches(bbox)) => {
                // For points, touching means on the boundary
                (p.x == bbox.min_x || p.x == bbox.max_x) && (p.y >= bbox.min_y && p.y <= bbox.max_y) ||
                (p.y == bbox.min_y || p.y == bbox.max_y) && (p.x >= bbox.min_x && p.x <= bbox.max_x)
            },
            (Self::Point(p), SpatialQuery::Disjoint(bbox)) => !bbox.contains_point(p),
        }
    }

    /// Format geometry for display/debugging
    pub fn format(&self) -> String {
        match self {
            Self::Point(p) => p.format(),
        }
    }
}

/// Parameters for creating a spatial index
#[derive(Default, Debug, Clone)]
pub struct GeoIndexParams {
    /// Node capacity for the R-tree (default: 32)
    pub node_capacity: Option<usize>,
}

impl GeoIndexParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node_capacity(mut self, capacity: usize) -> Self {
        self.node_capacity = Some(capacity);
        self
    }
}

impl IndexParams for GeoIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::RTree
    }

    fn index_name(&self) -> &str {
        LANCE_RTREE_INDEX
    }
}

/// Trait for spatial indices that can answer geometric queries
#[async_trait]
pub trait GeoIndex: Send + Sync + std::fmt::Debug + Index + DeepSizeOf {
    /// Search the spatial index for geometries matching the query
    async fn search(
        &self,
        query: &SpatialQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult>;

    /// Returns true if the query can be answered exactly by this index
    fn can_answer_exact(&self, query: &SpatialQuery) -> bool;

    /// Load the spatial index from storage
    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized;

    /// Get the bounding box that contains all geometries in this index
    fn total_bounds(&self) -> Option<BoundingBox>;

    /// Get statistics about the spatial index
    fn size(&self) -> usize;
}