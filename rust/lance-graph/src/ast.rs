// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Abstract Syntax Tree for Cypher queries
//!
//! This module defines the AST nodes for representing parsed Cypher queries.
//! The AST is designed to capture the essential graph patterns while being
//! simple enough to translate to SQL efficiently.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A complete Cypher query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CypherQuery {
    /// MATCH clauses
    pub match_clauses: Vec<MatchClause>,
    /// WHERE clause (optional)
    pub where_clause: Option<WhereClause>,
    /// RETURN clause
    pub return_clause: ReturnClause,
    /// LIMIT clause (optional)
    pub limit: Option<u64>,
    /// ORDER BY clause (optional)
    pub order_by: Option<OrderByClause>,
}

impl CypherQuery {
    /// Extract all node labels referenced in the query
    pub fn get_node_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();
        for match_clause in &self.match_clauses {
            for pattern in &match_clause.patterns {
                match pattern {
                    GraphPattern::Node(node) => {
                        for label in &node.labels {
                            if !labels.contains(label) {
                                labels.push(label.clone());
                            }
                        }
                    }
                    GraphPattern::Path(path) => {
                        for label in &path.start_node.labels {
                            if !labels.contains(label) {
                                labels.push(label.clone());
                            }
                        }
                        for segment in &path.segments {
                            for label in &segment.end_node.labels {
                                if !labels.contains(label) {
                                    labels.push(label.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        labels
    }

    /// Extract all relationship types referenced in the query
    pub fn get_relationship_types(&self) -> Vec<String> {
        let mut types = Vec::new();
        for match_clause in &self.match_clauses {
            for pattern in &match_clause.patterns {
                if let GraphPattern::Path(path) = pattern {
                    for segment in &path.segments {
                        for rel_type in &segment.relationship.types {
                            if !types.contains(rel_type) {
                                types.push(rel_type.clone());
                            }
                        }
                    }
                }
            }
        }
        types
    }
}

/// A MATCH clause containing graph patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchClause {
    /// Graph patterns to match
    pub patterns: Vec<GraphPattern>,
}

/// A graph pattern (nodes and relationships)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphPattern {
    /// A single node pattern
    Node(NodePattern),
    /// A path pattern (node-relationship-node sequence)
    Path(PathPattern),
}

/// A node pattern in a graph query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePattern {
    /// Variable name for the node (e.g., 'n' in (n:Person))
    pub variable: Option<String>,
    /// Node labels (e.g., ['Person', 'Employee'])
    pub labels: Vec<String>,
    /// Property constraints (e.g., {name: 'John', age: 30})
    pub properties: HashMap<String, PropertyValue>,
}

/// A path pattern connecting nodes through relationships
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathPattern {
    /// Starting node
    pub start_node: NodePattern,
    /// Relationships and intermediate nodes
    pub segments: Vec<PathSegment>,
}

/// A segment of a path (relationship + end node)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathSegment {
    /// The relationship in this segment
    pub relationship: RelationshipPattern,
    /// The end node of this segment
    pub end_node: NodePattern,
}

/// A relationship pattern
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationshipPattern {
    /// Variable name for the relationship (e.g., 'r' in [r:KNOWS])
    pub variable: Option<String>,
    /// Relationship types (e.g., ['KNOWS', 'FRIEND_OF'])
    pub types: Vec<String>,
    /// Direction of the relationship
    pub direction: RelationshipDirection,
    /// Property constraints on the relationship
    pub properties: HashMap<String, PropertyValue>,
    /// Length constraints (for variable-length paths)
    pub length: Option<LengthRange>,
}

/// Direction of a relationship
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipDirection {
    /// Outgoing relationship (->)
    Outgoing,
    /// Incoming relationship (<-)
    Incoming,
    /// Undirected relationship (-)
    Undirected,
}

/// Length range for variable-length paths
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LengthRange {
    /// Minimum length (inclusive)
    pub min: Option<u32>,
    /// Maximum length (inclusive)
    pub max: Option<u32>,
}

/// Property value in patterns and expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    /// String literal
    String(String),
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// Boolean literal
    Boolean(bool),
    /// Null value
    Null,
    /// Parameter reference (e.g., $param)
    Parameter(String),
    /// Property reference (e.g., node.property)
    Property(PropertyRef),
}

/// Reference to a property of a node or relationship
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyRef {
    /// Variable name (e.g., 'n' in n.name)
    pub variable: String,
    /// Property name
    pub property: String,
}

/// WHERE clause for filtering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhereClause {
    /// Boolean expression for filtering
    pub expression: BooleanExpression,
}

/// Boolean expressions in WHERE clauses
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BooleanExpression {
    /// Comparison operation (=, <>, <, >, <=, >=)
    Comparison {
        left: ValueExpression,
        operator: ComparisonOperator,
        right: ValueExpression,
    },
    /// Logical AND
    And(Box<BooleanExpression>, Box<BooleanExpression>),
    /// Logical OR
    Or(Box<BooleanExpression>, Box<BooleanExpression>),
    /// Logical NOT
    Not(Box<BooleanExpression>),
    /// Property existence check
    Exists(PropertyRef),
    /// IN clause
    In {
        expression: ValueExpression,
        list: Vec<ValueExpression>,
    },
    /// LIKE pattern matching
    Like {
        expression: ValueExpression,
        pattern: String,
    },
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// Value expressions (for comparisons and return values)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueExpression {
    /// Variable reference
    Variable(String),
    /// Property reference
    Property(PropertyRef),
    /// Literal value
    Literal(PropertyValue),
    /// Function call
    Function {
        name: String,
        args: Vec<ValueExpression>,
    },
    /// Arithmetic operation
    Arithmetic {
        left: Box<ValueExpression>,
        operator: ArithmeticOperator,
        right: Box<ValueExpression>,
    },
}

/// Arithmetic operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArithmeticOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
}

/// RETURN clause specifying what to return
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnClause {
    /// Whether DISTINCT was specified
    pub distinct: bool,
    /// Items to return
    pub items: Vec<ReturnItem>,
}

/// An item in the RETURN clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnItem {
    /// The expression to return
    pub expression: ValueExpression,
    /// Alias for the returned value
    pub alias: Option<String>,
}

/// ORDER BY clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByClause {
    /// Items to order by
    pub items: Vec<OrderByItem>,
}

/// An item in the ORDER BY clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByItem {
    /// Expression to order by
    pub expression: ValueExpression,
    /// Sort direction
    pub direction: SortDirection,
}

/// Sort direction for ORDER BY
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

impl NodePattern {
    /// Create a new node pattern
    pub fn new(variable: Option<String>) -> Self {
        Self {
            variable,
            labels: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Add a label to the node pattern
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Add a property constraint to the node pattern
    pub fn with_property<S: Into<String>>(mut self, key: S, value: PropertyValue) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

impl RelationshipPattern {
    /// Create a new relationship pattern
    pub fn new(direction: RelationshipDirection) -> Self {
        Self {
            variable: None,
            types: Vec::new(),
            direction,
            properties: HashMap::new(),
            length: None,
        }
    }

    /// Set the variable name for the relationship
    pub fn with_variable<S: Into<String>>(mut self, variable: S) -> Self {
        self.variable = Some(variable.into());
        self
    }

    /// Add a type to the relationship pattern
    pub fn with_type<S: Into<String>>(mut self, rel_type: S) -> Self {
        self.types.push(rel_type.into());
        self
    }

    /// Add a property constraint to the relationship pattern
    pub fn with_property<S: Into<String>>(mut self, key: S, value: PropertyValue) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

impl PropertyRef {
    /// Create a new property reference
    pub fn new<S: Into<String>>(variable: S, property: S) -> Self {
        Self {
            variable: variable.into(),
            property: property.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_pattern_creation() {
        let node = NodePattern::new(Some("n".to_string()))
            .with_label("Person")
            .with_property("name", PropertyValue::String("John".to_string()));

        assert_eq!(node.variable, Some("n".to_string()));
        assert_eq!(node.labels, vec!["Person"]);
        assert_eq!(node.properties.len(), 1);
    }

    #[test]
    fn test_relationship_pattern_creation() {
        let rel = RelationshipPattern::new(RelationshipDirection::Outgoing)
            .with_variable("r")
            .with_type("KNOWS");

        assert_eq!(rel.variable, Some("r".to_string()));
        assert_eq!(rel.types, vec!["KNOWS"]);
        assert_eq!(rel.direction, RelationshipDirection::Outgoing);
    }

    #[test]
    fn test_property_ref() {
        let prop_ref = PropertyRef::new("n", "name");
        assert_eq!(prop_ref.variable, "n");
        assert_eq!(prop_ref.property, "name");
    }
}
