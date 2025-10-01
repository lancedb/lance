// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Logical planning for graph queries
//!
//! This module implements the logical planning phase of the query pipeline:
//! Parse → Semantic Analysis → **Logical Plan** → Physical Plan
//!
//! Logical plans describe WHAT operations to perform, not HOW to perform them.

use crate::ast::*;
use crate::error::{GraphError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A logical plan operator - describes what operation to perform
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogicalOperator {
    /// Scan all nodes with a specific label
    ScanByLabel {
        variable: String,
        label: String,
        properties: HashMap<String, PropertyValue>,
    },

    /// Apply a filter predicate (WHERE clause)
    Filter {
        input: Box<LogicalOperator>,
        predicate: BooleanExpression,
    },

    /// Traverse relationships (the core graph operation)
    Expand {
        input: Box<LogicalOperator>,
        source_variable: String,
        target_variable: String,
        relationship_types: Vec<String>,
        direction: RelationshipDirection,
        properties: HashMap<String, PropertyValue>,
    },

    /// Variable-length path expansion (*1..2 syntax)
    VariableLengthExpand {
        input: Box<LogicalOperator>,
        source_variable: String,
        target_variable: String,
        relationship_types: Vec<String>,
        direction: RelationshipDirection,
        min_length: Option<u32>,
        max_length: Option<u32>,
    },

    /// Project specific columns (RETURN clause)
    Project {
        input: Box<LogicalOperator>,
        projections: Vec<ProjectionItem>,
    },

    /// Join multiple disconnected patterns
    Join {
        left: Box<LogicalOperator>,
        right: Box<LogicalOperator>,
        join_type: JoinType,
    },

    /// Apply DISTINCT
    Distinct { input: Box<LogicalOperator> },

    /// Apply ORDER BY
    Sort {
        input: Box<LogicalOperator>,
        sort_items: Vec<SortItem>,
    },

    /// Apply LIMIT
    Limit {
        input: Box<LogicalOperator>,
        count: u64,
    },
}

/// Projection item for SELECT/RETURN clauses
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionItem {
    pub expression: ValueExpression,
    pub alias: Option<String>,
}

/// Join types for combining multiple patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Sort specification for ORDER BY
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortItem {
    pub expression: ValueExpression,
    pub direction: SortDirection,
}

/// Logical plan builder - converts AST to logical plan
pub struct LogicalPlanner {
    /// Track variables in scope
    variables: HashMap<String, String>, // variable -> label
}

impl LogicalPlanner {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Convert a Cypher AST to a logical plan
    pub fn plan(&mut self, query: &CypherQuery) -> Result<LogicalOperator> {
        // Start with the MATCH clause(s)
        let mut plan = self.plan_match_clauses(&query.match_clauses)?;

        // Apply WHERE clause if present
        if let Some(where_clause) = &query.where_clause {
            plan = LogicalOperator::Filter {
                input: Box::new(plan),
                predicate: where_clause.expression.clone(),
            };
        }

        // Apply RETURN clause
        plan = self.plan_return_clause(&query.return_clause, plan)?;

        // Apply ORDER BY if present
        if let Some(order_by) = &query.order_by {
            plan = LogicalOperator::Sort {
                input: Box::new(plan),
                sort_items: order_by
                    .items
                    .iter()
                    .map(|item| SortItem {
                        expression: item.expression.clone(),
                        direction: item.direction.clone(),
                    })
                    .collect(),
            };
        }

        // Apply LIMIT if present
        if let Some(limit) = query.limit {
            plan = LogicalOperator::Limit {
                input: Box::new(plan),
                count: limit,
            };
        }

        Ok(plan)
    }

    /// Plan MATCH clauses - the core graph pattern matching
    fn plan_match_clauses(&mut self, match_clauses: &[MatchClause]) -> Result<LogicalOperator> {
        if match_clauses.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one MATCH clause".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let plan = match_clauses.iter().try_fold(None, |plan, clause| {
            self.plan_match_clause_with_base(plan, clause).map(Some)
        })?;

        plan.ok_or_else(|| GraphError::PlanError {
            message: "Failed to plan MATCH clauses".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Plan a single MATCH clause, optionally starting from an existing base plan
    fn plan_match_clause_with_base(
        &mut self,
        base: Option<LogicalOperator>,
        match_clause: &MatchClause,
    ) -> Result<LogicalOperator> {
        if match_clause.patterns.is_empty() {
            return Err(GraphError::PlanError {
                message: "MATCH clause must have at least one pattern".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let mut plan = base;
        for pattern in &match_clause.patterns {
            match pattern {
                GraphPattern::Node(node) => {
                    let already_bound = node
                        .variable
                        .as_deref()
                        .is_some_and(|v| self.variables.contains_key(v));

                    match (already_bound, plan.as_ref()) {
                        (true, _) => { /* no-op */ }
                        (false, None) => plan = Some(self.plan_node_scan(node)?),
                        (false, Some(_)) => {
                            let right = self.plan_node_scan(node)?;
                            plan = Some(LogicalOperator::Join {
                                left: Box::new(plan.unwrap()),
                                right: Box::new(right),
                                join_type: JoinType::Cross, // TODO: infer better join type based on shared vars
                            });
                        }
                    }
                }
                GraphPattern::Path(path) => plan = Some(self.plan_path(plan, path)?),
            }
        }

        plan.ok_or_else(|| GraphError::PlanError {
            message: "Failed to plan MATCH clause".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Plan a node scan (ScanByLabel)
    fn plan_node_scan(&mut self, node: &NodePattern) -> Result<LogicalOperator> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_node_{}", self.variables.len()));

        let label = node
            .labels
            .first()
            .cloned()
            .unwrap_or_else(|| "Node".to_string());

        // Register variable
        self.variables.insert(variable.clone(), label.clone());

        Ok(LogicalOperator::ScanByLabel {
            variable,
            label,
            properties: node.properties.clone(),
        })
    }

    // (removed) plan_path_segment is superseded by plan_path

    /// Plan a full path pattern, respecting the starting variable if provided
    fn plan_path(
        &mut self,
        base: Option<LogicalOperator>,
        path: &PathPattern,
    ) -> Result<LogicalOperator> {
        // Establish a base plan
        let mut plan = if let Some(p) = base {
            p
        } else {
            self.plan_node_scan(&path.start_node)?
        };

        // Determine the current source variable for the first hop
        let mut current_src = match &path.start_node.variable {
            Some(var) => var.clone(),
            None => self.extract_variable_from_plan(&plan)?,
        };

        // For each segment, add an expand
        for segment in &path.segments {
            // Determine / register target variable
            let target_variable = segment
                .end_node
                .variable
                .clone()
                .unwrap_or_else(|| format!("_node_{}", self.variables.len()));

            let target_label = segment
                .end_node
                .labels
                .first()
                .cloned()
                .unwrap_or_else(|| "Node".to_string());
            self.variables.insert(target_variable.clone(), target_label);

            // Optimize fixed-length var-length expansions (*1 or *1..1)
            let next_plan = match segment.relationship.length.as_ref() {
                Some(length_range)
                    if length_range.min == Some(1) && length_range.max == Some(1) =>
                {
                    LogicalOperator::Expand {
                        input: Box::new(plan),
                        source_variable: current_src.clone(),
                        target_variable: target_variable.clone(),
                        relationship_types: segment.relationship.types.clone(),
                        direction: segment.relationship.direction.clone(),
                        properties: segment.relationship.properties.clone(),
                    }
                }
                Some(length_range) => LogicalOperator::VariableLengthExpand {
                    input: Box::new(plan),
                    source_variable: current_src.clone(),
                    target_variable: target_variable.clone(),
                    relationship_types: segment.relationship.types.clone(),
                    direction: segment.relationship.direction.clone(),
                    min_length: length_range.min,
                    max_length: length_range.max,
                },
                None => LogicalOperator::Expand {
                    input: Box::new(plan),
                    source_variable: current_src.clone(),
                    target_variable: target_variable.clone(),
                    relationship_types: segment.relationship.types.clone(),
                    direction: segment.relationship.direction.clone(),
                    properties: segment.relationship.properties.clone(),
                },
            };

            plan = next_plan;
            current_src = target_variable;
        }

        Ok(plan)
    }

    /// Extract the main variable from a logical plan (for chaining)
    #[allow(clippy::only_used_in_recursion)]
    fn extract_variable_from_plan(&self, plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::ScanByLabel { variable, .. } => Ok(variable.clone()),
            LogicalOperator::Expand {
                target_variable, ..
            } => Ok(target_variable.clone()),
            LogicalOperator::VariableLengthExpand {
                target_variable, ..
            } => Ok(target_variable.clone()),
            LogicalOperator::Filter { input, .. } => self.extract_variable_from_plan(input),
            LogicalOperator::Project { input, .. } => self.extract_variable_from_plan(input),
            LogicalOperator::Distinct { input } => self.extract_variable_from_plan(input),
            LogicalOperator::Sort { input, .. } => self.extract_variable_from_plan(input),
            LogicalOperator::Limit { input, .. } => self.extract_variable_from_plan(input),
            LogicalOperator::Join { left, right, .. } => {
                // Prefer the right branch's tail variable, else fall back to left
                self.extract_variable_from_plan(right)
                    .or_else(|_| self.extract_variable_from_plan(left))
            }
        }
    }

    /// Plan RETURN clause (Project)
    fn plan_return_clause(
        &self,
        return_clause: &ReturnClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let projections = return_clause
            .items
            .iter()
            .map(|item| ProjectionItem {
                expression: item.expression.clone(),
                alias: item.alias.clone(),
            })
            .collect();

        let mut plan = LogicalOperator::Project {
            input: Box::new(input),
            projections,
        };

        // Add DISTINCT if needed
        if return_clause.distinct {
            plan = LogicalOperator::Distinct {
                input: Box::new(plan),
            };
        }

        Ok(plan)
    }
}

impl Default for LogicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_cypher_query;

    #[test]
    fn test_relationship_query_logical_plan_structure() {
        let query_text = r#"MATCH (p:Person {name: "Alice"})-[:KNOWS]->(f:Person) WHERE f.age > 30 RETURN f.name"#;

        // Parse the query
        let ast = parse_cypher_query(query_text).unwrap();

        // Plan to logical operators
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Verify the overall structure is a projection
        match &logical_plan {
            LogicalOperator::Project { input, projections } => {
                // Verify projection is f.name
                assert_eq!(projections.len(), 1);
                match &projections[0].expression {
                    ValueExpression::Property(prop_ref) => {
                        assert_eq!(prop_ref.variable, "f");
                        assert_eq!(prop_ref.property, "name");
                    }
                    _ => panic!("Expected property reference for f.name"),
                }

                // Verify input is a filter for f.age > 30
                match input.as_ref() {
                    LogicalOperator::Filter {
                        predicate,
                        input: filter_input,
                    } => {
                        // Verify the predicate is f.age > 30
                        match predicate {
                            BooleanExpression::Comparison {
                                left,
                                operator,
                                right,
                            } => {
                                match left {
                                    ValueExpression::Property(prop_ref) => {
                                        assert_eq!(prop_ref.variable, "f");
                                        assert_eq!(prop_ref.property, "age");
                                    }
                                    _ => panic!("Expected property reference for f.age"),
                                }
                                assert_eq!(*operator, ComparisonOperator::GreaterThan);
                                match right {
                                    ValueExpression::Literal(PropertyValue::Integer(val)) => {
                                        assert_eq!(*val, 30);
                                    }
                                    _ => panic!("Expected integer literal 30"),
                                }
                            }
                            _ => panic!("Expected comparison expression"),
                        }

                        // Verify the input to the filter is an expand operation
                        match filter_input.as_ref() {
                            LogicalOperator::Expand {
                                input: expand_input,
                                source_variable,
                                target_variable,
                                relationship_types,
                                direction,
                                ..
                            } => {
                                assert_eq!(source_variable, "p");
                                assert_eq!(target_variable, "f");
                                assert_eq!(relationship_types, &vec!["KNOWS".to_string()]);
                                assert_eq!(*direction, RelationshipDirection::Outgoing);

                                // Verify the input to expand is a scan with properties for p.name = 'Alice'
                                match expand_input.as_ref() {
                                    LogicalOperator::ScanByLabel {
                                        variable,
                                        label,
                                        properties,
                                    } => {
                                        assert_eq!(variable, "p");
                                        assert_eq!(label, "Person");

                                        // Verify the properties contain name = "Alice"
                                        assert_eq!(properties.len(), 1);
                                        match properties.get("name") {
                                            Some(PropertyValue::String(val)) => {
                                                assert_eq!(val, "Alice");
                                            }
                                            _ => {
                                                panic!("Expected name property with value 'Alice'")
                                            }
                                        }
                                    }
                                    _ => panic!("Expected ScanByLabel with properties for Person"),
                                }
                            }
                            _ => panic!("Expected Expand operation"),
                        }
                    }
                    _ => panic!("Expected Filter for f.age > 30"),
                }
            }
            _ => panic!("Expected Project at the top level"),
        }
    }

    #[test]
    fn test_simple_node_query_logical_plan() {
        let query_text = "MATCH (n:Person) RETURN n.name";

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Should be: Project { input: ScanByLabel }
        match &logical_plan {
            LogicalOperator::Project { input, projections } => {
                assert_eq!(projections.len(), 1);
                match input.as_ref() {
                    LogicalOperator::ScanByLabel {
                        variable, label, ..
                    } => {
                        assert_eq!(variable, "n");
                        assert_eq!(label, "Person");
                    }
                    _ => panic!("Expected ScanByLabel"),
                }
            }
            _ => panic!("Expected Project"),
        }
    }

    #[test]
    fn test_node_with_properties_logical_plan() {
        let query_text = "MATCH (n:Person {age: 25}) RETURN n.name";

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Should be: Project { input: ScanByLabel with properties }
        // Properties from MATCH clause are pushed down to the scan level
        match &logical_plan {
            LogicalOperator::Project { input, .. } => {
                match input.as_ref() {
                    LogicalOperator::ScanByLabel {
                        variable,
                        label,
                        properties,
                    } => {
                        assert_eq!(variable, "n");
                        assert_eq!(label, "Person");

                        // Verify the properties are in the scan
                        assert_eq!(properties.len(), 1);
                        match properties.get("age") {
                            Some(PropertyValue::Integer(25)) => {}
                            _ => panic!("Expected age property with value 25"),
                        }
                    }
                    _ => panic!("Expected ScanByLabel with properties"),
                }
            }
            _ => panic!("Expected Project"),
        }
    }

    #[test]
    fn test_variable_length_path_logical_plan() {
        let query_text = "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN b.name";

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Should be: Project { input: VariableLengthExpand { input: ScanByLabel } }
        match &logical_plan {
            LogicalOperator::Project { input, .. } => match input.as_ref() {
                LogicalOperator::VariableLengthExpand {
                    input: expand_input,
                    source_variable,
                    target_variable,
                    relationship_types,
                    min_length,
                    max_length,
                    ..
                } => {
                    assert_eq!(source_variable, "a");
                    assert_eq!(target_variable, "b");
                    assert_eq!(relationship_types, &vec!["KNOWS".to_string()]);
                    assert_eq!(*min_length, Some(1));
                    assert_eq!(*max_length, Some(2));

                    match expand_input.as_ref() {
                        LogicalOperator::ScanByLabel {
                            variable, label, ..
                        } => {
                            assert_eq!(variable, "a");
                            assert_eq!(label, "Person");
                        }
                        _ => panic!("Expected ScanByLabel"),
                    }
                }
                _ => panic!("Expected VariableLengthExpand"),
            },
            _ => panic!("Expected Project"),
        }
    }

    #[test]
    fn test_where_clause_logical_plan() {
        // Note: Current parser only supports simple comparisons, not AND/OR
        let query_text = r#"MATCH (n:Person) WHERE n.age > 25 RETURN n.name"#;

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Should be: Project { input: Filter { input: ScanByLabel } }
        match &logical_plan {
            LogicalOperator::Project { input, .. } => {
                match input.as_ref() {
                    LogicalOperator::Filter {
                        predicate,
                        input: scan_input,
                    } => {
                        // Verify it's a simple comparison: n.age > 25
                        match predicate {
                            BooleanExpression::Comparison {
                                left,
                                operator,
                                right: _,
                            } => {
                                match left {
                                    ValueExpression::Property(prop_ref) => {
                                        assert_eq!(prop_ref.variable, "n");
                                        assert_eq!(prop_ref.property, "age");
                                    }
                                    _ => panic!("Expected property reference for age"),
                                }
                                assert_eq!(*operator, ComparisonOperator::GreaterThan);
                            }
                            _ => panic!("Expected comparison expression"),
                        }

                        match scan_input.as_ref() {
                            LogicalOperator::ScanByLabel { .. } => {}
                            _ => panic!("Expected ScanByLabel"),
                        }
                    }
                    _ => panic!("Expected Filter"),
                }
            }
            _ => panic!("Expected Project"),
        }
    }

    #[test]
    fn test_multiple_node_patterns_join_in_match() {
        let query_text = "MATCH (a:Person), (b:Company) RETURN a.name, b.name";

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Expect: Project { input: Join { left: Scan(a:Person), right: Scan(b:Company) } }
        match &logical_plan {
            LogicalOperator::Project { input, projections } => {
                assert_eq!(projections.len(), 2);
                match input.as_ref() {
                    LogicalOperator::Join {
                        left,
                        right,
                        join_type,
                    } => {
                        assert!(matches!(join_type, JoinType::Cross));
                        match left.as_ref() {
                            LogicalOperator::ScanByLabel {
                                variable, label, ..
                            } => {
                                assert_eq!(variable, "a");
                                assert_eq!(label, "Person");
                            }
                            _ => panic!("Expected left ScanByLabel for a:Person"),
                        }
                        match right.as_ref() {
                            LogicalOperator::ScanByLabel {
                                variable, label, ..
                            } => {
                                assert_eq!(variable, "b");
                                assert_eq!(label, "Company");
                            }
                            _ => panic!("Expected right ScanByLabel for b:Company"),
                        }
                    }
                    _ => panic!("Expected Join after Project"),
                }
            }
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_shared_variable_chained_paths_in_match() {
        let query_text =
            "MATCH (a:Person)-[:KNOWS]->(b:Person), (b)-[:LIKES]->(c:Thing) RETURN c.name";

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        // Expect: Project { input: Expand (b->c) { input: Expand (a->b) { input: Scan(a) } } }
        match &logical_plan {
            LogicalOperator::Project { input, .. } => match input.as_ref() {
                LogicalOperator::Expand {
                    source_variable: src2,
                    target_variable: tgt2,
                    input: inner2,
                    ..
                } => {
                    assert_eq!(src2, "b");
                    assert_eq!(tgt2, "c");
                    match inner2.as_ref() {
                        LogicalOperator::Expand {
                            source_variable: src1,
                            target_variable: tgt1,
                            input: inner1,
                            ..
                        } => {
                            assert_eq!(src1, "a");
                            assert_eq!(tgt1, "b");
                            match inner1.as_ref() {
                                LogicalOperator::ScanByLabel {
                                    variable, label, ..
                                } => {
                                    assert_eq!(variable, "a");
                                    assert_eq!(label, "Person");
                                }
                                _ => panic!("Expected ScanByLabel for a:Person"),
                            }
                        }
                        _ => panic!("Expected first Expand a->b"),
                    }
                }
                _ => panic!("Expected second Expand b->c at top of input"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_fixed_length_variable_path_is_expand() {
        let query_text = "MATCH (a:Person)-[:KNOWS*1..1]->(b:Person) RETURN b.name";

        let ast = parse_cypher_query(query_text).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical_plan = planner.plan(&ast).unwrap();

        match &logical_plan {
            LogicalOperator::Project { input, .. } => match input.as_ref() {
                LogicalOperator::Expand {
                    source_variable,
                    target_variable,
                    ..
                } => {
                    assert_eq!(source_variable, "a");
                    assert_eq!(target_variable, "b");
                }
                _ => panic!("Expected Expand for fixed-length *1..1"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_distinct_and_order_limit_wrapping() {
        // DISTINCT should wrap Project with Distinct
        let q1 = "MATCH (n:Person) RETURN DISTINCT n.name";
        let ast1 = parse_cypher_query(q1).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical1 = planner.plan(&ast1).unwrap();
        match logical1 {
            LogicalOperator::Distinct { input } => match *input {
                LogicalOperator::Project { .. } => {}
                _ => panic!("Expected Project under Distinct"),
            },
            _ => panic!("Expected Distinct at top level"),
        }

        // ORDER BY + LIMIT should be Limit(Sort(Project(..)))
        let q2 = "MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 10";
        let ast2 = parse_cypher_query(q2).unwrap();
        let mut planner2 = LogicalPlanner::new();
        let logical2 = planner2.plan(&ast2).unwrap();
        match logical2 {
            LogicalOperator::Limit { input, count } => {
                assert_eq!(count, 10);
                match *input {
                    LogicalOperator::Sort { input: inner, .. } => match *inner {
                        LogicalOperator::Project { .. } => {}
                        _ => panic!("Expected Project under Sort"),
                    },
                    _ => panic!("Expected Sort under Limit"),
                }
            }
            _ => panic!("Expected Limit at top level"),
        }
    }

    #[test]
    fn test_relationship_properties_pushed_into_expand() {
        let q = "MATCH (a)-[:KNOWS {since: 2020}]->(b) RETURN b";
        let ast = parse_cypher_query(q).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical = planner.plan(&ast).unwrap();
        match logical {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::Expand { properties, .. } => match properties.get("since") {
                    Some(PropertyValue::Integer(2020)) => {}
                    _ => panic!("Expected relationship property since=2020 in Expand"),
                },
                _ => panic!("Expected Expand under Project"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_multiple_match_clauses_cross_join() {
        let q = "MATCH (a:Person) MATCH (b:Company) RETURN a.name, b.name";
        let ast = parse_cypher_query(q).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical = planner.plan(&ast).unwrap();
        match logical {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::Join {
                    left,
                    right,
                    join_type,
                } => {
                    assert!(matches!(join_type, JoinType::Cross));
                    match (*left, *right) {
                        (
                            LogicalOperator::ScanByLabel {
                                variable: va,
                                label: la,
                                ..
                            },
                            LogicalOperator::ScanByLabel {
                                variable: vb,
                                label: lb,
                                ..
                            },
                        ) => {
                            assert_eq!(va, "a");
                            assert_eq!(la, "Person");
                            assert_eq!(vb, "b");
                            assert_eq!(lb, "Company");
                        }
                        _ => panic!("Expected two scans under Join"),
                    }
                }
                _ => panic!("Expected Join under Project"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_variable_only_node_default_label() {
        let q = "MATCH (x) RETURN x";
        let ast = parse_cypher_query(q).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical = planner.plan(&ast).unwrap();
        match logical {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::ScanByLabel {
                    variable, label, ..
                } => {
                    assert_eq!(variable, "x");
                    assert_eq!(label, "Node");
                }
                _ => panic!("Expected ScanByLabel under Project"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_multi_label_node_uses_first_label() {
        let q = "MATCH (n:Person:Employee) RETURN n";
        let ast = parse_cypher_query(q).unwrap();
        let mut planner = LogicalPlanner::new();
        let logical = planner.plan(&ast).unwrap();
        match logical {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::ScanByLabel { label, .. } => {
                    assert_eq!(label, "Person");
                }
                _ => panic!("Expected ScanByLabel under Project"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }

    #[test]
    fn test_open_ended_and_partial_var_length_ranges() {
        // * (unbounded)
        let q1 = "MATCH (a)-[:R*]->(b) RETURN b";
        let ast1 = parse_cypher_query(q1).unwrap();
        let mut planner1 = LogicalPlanner::new();
        let plan1 = planner1.plan(&ast1).unwrap();
        match plan1 {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::VariableLengthExpand {
                    min_length,
                    max_length,
                    ..
                } => {
                    assert_eq!(min_length, None);
                    assert_eq!(max_length, None);
                }
                _ => panic!("Expected VariableLengthExpand for *"),
            },
            _ => panic!("Expected Project at top level"),
        }

        // *2.. (min only)
        let q2 = "MATCH (a)-[:R*2..]->(b) RETURN b";
        let ast2 = parse_cypher_query(q2).unwrap();
        let mut planner2 = LogicalPlanner::new();
        let plan2 = planner2.plan(&ast2).unwrap();
        match plan2 {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::VariableLengthExpand {
                    min_length,
                    max_length,
                    ..
                } => {
                    assert_eq!(min_length, Some(2));
                    assert_eq!(max_length, None);
                }
                _ => panic!("Expected VariableLengthExpand for *2.."),
            },
            _ => panic!("Expected Project at top level"),
        }

        // *..3 (max only)
        let q3 = "MATCH (a)-[:R*..3]->(b) RETURN b";
        let ast3 = parse_cypher_query(q3).unwrap();
        let mut planner3 = LogicalPlanner::new();
        let plan3 = planner3.plan(&ast3).unwrap();
        match plan3 {
            LogicalOperator::Project { input, .. } => match *input {
                LogicalOperator::VariableLengthExpand {
                    min_length,
                    max_length,
                    ..
                } => {
                    assert_eq!(min_length, None);
                    assert_eq!(max_length, Some(3));
                }
                _ => panic!("Expected VariableLengthExpand for *..3"),
            },
            _ => panic!("Expected Project at top level"),
        }
    }
}
