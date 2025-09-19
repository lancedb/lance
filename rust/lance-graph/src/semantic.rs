// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Semantic analysis for graph queries
//!
//! This module implements the semantic analysis phase of the query pipeline:
//! Parse → **Semantic Analysis** → Logical Plan → Physical Plan
//!
//! Semantic analysis validates the query and enriches the AST with type information.

use crate::ast::*;
use crate::config::GraphConfig;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};

/// Semantic analyzer - validates and enriches the AST
pub struct SemanticAnalyzer {
    config: GraphConfig,
    variables: HashMap<String, VariableInfo>,
    current_scope: ScopeType,
}

/// Information about a variable in the query
#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub name: String,
    pub variable_type: VariableType,
    pub labels: Vec<String>,
    pub properties: HashSet<String>,
    pub defined_in: ScopeType,
}

/// Type of a variable
#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    Node,
    Relationship,
    Path,
    Property,
}

/// Scope where a variable is defined
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeType {
    Match,
    Where,
    Return,
    OrderBy,
}

/// Semantic analysis result with validated and enriched AST
#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub query: CypherQuery,
    pub variables: HashMap<String, VariableInfo>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl SemanticAnalyzer {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            variables: HashMap::new(),
            current_scope: ScopeType::Match,
        }
    }

    /// Analyze a Cypher query AST
    pub fn analyze(&mut self, query: &CypherQuery) -> Result<SemanticResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Phase 1: Variable discovery in MATCH clauses
        self.current_scope = ScopeType::Match;
        for match_clause in &query.match_clauses {
            if let Err(e) = self.analyze_match_clause(match_clause) {
                errors.push(format!("MATCH clause error: {}", e));
            }
        }

        // Phase 2: Validate WHERE clause
        if let Some(where_clause) = &query.where_clause {
            self.current_scope = ScopeType::Where;
            if let Err(e) = self.analyze_where_clause(where_clause) {
                errors.push(format!("WHERE clause error: {}", e));
            }
        }

        // Phase 3: Validate RETURN clause
        self.current_scope = ScopeType::Return;
        if let Err(e) = self.analyze_return_clause(&query.return_clause) {
            errors.push(format!("RETURN clause error: {}", e));
        }

        // Phase 4: Validate ORDER BY clause
        if let Some(order_by) = &query.order_by {
            self.current_scope = ScopeType::OrderBy;
            if let Err(e) = self.analyze_order_by_clause(order_by) {
                errors.push(format!("ORDER BY clause error: {}", e));
            }
        }

        // Phase 5: Schema validation
        self.validate_schema(&mut warnings);

        // Phase 6: Type checking
        self.validate_types(&mut errors);

        Ok(SemanticResult {
            query: query.clone(),
            variables: self.variables.clone(),
            errors,
            warnings,
        })
    }

    /// Analyze MATCH clause and discover variables
    fn analyze_match_clause(&mut self, match_clause: &MatchClause) -> Result<()> {
        for pattern in &match_clause.patterns {
            self.analyze_graph_pattern(pattern)?;
        }
        Ok(())
    }

    /// Analyze a graph pattern and register variables
    fn analyze_graph_pattern(&mut self, pattern: &GraphPattern) -> Result<()> {
        match pattern {
            GraphPattern::Node(node) => {
                self.register_node_variable(node)?;
            }
            GraphPattern::Path(path) => {
                // Register start node
                self.register_node_variable(&path.start_node)?;

                // Register variables in each segment
                for segment in &path.segments {
                    // Register relationship variable if present
                    if let Some(rel_var) = &segment.relationship.variable {
                        self.register_relationship_variable(rel_var, &segment.relationship)?;
                    }

                    // Register end node
                    self.register_node_variable(&segment.end_node)?;
                }
            }
        }
        Ok(())
    }

    /// Register a node variable
    fn register_node_variable(&mut self, node: &NodePattern) -> Result<()> {
        if let Some(var_name) = &node.variable {
            let var_info = VariableInfo {
                name: var_name.clone(),
                variable_type: VariableType::Node,
                labels: node.labels.clone(),
                properties: node.properties.keys().cloned().collect(),
                defined_in: self.current_scope.clone(),
            };

            if self.variables.contains_key(var_name) {
                // Variable redefinition - check if it's consistent
                let existing = &self.variables[var_name];
                if existing.variable_type != VariableType::Node {
                    return Err(GraphError::PlanError {
                        message: format!("Variable '{}' redefined with different type", var_name),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }
            }

            self.variables.insert(var_name.clone(), var_info);
        }
        Ok(())
    }

    /// Register a relationship variable
    fn register_relationship_variable(
        &mut self,
        var_name: &str,
        rel: &RelationshipPattern,
    ) -> Result<()> {
        let var_info = VariableInfo {
            name: var_name.to_string(),
            variable_type: VariableType::Relationship,
            labels: rel.types.clone(), // Relationship types are like labels
            properties: rel.properties.keys().cloned().collect(),
            defined_in: self.current_scope.clone(),
        };

        if self.variables.contains_key(var_name) {
            let existing = &self.variables[var_name];
            if existing.variable_type != VariableType::Relationship {
                return Err(GraphError::PlanError {
                    message: format!("Variable '{}' redefined with different type", var_name),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        }

        self.variables.insert(var_name.to_string(), var_info);
        Ok(())
    }

    /// Analyze WHERE clause
    fn analyze_where_clause(&mut self, where_clause: &WhereClause) -> Result<()> {
        self.analyze_boolean_expression(&where_clause.expression)
    }

    /// Analyze boolean expression and check variable references
    fn analyze_boolean_expression(&mut self, expr: &BooleanExpression) -> Result<()> {
        match expr {
            BooleanExpression::Comparison { left, right, .. } => {
                self.analyze_value_expression(left)?;
                self.analyze_value_expression(right)?;
            }
            BooleanExpression::And(left, right) | BooleanExpression::Or(left, right) => {
                self.analyze_boolean_expression(left)?;
                self.analyze_boolean_expression(right)?;
            }
            BooleanExpression::Not(inner) => {
                self.analyze_boolean_expression(inner)?;
            }
            BooleanExpression::Exists(prop_ref) => {
                self.validate_property_reference(prop_ref)?;
            }
            BooleanExpression::In { expression, list } => {
                self.analyze_value_expression(expression)?;
                for item in list {
                    self.analyze_value_expression(item)?;
                }
            }
            BooleanExpression::Like { expression, .. } => {
                self.analyze_value_expression(expression)?;
            }
        }
        Ok(())
    }

    /// Analyze value expression and check variable references
    fn analyze_value_expression(&mut self, expr: &ValueExpression) -> Result<()> {
        match expr {
            ValueExpression::Property(prop_ref) => {
                self.validate_property_reference(prop_ref)?;
            }
            ValueExpression::Literal(_) => {
                // Literals are always valid
            }
            ValueExpression::Variable(var) => {
                if !self.variables.contains_key(var) {
                    return Err(GraphError::PlanError {
                        message: format!("Undefined variable: '{}'", var),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }
            }
            ValueExpression::Function { .. } => {
                // TODO: Implement function validation
            }
            ValueExpression::Arithmetic { .. } => {
                // TODO: Implement arithmetic validation
            }
        }
        Ok(())
    }

    /// Validate property reference
    fn validate_property_reference(&self, prop_ref: &PropertyRef) -> Result<()> {
        if !self.variables.contains_key(&prop_ref.variable) {
            return Err(GraphError::PlanError {
                message: format!("Undefined variable: '{}'", prop_ref.variable),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }
        Ok(())
    }

    /// Analyze RETURN clause
    fn analyze_return_clause(&mut self, return_clause: &ReturnClause) -> Result<()> {
        for item in &return_clause.items {
            self.analyze_value_expression(&item.expression)?;
        }
        Ok(())
    }

    /// Analyze ORDER BY clause
    fn analyze_order_by_clause(&mut self, order_by: &OrderByClause) -> Result<()> {
        for item in &order_by.items {
            self.analyze_value_expression(&item.expression)?;
        }
        Ok(())
    }

    /// Validate schema references against configuration
    fn validate_schema(&self, warnings: &mut Vec<String>) {
        for var_info in self.variables.values() {
            match var_info.variable_type {
                VariableType::Node => {
                    for label in &var_info.labels {
                        if self.config.get_node_mapping(label).is_none() {
                            warnings.push(format!("Node label '{}' not found in schema", label));
                        }
                    }
                }
                VariableType::Relationship => {
                    for rel_type in &var_info.labels {
                        if self.config.get_relationship_mapping(rel_type).is_none() {
                            warnings.push(format!(
                                "Relationship type '{}' not found in schema",
                                rel_type
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Validate types and operations
    fn validate_types(&self, errors: &mut Vec<String>) {
        // TODO: Implement type checking
        // - Check that properties exist on nodes/relationships
        // - Check that comparison operations are valid for data types
        // - Check that arithmetic operations are valid

        // For now, just check that variables are properly scoped
        for var_info in self.variables.values() {
            if var_info.defined_in == ScopeType::Return
                && !self
                    .variables
                    .values()
                    .any(|v| v.name == var_info.name && v.defined_in == ScopeType::Match)
            {
                errors.push(format!(
                    "Variable '{}' used in RETURN but not defined in MATCH",
                    var_info.name
                ));
            }
        }
    }
}
