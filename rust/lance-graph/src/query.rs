// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-level Cypher query interface for Lance datasets

use crate::ast::CypherQuery as CypherAST;
use crate::config::GraphConfig;
use crate::error::{GraphError, Result};
use crate::parser::parse_cypher_query;
use datafusion::logical_expr::JoinType;
use std::collections::HashMap;

/// A Cypher query that can be executed against Lance datasets
#[derive(Debug, Clone)]
pub struct CypherQuery {
    /// The original Cypher query string
    query_text: String,
    /// Parsed AST representation
    ast: CypherAST,
    /// Graph configuration for mapping
    config: Option<GraphConfig>,
    /// Query parameters
    parameters: HashMap<String, serde_json::Value>,
}

// Internal helper that plans and executes a single path by chaining joins.
struct PathExecutor<'a> {
    ctx: &'a datafusion::prelude::SessionContext,
    path: &'a crate::ast::PathPattern,
    start_label: &'a str,
    start_alias: String,
    segs: Vec<SegMeta<'a>>,
    node_maps: std::collections::HashMap<String, &'a crate::config::NodeMapping>,
    rel_maps: std::collections::HashMap<String, &'a crate::config::RelationshipMapping>,
}

#[derive(Clone)]
struct SegMeta<'a> {
    rel_type: &'a str,
    end_label: &'a str,
    dir: crate::ast::RelationshipDirection,
    rel_alias: String,
    end_alias: String,
}

impl<'a> PathExecutor<'a> {
    fn new(
        ctx: &'a datafusion::prelude::SessionContext,
        cfg: &'a crate::config::GraphConfig,
        path: &'a crate::ast::PathPattern,
    ) -> Result<Self> {
        use std::collections::{HashMap, HashSet};
        let mut used: HashSet<String> = HashSet::new();
        let mut uniq = |desired: &str| -> String {
            if used.insert(desired.to_string()) {
                return desired.to_string();
            }
            let mut i = 2usize;
            loop {
                let cand = format!("{}_{}", desired, i);
                if used.insert(cand.clone()) {
                    break cand;
                }
                i += 1;
            }
        };

        let start_label = path
            .start_node
            .labels
            .first()
            .map(|s| s.as_str())
            .ok_or_else(|| GraphError::PlanError {
                message: "Start node must have a label".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let start_alias = uniq(
            &path
                .start_node
                .variable
                .as_deref()
                .unwrap_or(start_label)
                .to_lowercase(),
        );

        let mut segs: Vec<SegMeta> = Vec::with_capacity(path.segments.len());
        for seg in &path.segments {
            let rel_type = seg
                .relationship
                .types
                .first()
                .map(|s| s.as_str())
                .ok_or_else(|| GraphError::PlanError {
                    message: "Relationship must have a type".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            let end_label = seg
                .end_node
                .labels
                .first()
                .map(|s| s.as_str())
                .ok_or_else(|| GraphError::PlanError {
                    message: "End node must have a label".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            let rel_alias = uniq(
                &seg.relationship
                    .variable
                    .as_deref()
                    .unwrap_or(rel_type)
                    .to_lowercase(),
            );
            let end_alias = uniq(
                &seg.end_node
                    .variable
                    .as_deref()
                    .unwrap_or(end_label)
                    .to_lowercase(),
            );
            segs.push(SegMeta {
                rel_type,
                end_label,
                dir: seg.relationship.direction.clone(),
                rel_alias,
                end_alias,
            });
        }

        let mut node_maps: HashMap<String, &crate::config::NodeMapping> = HashMap::new();
        let mut rel_maps: HashMap<String, &crate::config::RelationshipMapping> = HashMap::new();
        node_maps.insert(
            start_alias.clone(),
            cfg.get_node_mapping(start_label)
                .ok_or_else(|| GraphError::PlanError {
                    message: format!("No node mapping for '{}'", start_label),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?,
        );
        for seg in &segs {
            node_maps.insert(
                seg.end_alias.clone(),
                cfg.get_node_mapping(seg.end_label)
                    .ok_or_else(|| GraphError::PlanError {
                        message: format!("No node mapping for '{}'", seg.end_label),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?,
            );
            rel_maps.insert(
                seg.rel_alias.clone(),
                cfg.get_relationship_mapping(seg.rel_type).ok_or_else(|| {
                    GraphError::PlanError {
                        message: format!("No relationship mapping for '{}'", seg.rel_type),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?,
            );
        }

        Ok(Self {
            ctx,
            path,
            start_label,
            start_alias,
            segs,
            node_maps,
            rel_maps,
        })
    }

    async fn open_aliased(
        &self,
        table: &str,
        alias: &str,
    ) -> Result<datafusion::dataframe::DataFrame> {
        let df = self
            .ctx
            .table(table)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", table, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let schema = df.schema();
        let proj: Vec<datafusion::logical_expr::Expr> = schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!("{}__{}", alias, f.name()))
            })
            .collect();
        df.alias(alias)?
            .select(proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", table, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    async fn build_chain(&self) -> Result<datafusion::dataframe::DataFrame> {
        // Start node
        let mut df = self
            .open_aliased(self.start_label, &self.start_alias)
            .await?;
        // Inline property filters on start node
        for (k, v) in &self.path.start_node.properties {
            let expr = to_df_literal(v);
            df = df
                .filter(datafusion::logical_expr::Expr::BinaryExpr(
                    datafusion::logical_expr::BinaryExpr {
                        left: Box::new(datafusion::logical_expr::col(format!(
                            "{}__{}",
                            self.start_alias, k
                        ))),
                        op: datafusion::logical_expr::Operator::Eq,
                        right: Box::new(expr),
                    },
                ))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        // Chain joins for each hop
        let mut current_node_alias = self.start_alias.as_str();
        for s in &self.segs {
            let rel_df = self.open_aliased(s.rel_type, &s.rel_alias).await?;
            let node_map = self.node_maps.get(current_node_alias).unwrap();
            let rel_map = self.rel_maps.get(&s.rel_alias).unwrap();
            let (left_key, right_key) = match s.dir {
                crate::ast::RelationshipDirection::Outgoing
                | crate::ast::RelationshipDirection::Undirected => (
                    format!("{}__{}", current_node_alias, node_map.id_field),
                    format!("{}__{}", s.rel_alias, rel_map.source_id_field),
                ),
                crate::ast::RelationshipDirection::Incoming => (
                    format!("{}__{}", current_node_alias, node_map.id_field),
                    format!("{}__{}", s.rel_alias, rel_map.target_id_field),
                ),
            };
            df = df
                .join(
                    rel_df,
                    JoinType::Inner,
                    &[left_key.as_str()],
                    &[right_key.as_str()],
                    None,
                )
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (node->rel): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let end_df = self.open_aliased(s.end_label, &s.end_alias).await?;
            let (left_key2, right_key2) = match s.dir {
                crate::ast::RelationshipDirection::Outgoing
                | crate::ast::RelationshipDirection::Undirected => (
                    format!("{}__{}", s.rel_alias, rel_map.target_id_field),
                    format!(
                        "{}__{}",
                        s.end_alias,
                        self.node_maps.get(&s.end_alias).unwrap().id_field
                    ),
                ),
                crate::ast::RelationshipDirection::Incoming => (
                    format!("{}__{}", s.rel_alias, rel_map.source_id_field),
                    format!(
                        "{}__{}",
                        s.end_alias,
                        self.node_maps.get(&s.end_alias).unwrap().id_field
                    ),
                ),
            };
            df = df
                .join(
                    end_df,
                    JoinType::Inner,
                    &[left_key2.as_str()],
                    &[right_key2.as_str()],
                    None,
                )
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (rel->node): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            current_node_alias = &s.end_alias;
        }

        Ok(df)
    }

    fn resolve_var_alias<'b>(&'b self, var: &str) -> Option<&'b str> {
        if Some(var) == self.path.start_node.variable.as_deref() {
            return Some(self.start_alias.as_str());
        }
        for (i, seg) in self.path.segments.iter().enumerate() {
            if Some(var) == seg.relationship.variable.as_deref() {
                return Some(self.segs[i].rel_alias.as_str());
            }
            if Some(var) == seg.end_node.variable.as_deref() {
                return Some(self.segs[i].end_alias.as_str());
            }
        }
        None
    }

    fn apply_where(
        &self,
        mut df: datafusion::dataframe::DataFrame,
        ast: &crate::ast::CypherQuery,
    ) -> Result<datafusion::dataframe::DataFrame> {
        if let Some(where_clause) = &ast.where_clause {
            if let Some(expr) =
                to_df_boolean_expr_with_vars(&where_clause.expression, &|var, prop| {
                    let alias = self.resolve_var_alias(var).unwrap_or(var);
                    format!("{}__{}", alias, prop)
                })
            {
                df = df.filter(expr).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply WHERE: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
        }
        Ok(df)
    }

    fn apply_return(
        &self,
        mut df: datafusion::dataframe::DataFrame,
        ast: &crate::ast::CypherQuery,
    ) -> Result<datafusion::dataframe::DataFrame> {
        use datafusion::logical_expr::Expr;
        let mut proj: Vec<Expr> = Vec::new();
        for item in &ast.return_clause.items {
            if let crate::ast::ValueExpression::Property(prop) = &item.expression {
                let alias = self
                    .resolve_var_alias(&prop.variable)
                    .unwrap_or(&prop.variable);
                let mut e = datafusion::logical_expr::col(format!("{}__{}", alias, prop.property));
                if let Some(a) = &item.alias {
                    e = e.alias(a);
                }
                proj.push(e);
            }
        }
        if !proj.is_empty() {
            df = df.select(proj).map_err(|e| GraphError::PlanError {
                message: format!("Failed to project: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }
        if ast.return_clause.distinct {
            df = df.distinct().map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply DISTINCT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }
        if let Some(limit) = ast.limit {
            df = df
                .limit(0, Some(limit as usize))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply LIMIT: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }
        Ok(df)
    }
}

impl CypherQuery {
    /// Create a new Cypher query from a query string
    pub fn new(query: &str) -> Result<Self> {
        let ast = parse_cypher_query(query)?;

        Ok(Self {
            query_text: query.to_string(),
            ast,
            config: None,
            parameters: HashMap::new(),
        })
    }

    /// Set the graph configuration for this query
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a parameter to the query
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Add multiple parameters to the query
    pub fn with_parameters(mut self, params: HashMap<String, serde_json::Value>) -> Self {
        self.parameters.extend(params);
        self
    }

    /// Get the original query text
    pub fn query_text(&self) -> &str {
        &self.query_text
    }

    /// Get the parsed AST
    pub fn ast(&self) -> &CypherAST {
        &self.ast
    }

    /// Get the graph configuration
    pub fn config(&self) -> Option<&GraphConfig> {
        self.config.as_ref()
    }

    /// Get query parameters
    pub fn parameters(&self) -> &HashMap<String, serde_json::Value> {
        &self.parameters
    }

    /// Execute this Cypher query against Lance datasets
    ///
    /// Note: This initial implementation supports a single-table projection/filter/limit
    /// workflow to enable basic end-to-end execution. Multi-table/path execution will be
    /// wired up via the DataFusion planner in a follow-up.
    pub async fn execute(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::compute::concat_batches;
        use datafusion::datasource::MemTable;
        use datafusion::prelude::*;
        use std::sync::Arc;

        // Require a config for now, even if we don't fully exploit it yet
        let _config = self.config.as_ref().ok_or_else(|| GraphError::PlanError {
            message: "Graph configuration is required for query execution".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        if datasets.is_empty() {
            return Err(GraphError::PlanError {
                message: "No input datasets provided".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Create DataFusion context and register all provided tables
        let ctx = SessionContext::new();
        for (name, batch) in &datasets {
            let table =
                MemTable::try_new(batch.schema(), vec![vec![batch.clone()]]).map_err(|e| {
                    GraphError::PlanError {
                        message: format!("Failed to create DataFusion table: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?;
            ctx.register_table(name, Arc::new(table))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to register table '{}': {}", name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        // Try to execute a path (1+ hops) if the query is a simple pattern
        if let Some(df) = self.try_execute_path_generic(&ctx).await? {
            let batches = df.collect().await.map_err(|e| GraphError::PlanError {
                message: format!("Failed to collect results: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
            if batches.is_empty() {
                let schema = datasets.values().next().unwrap().schema();
                return Ok(arrow_array::RecordBatch::new_empty(schema));
            }
            let merged = concat_batches(&batches[0].schema(), &batches).map_err(|e| {
                GraphError::PlanError {
                    message: format!("Failed to concatenate result batches: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                }
            })?;
            return Ok(merged);
        }

        // Fallback: single-table style query on the first provided table
        let (table_name, batch) = datasets.iter().next().unwrap();
        let schema = batch.schema();

        // Start a DataFrame from the registered table
        let mut df = ctx
            .table(table_name)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to create DataFrame for '{}': {}", table_name, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Apply WHERE if present (limited support: simple comparisons on a single column)
        if let Some(where_clause) = &self.ast.where_clause {
            if let Some(filter_expr) = to_df_boolean_expr_simple(&where_clause.expression) {
                df = df.filter(filter_expr).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
        }

        // Build projection from RETURN clause
        let proj_exprs: Vec<Expr> = self
            .ast
            .return_clause
            .items
            .iter()
            .map(|item| to_df_value_expr_simple(&item.expression))
            .collect();
        if !proj_exprs.is_empty() {
            df = df.select(proj_exprs).map_err(|e| GraphError::PlanError {
                message: format!("Failed to project: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // Apply DISTINCT
        if self.ast.return_clause.distinct {
            df = df.distinct().map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply DISTINCT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // Apply LIMIT if present
        if let Some(limit) = self.ast.limit {
            df = df
                .limit(0, Some(limit as usize))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply LIMIT: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        // Collect results and concat into a single RecordBatch
        let batches = df.collect().await.map_err(|e| GraphError::PlanError {
            message: format!("Failed to collect results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        if batches.is_empty() {
            // Return an empty batch with the source schema
            return Ok(arrow_array::RecordBatch::new_empty(schema));
        }

        let merged =
            concat_batches(&batches[0].schema(), &batches).map_err(|e| GraphError::PlanError {
                message: format!("Failed to concatenate result batches: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        Ok(merged)
    }

    /// Validate the query against the provided configuration
    pub fn validate(&self) -> Result<()> {
        // Check that all referenced labels exist in configuration
        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.validate_pattern(pattern)?;
            }
        }

        // Validate WHERE clause if present
        if let Some(where_clause) = &self.ast.where_clause {
            self.validate_boolean_expression(&where_clause.expression)?;
        }

        // Validate RETURN clause
        for item in &self.ast.return_clause.items {
            self.validate_value_expression(&item.expression)?;
        }

        Ok(())
    }

    /// Get all node labels referenced in this query
    pub fn referenced_node_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();

        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.collect_node_labels_from_pattern(pattern, &mut labels);
            }
        }

        labels.sort();
        labels.dedup();
        labels
    }

    /// Get all relationship types referenced in this query
    pub fn referenced_relationship_types(&self) -> Vec<String> {
        let mut types = Vec::new();

        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.collect_relationship_types_from_pattern(pattern, &mut types);
            }
        }

        types.sort();
        types.dedup();
        types
    }

    /// Get all variables used in this query
    pub fn variables(&self) -> Vec<String> {
        let mut variables = Vec::new();

        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.collect_variables_from_pattern(pattern, &mut variables);
            }
        }

        variables.sort();
        variables.dedup();
        variables
    }

    // Validation helper methods

    fn validate_pattern(&self, pattern: &crate::ast::GraphPattern) -> Result<()> {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                for label in &node.labels {
                    if let Some(config) = &self.config {
                        if config.get_node_mapping(label).is_none() {
                            return Err(GraphError::PlanError {
                                message: format!("No mapping found for node label '{}'", label),
                                location: snafu::Location::new(file!(), line!(), column!()),
                            });
                        }
                    }
                }
                Ok(())
            }
            crate::ast::GraphPattern::Path(path) => {
                self.validate_pattern(&crate::ast::GraphPattern::Node(path.start_node.clone()))?;
                for segment in &path.segments {
                    for rel_type in &segment.relationship.types {
                        if let Some(config) = &self.config {
                            if config.get_relationship_mapping(rel_type).is_none() {
                                return Err(GraphError::PlanError {
                                    message: format!(
                                        "No mapping found for relationship type '{}'",
                                        rel_type
                                    ),
                                    location: snafu::Location::new(file!(), line!(), column!()),
                                });
                            }
                        }
                    }
                    self.validate_pattern(&crate::ast::GraphPattern::Node(
                        segment.end_node.clone(),
                    ))?;
                }
                Ok(())
            }
        }
    }

    fn validate_boolean_expression(&self, _expr: &crate::ast::BooleanExpression) -> Result<()> {
        // TODO: Implement validation of boolean expressions
        Ok(())
    }

    fn validate_value_expression(&self, _expr: &crate::ast::ValueExpression) -> Result<()> {
        // TODO: Implement validation of value expressions
        Ok(())
    }

    // Collection helper methods

    fn collect_node_labels_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        labels: &mut Vec<String>,
    ) {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                labels.extend(node.labels.clone());
            }
            crate::ast::GraphPattern::Path(path) => {
                labels.extend(path.start_node.labels.clone());
                for segment in &path.segments {
                    labels.extend(segment.end_node.labels.clone());
                }
            }
        }
    }

    fn collect_relationship_types_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        types: &mut Vec<String>,
    ) {
        if let crate::ast::GraphPattern::Path(path) = pattern {
            for segment in &path.segments {
                types.extend(segment.relationship.types.clone());
            }
        }
    }

    fn collect_variables_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        variables: &mut Vec<String>,
    ) {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                if let Some(var) = &node.variable {
                    variables.push(var.clone());
                }
            }
            crate::ast::GraphPattern::Path(path) => {
                if let Some(var) = &path.start_node.variable {
                    variables.push(var.clone());
                }
                for segment in &path.segments {
                    if let Some(var) = &segment.relationship.variable {
                        variables.push(var.clone());
                    }
                    if let Some(var) = &segment.end_node.variable {
                        variables.push(var.clone());
                    }
                }
            }
        }
    }
}

impl CypherQuery {
    // Generic path executor (N-hop) entrypoint.
    async fn try_execute_path_generic(
        &self,
        ctx: &datafusion::prelude::SessionContext,
    ) -> Result<Option<datafusion::dataframe::DataFrame>> {
        use crate::ast::GraphPattern;
        let [mc] = self.ast.match_clauses.as_slice() else {
            return Ok(None);
        };
        let match_clause = mc;
        let path = match match_clause.patterns.as_slice() {
            [GraphPattern::Path(p)] if !p.segments.is_empty() => p,
            _ => return Ok(None),
        };
        let cfg = self.config.as_ref().ok_or_else(|| GraphError::PlanError {
            message: "Graph configuration is required for execution".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        let exec = PathExecutor::new(ctx, cfg, path)?;
        let df = exec.build_chain().await?;
        let df = exec.apply_where(df, &self.ast)?;
        let df = exec.apply_return(df, &self.ast)?;
        Ok(Some(df))
    }

    // Attempt execution for a single-path pattern using joins.
    // Supports single-hop and two-hop expansions
    #[allow(dead_code)]
    async fn try_execute_single_hop_path(
        &self,
        ctx: &datafusion::prelude::SessionContext,
    ) -> Result<Option<datafusion::dataframe::DataFrame>> {
        use crate::ast::{GraphPattern, RelationshipDirection, ValueExpression};
        use datafusion::prelude::*;

        // Only handle a single MATCH with a single path and exactly one segment
        let [mc] = self.ast.match_clauses.as_slice() else {
            return Ok(None);
        };
        let match_clause = mc;
        let path = match match_clause.patterns.as_slice() {
            [GraphPattern::Path(p)] if (p.segments.len() == 1 || p.segments.len() == 2) => p,
            _ => return Ok(None),
        };
        let seg = &path.segments[0];
        let rel_type = match seg.relationship.types.first() {
            Some(t) => t.as_str(),
            None => return Ok(None),
        };
        let start_label = match path.start_node.labels.first() {
            Some(l) => l.as_str(),
            None => return Ok(None),
        };
        let end_label = match seg.end_node.labels.first() {
            Some(l) => l.as_str(),
            None => return Ok(None),
        };

        let start_alias = path.start_node.variable.as_deref().unwrap_or(start_label);
        let rel_alias = seg.relationship.variable.as_deref().unwrap_or(rel_type);
        let end_alias = seg.end_node.variable.as_deref().unwrap_or(end_label);

        // Validate mappings
        let cfg = self.config.as_ref().ok_or_else(|| GraphError::PlanError {
            message: "Graph configuration is required for execution".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        let start_map = cfg
            .get_node_mapping(start_label)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("No node mapping for '{}'", start_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let end_map = cfg
            .get_node_mapping(end_label)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("No node mapping for '{}'", end_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let rel_map =
            cfg.get_relationship_mapping(rel_type)
                .ok_or_else(|| GraphError::PlanError {
                    message: format!("No relationship mapping for '{}'", rel_type),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        // Read tables and alias
        let mut left = ctx
            .table(start_label)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", start_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        // Alias and flatten columns to '<alias>__<col>' to avoid ambiguity
        let left_schema = left.schema();
        let left_proj: Vec<datafusion::logical_expr::Expr> = left_schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!(
                    "{}__{}",
                    start_alias,
                    f.name()
                ))
            })
            .collect();
        left = left
            .alias(start_alias)?
            .select(left_proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", start_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        for (k, v) in &path.start_node.properties {
            let expr = to_df_literal(v);
            left = left
                .filter(datafusion::logical_expr::Expr::BinaryExpr(
                    datafusion::logical_expr::BinaryExpr {
                        left: Box::new(datafusion::logical_expr::col(format!(
                            "{}__{}",
                            start_alias, k
                        ))),
                        op: datafusion::logical_expr::Operator::Eq,
                        right: Box::new(expr),
                    },
                ))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        let mut rel_df = ctx
            .table(rel_type)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", rel_type, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let rel_schema = rel_df.schema();
        let rel_proj: Vec<datafusion::logical_expr::Expr> = rel_schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!(
                    "{}__{}",
                    rel_alias,
                    f.name()
                ))
            })
            .collect();
        rel_df = rel_df
            .alias(rel_alias)?
            .select(rel_proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", rel_type, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Join start -> relationship
        let (left_key, right_key) = match seg.relationship.direction {
            RelationshipDirection::Outgoing | RelationshipDirection::Undirected => (
                format!("{}__{}", start_alias, start_map.id_field),
                format!("{}__{}", rel_alias, rel_map.source_id_field),
            ),
            RelationshipDirection::Incoming => (
                format!("{}__{}", start_alias, start_map.id_field),
                format!("{}__{}", rel_alias, rel_map.target_id_field),
            ),
        };
        let mut joined = left
            .join(
                rel_df,
                JoinType::Inner,
                &[left_key.as_str()],
                &[right_key.as_str()],
                None,
            )
            .map_err(|e| GraphError::PlanError {
                message: format!("Join failed (node->rel): {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Join relationship -> end (or mid, for 2-hop)
        let mut right = ctx
            .table(end_label)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", end_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let right_schema = right.schema();
        let right_proj: Vec<datafusion::logical_expr::Expr> = right_schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!(
                    "{}__{}",
                    end_alias,
                    f.name()
                ))
            })
            .collect();
        right = right
            .alias(end_alias)?
            .select(right_proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", end_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let (left_key2, right_key2) = match seg.relationship.direction {
            RelationshipDirection::Outgoing | RelationshipDirection::Undirected => (
                format!("{}__{}", rel_alias, rel_map.target_id_field),
                format!("{}__{}", end_alias, end_map.id_field),
            ),
            RelationshipDirection::Incoming => (
                format!("{}__{}", rel_alias, rel_map.source_id_field),
                format!("{}__{}", end_alias, end_map.id_field),
            ),
        };
        joined = joined
            .join(
                right,
                JoinType::Inner,
                &[left_key2.as_str()],
                &[right_key2.as_str()],
                None,
            )
            .map_err(|e| GraphError::PlanError {
                message: format!("Join failed (rel->node): {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // If there is a second segment (two-hop), continue chaining joins
        if path.segments.len() == 2 {
            let seg2 = &path.segments[1];
            let rel_type2 = match seg2.relationship.types.first() {
                Some(t) => t.as_str(),
                None => return Ok(None),
            };
            let end2_label = match seg2.end_node.labels.first() {
                Some(l) => l.as_str(),
                None => return Ok(None),
            };

            let mid_alias = end_alias; // end of seg1 is the mid node
            let mut rel2_alias = seg2
                .relationship
                .variable
                .as_deref()
                .unwrap_or(rel_type2)
                .to_string();
            let mut end2_alias = seg2
                .end_node
                .variable
                .as_deref()
                .unwrap_or(end2_label)
                .to_string();
            // Ensure unique aliases to avoid duplicate-qualified column names
            use std::collections::HashSet;
            let mut used_aliases: HashSet<String> = [
                start_alias.to_string(),
                rel_alias.to_string(),
                end_alias.to_string(),
            ]
            .into_iter()
            .collect();
            let mut uniquify = |alias: &mut String| {
                if used_aliases.insert(alias.clone()) {
                    return;
                }
                let base = alias.clone();
                let mut i = 2usize;
                loop {
                    let cand = format!("{}_{}", base, i);
                    if used_aliases.insert(cand.clone()) {
                        *alias = cand;
                        break;
                    }
                    i += 1;
                }
            };
            uniquify(&mut rel2_alias);
            uniquify(&mut end2_alias);

            // Validate mappings
            let _mid_map = end_map; // end of seg1
            let rel2_map =
                cfg.get_relationship_mapping(rel_type2)
                    .ok_or_else(|| GraphError::PlanError {
                        message: format!("No relationship mapping for '{}'", rel_type2),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
            let end2_map =
                cfg.get_node_mapping(end2_label)
                    .ok_or_else(|| GraphError::PlanError {
                        message: format!("No node mapping for '{}'", end2_label),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;

            // Read rel2 and end2
            let mut rel2_df = ctx
                .table(rel_type2)
                .await
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to read table '{}': {}", rel_type2, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            rel2_df = rel2_df.alias(&rel2_alias)?;

            // Determine the mid-equivalent column from rel1 to avoid ambiguous mid.id on the left
            let mid_equiv_from_rel1 = match seg.relationship.direction {
                RelationshipDirection::Outgoing | RelationshipDirection::Undirected => {
                    rel_map.target_id_field.as_str()
                }
                RelationshipDirection::Incoming => rel_map.source_id_field.as_str(),
            };

            // Join mid -> rel2 using mid-equivalent column from rel1
            let (left_key3, right_key3) = match seg2.relationship.direction {
                RelationshipDirection::Outgoing | RelationshipDirection::Undirected => {
                    (mid_equiv_from_rel1, rel2_map.source_id_field.as_str())
                }
                RelationshipDirection::Incoming => {
                    (mid_equiv_from_rel1, rel2_map.target_id_field.as_str())
                }
            };
            joined = joined
                .join(rel2_df, JoinType::Inner, &[left_key3], &[right_key3], None)
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (mid->rel2): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            // Join rel2 -> end2
            let mut end2_df = ctx
                .table(end2_label)
                .await
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to read table '{}': {}", end2_label, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            end2_df = end2_df.alias(&end2_alias)?;
            // If left side already contains a column with the same name as the right join key,
            // rename the right key to avoid ambiguous unqualified field references.
            let mut right_join_key = end2_map.id_field.clone();
            {
                let left_schema = joined.schema();
                if left_schema
                    .fields()
                    .iter()
                    .any(|f| f.name() == &right_join_key)
                {
                    use datafusion::logical_expr::{col, Expr};
                    let new_key = format!("{}__rhs", right_join_key);
                    let schema = end2_df.schema();
                    let mut proj: Vec<Expr> = Vec::with_capacity(schema.fields().len());
                    for f in schema.fields() {
                        if f.name() == &right_join_key {
                            proj.push(col(f.name()).alias(&new_key));
                        } else {
                            proj.push(col(f.name()));
                        }
                    }
                    end2_df = end2_df.select(proj).map_err(|e| GraphError::PlanError {
                        message: format!("Failed to prepare right join side: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
                    right_join_key = new_key;
                }
            }

            let (left_key4, right_key4) = match seg2.relationship.direction {
                RelationshipDirection::Outgoing | RelationshipDirection::Undirected => {
                    (rel2_map.target_id_field.as_str(), right_join_key.as_str())
                }
                RelationshipDirection::Incoming => {
                    (rel2_map.source_id_field.as_str(), right_join_key.as_str())
                }
            };
            joined = joined
                .join(end2_df, JoinType::Inner, &[left_key4], &[right_key4], None)
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (rel2->end2): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            // Update end_alias to refer to final node for projection/WHERE qualification below
            let end_alias = end2_alias.as_str();

            // WHERE (qualified across all known aliases)
            if let Some(where_clause) = &self.ast.where_clause {
                if let Some(expr) =
                    to_df_boolean_expr_with_vars(&where_clause.expression, &|var, prop| {
                        let alias = if Some(var) == path.start_node.variable.as_deref() {
                            start_alias
                        } else if Some(var) == seg.relationship.variable.as_deref() {
                            rel_alias
                        } else if Some(var) == seg.end_node.variable.as_deref() {
                            mid_alias
                        } else if Some(var) == seg2.relationship.variable.as_deref() {
                            &rel2_alias
                        } else if Some(var) == seg2.end_node.variable.as_deref() {
                            end_alias
                        } else {
                            var
                        };
                        format!("{}.{}", alias, prop)
                    })
                {
                    joined = joined.filter(expr).map_err(|e| GraphError::PlanError {
                        message: format!("Failed to apply WHERE: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
                }
            }

            // Project RETURN across aliases
            let mut proj: Vec<datafusion::logical_expr::Expr> = Vec::new();
            for item in &self.ast.return_clause.items {
                if let ValueExpression::Property(prop) = &item.expression {
                    let col_name = if Some(prop.variable.as_str())
                        == path.start_node.variable.as_deref()
                    {
                        format!("{}.{}", start_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.relationship.variable.as_deref() {
                        format!("{}.{}", rel_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.end_node.variable.as_deref() {
                        format!("{}.{}", mid_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg2.relationship.variable.as_deref()
                    {
                        format!("{}.{}", rel2_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg2.end_node.variable.as_deref() {
                        format!("{}.{}", end_alias, prop.property)
                    } else {
                        format!("{}.{}", prop.variable, prop.property)
                    };
                    let mut e = datafusion::logical_expr::col(&col_name);
                    if let Some(a) = &item.alias {
                        e = e.alias(a);
                    }
                    proj.push(e);
                }
            }
            if !proj.is_empty() {
                joined = joined.select(proj).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to project: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }

            // DISTINCT and LIMIT
            if self.ast.return_clause.distinct {
                joined = joined.distinct().map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply DISTINCT: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
            if let Some(limit) = self.ast.limit {
                joined =
                    joined
                        .limit(0, Some(limit as usize))
                        .map_err(|e| GraphError::PlanError {
                            message: format!("Failed to apply LIMIT: {}", e),
                            location: snafu::Location::new(file!(), line!(), column!()),
                        })?;
            }

            return Ok(Some(joined));
        }

        // WHERE (qualified)
        if let Some(where_clause) = &self.ast.where_clause {
            if let Some(expr) =
                to_df_boolean_expr_with_vars(&where_clause.expression, &|var, prop| {
                    let alias = if Some(var) == path.start_node.variable.as_deref() {
                        start_alias
                    } else if Some(var) == seg.relationship.variable.as_deref() {
                        rel_alias
                    } else if Some(var) == seg.end_node.variable.as_deref() {
                        end_alias
                    } else {
                        var
                    };
                    format!("{}.{}", alias, prop)
                })
            {
                joined = joined.filter(expr).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply WHERE: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
        }

        // Project RETURN
        let mut proj: Vec<datafusion::logical_expr::Expr> = Vec::new();
        for item in &self.ast.return_clause.items {
            if let ValueExpression::Property(prop) = &item.expression {
                let col_name =
                    if Some(prop.variable.as_str()) == path.start_node.variable.as_deref() {
                        format!("{}.{}", start_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.relationship.variable.as_deref() {
                        format!("{}.{}", rel_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.end_node.variable.as_deref() {
                        format!("{}.{}", end_alias, prop.property)
                    } else {
                        format!("{}.{}", prop.variable, prop.property)
                    };
                let mut e = datafusion::logical_expr::col(&col_name);
                if let Some(a) = &item.alias {
                    e = e.alias(a);
                }
                proj.push(e);
            }
        }
        if !proj.is_empty() {
            joined = joined.select(proj).map_err(|e| GraphError::PlanError {
                message: format!("Failed to project: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // DISTINCT and LIMIT
        if self.ast.return_clause.distinct {
            joined = joined.distinct().map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply DISTINCT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }
        if let Some(limit) = self.ast.limit {
            joined = joined
                .limit(0, Some(limit as usize))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply LIMIT: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        Ok(Some(joined))
    }
}

fn to_df_boolean_expr_with_vars<F>(
    expr: &crate::ast::BooleanExpression,
    qualify: &F,
) -> Option<datafusion::logical_expr::Expr>
where
    F: Fn(&str, &str) -> String,
{
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use datafusion::logical_expr::{col, Expr, Operator};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            let (var, prop, lit_expr) = match (left, right) {
                (VE::Property(p), VE::Literal(val)) => {
                    (p.variable.as_str(), p.property.as_str(), to_df_literal(val))
                }
                (VE::Literal(val), VE::Property(p)) => {
                    (p.variable.as_str(), p.property.as_str(), to_df_literal(val))
                }
                _ => return None,
            };
            let qualified = qualify(var, prop);
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(col(&qualified)),
                op,
                right: Box::new(lit_expr),
            }))
        }
        BE::And(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_with_vars(l, qualify)?),
                op: Operator::And,
                right: Box::new(to_df_boolean_expr_with_vars(r, qualify)?),
            },
        )),
        BE::Or(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_with_vars(l, qualify)?),
                op: Operator::Or,
                right: Box::new(to_df_boolean_expr_with_vars(r, qualify)?),
            },
        )),
        BE::Not(inner) => Some(datafusion::logical_expr::Expr::Not(Box::new(
            to_df_boolean_expr_with_vars(inner, qualify)?,
        ))),
        _ => None,
    }
}

/// Builder for constructing Cypher queries programmatically
#[derive(Debug, Default)]
pub struct CypherQueryBuilder {
    match_clauses: Vec<crate::ast::MatchClause>,
    where_expression: Option<crate::ast::BooleanExpression>,
    return_items: Vec<crate::ast::ReturnItem>,
    order_by_items: Vec<crate::ast::OrderByItem>,
    limit: Option<u64>,
    distinct: bool,
    config: Option<GraphConfig>,
    parameters: HashMap<String, serde_json::Value>,
}

impl CypherQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a MATCH clause for a node pattern
    pub fn match_node(mut self, variable: &str, label: &str) -> Self {
        let node = crate::ast::NodePattern {
            variable: Some(variable.to_string()),
            labels: vec![label.to_string()],
            properties: HashMap::new(),
        };

        let match_clause = crate::ast::MatchClause {
            patterns: vec![crate::ast::GraphPattern::Node(node)],
        };

        self.match_clauses.push(match_clause);
        self
    }

    /// Set the graph configuration
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a RETURN item
    pub fn return_property(mut self, variable: &str, property: &str) -> Self {
        let prop_ref = crate::ast::PropertyRef::new(variable, property);
        let return_item = crate::ast::ReturnItem {
            expression: crate::ast::ValueExpression::Property(prop_ref),
            alias: None,
        };

        self.return_items.push(return_item);
        self
    }

    /// Set DISTINCT flag
    pub fn distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    /// Add a LIMIT clause
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Build the final CypherQuery
    pub fn build(self) -> Result<CypherQuery> {
        if self.match_clauses.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one MATCH clause".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        if self.return_items.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one RETURN item".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let ast = crate::ast::CypherQuery {
            match_clauses: self.match_clauses,
            where_clause: self
                .where_expression
                .map(|expr| crate::ast::WhereClause { expression: expr }),
            return_clause: crate::ast::ReturnClause {
                distinct: self.distinct,
                items: self.return_items,
            },
            order_by: if self.order_by_items.is_empty() {
                None
            } else {
                Some(crate::ast::OrderByClause {
                    items: self.order_by_items,
                })
            },
            limit: self.limit,
        };

        // Generate query text from AST (simplified)
        let query_text = "MATCH ... RETURN ...".to_string(); // TODO: Implement AST->text conversion

        let query = CypherQuery {
            query_text,
            ast,
            config: self.config,
            parameters: self.parameters,
        };

        query.validate()?;
        Ok(query)
    }
}

/// Minimal translator for simple boolean expressions into DataFusion Expr
fn to_df_boolean_expr_simple(
    expr: &crate::ast::BooleanExpression,
) -> Option<datafusion::logical_expr::Expr> {
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use datafusion::logical_expr::{col, Expr, Operator};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            // Only support property <op> literal
            let (col_name, lit_expr) = match (left, right) {
                (VE::Property(prop), VE::Literal(val)) => {
                    (prop.property.clone(), to_df_literal(val))
                }
                (VE::Literal(val), VE::Property(prop)) => {
                    (prop.property.clone(), to_df_literal(val))
                }
                _ => return None,
            };
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(col(col_name)),
                op,
                right: Box::new(lit_expr),
            }))
        }
        BE::And(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_simple(l)?),
                op: Operator::And,
                right: Box::new(to_df_boolean_expr_simple(r)?),
            },
        )),
        BE::Or(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_simple(l)?),
                op: Operator::Or,
                right: Box::new(to_df_boolean_expr_simple(r)?),
            },
        )),
        BE::Not(inner) => Some(datafusion::logical_expr::Expr::Not(Box::new(
            to_df_boolean_expr_simple(inner)?,
        ))),
        BE::Exists(prop) => Some(datafusion::logical_expr::Expr::IsNotNull(Box::new(
            datafusion::logical_expr::Expr::Column(datafusion::common::Column::from_name(
                prop.property.clone(),
            )),
        ))),
        _ => None,
    }
}

fn to_df_value_expr_simple(expr: &crate::ast::ValueExpression) -> datafusion::logical_expr::Expr {
    use crate::ast::ValueExpression as VE;
    use datafusion::logical_expr::{col, lit};
    match expr {
        VE::Property(prop) => col(&prop.property),
        VE::Variable(v) => col(v),
        VE::Literal(v) => to_df_literal(v),
        VE::Function { .. } | VE::Arithmetic { .. } => lit(0),
    }
}

fn to_df_literal(val: &crate::ast::PropertyValue) -> datafusion::logical_expr::Expr {
    use datafusion::logical_expr::lit;
    match val {
        crate::ast::PropertyValue::String(s) => lit(s.clone()),
        crate::ast::PropertyValue::Integer(i) => lit(*i),
        crate::ast::PropertyValue::Float(f) => lit(*f),
        crate::ast::PropertyValue::Boolean(b) => lit(*b),
        crate::ast::PropertyValue::Null => {
            datafusion::logical_expr::Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
        }
        crate::ast::PropertyValue::Parameter(_) => lit(0),
        crate::ast::PropertyValue::Property(prop) => datafusion::logical_expr::col(&prop.property),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GraphConfig;

    #[test]
    fn test_parse_simple_cypher_query() {
        let query = CypherQuery::new("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(query.query_text(), "MATCH (n:Person) RETURN n.name");
        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.variables(), vec!["n"]);
    }

    #[test]
    fn test_query_with_parameters() {
        let mut params = HashMap::new();
        params.insert("minAge".to_string(), serde_json::Value::Number(30.into()));

        let query = CypherQuery::new("MATCH (n:Person) WHERE n.age > $minAge RETURN n.name")
            .unwrap()
            .with_parameters(params);

        assert!(query.parameters().contains_key("minAge"));
    }

    #[test]
    fn test_query_builder() {
        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .build()
            .unwrap();

        let query = CypherQueryBuilder::new()
            .with_config(config)
            .match_node("n", "Person")
            .return_property("n", "name")
            .limit(10)
            .build()
            .unwrap();

        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.variables(), vec!["n"]);
    }

    #[test]
    fn test_relationship_query_parsing() {
        let query =
            CypherQuery::new("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name")
                .unwrap();
        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.referenced_relationship_types(), vec!["KNOWS"]);
        assert_eq!(query.variables(), vec!["a", "b", "r"]);
    }

    #[tokio::test]
    async fn test_execute_basic_projection_and_filter() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // Build a simple batch: name (Utf8), age (Int64)
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let q = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("people".to_string(), batch);

        let out = q.execute(data).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let names = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // Expect Bob (34) and David (42)
        let result: Vec<(String, i64)> = (0..out.num_rows())
            .map(|i| (names.value(i).to_string(), ages.value(i)))
            .collect();
        assert!(result.contains(&("Bob".to_string(), 34)));
        assert!(result.contains(&("David".to_string(), 42)));
    }

    #[tokio::test]
    async fn test_execute_single_hop_path_join_projection() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // People table: id, name, age
        let person_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let people = RecordBatch::try_new(
            person_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
                Arc::new(Int64Array::from(vec![28, 34, 29])),
            ],
        )
        .unwrap();

        // KNOWS relationship: src_person_id -> dst_person_id
        let rel_schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::Int64, false),
            Field::new("dst_person_id", DataType::Int64, false),
        ]));
        let knows = RecordBatch::try_new(
            rel_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])), // Alice -> Bob, Bob -> Carol
                Arc::new(Int64Array::from(vec![2, 3])),
            ],
        )
        .unwrap();

        // Config: Person(id) and KNOWS(src_person_id -> dst_person_id)
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();

        // Query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name
        let q = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        // Register tables using labels / rel types as names
        data.insert("Person".to_string(), people);
        data.insert("KNOWS".to_string(), knows);

        let out = q.execute(data).await.unwrap();
        // Expect two rows: Bob, Carol (the targets)
        let names = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let got: Vec<String> = (0..out.num_rows())
            .map(|i| names.value(i).to_string())
            .collect();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&"Bob".to_string()));
        assert!(got.contains(&"Carol".to_string()));
    }
}
