// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Unified query processor implementing the full pipeline
//!
//! This module implements the complete query processing pipeline:
//! Parse → Semantic Analysis → Logical Plan → Physical Plan → Execution

use crate::ast::CypherQuery;
use crate::config::GraphConfig;
// use crate::datafusion_planner::DataFusionPlanner;
use crate::datafusion_planner::{DataFusionPlanner, GraphPhysicalPlanner};
use crate::error::{GraphError, Result};
use crate::logical_plan::{LogicalOperator, LogicalPlanner};
use crate::parser::parse_cypher_query;
use crate::semantic::{SemanticAnalyzer, SemanticResult};
use datafusion::logical_expr::LogicalPlan;

/// Complete query processing pipeline
pub struct QueryProcessor {
    config: GraphConfig,
}

/// Query execution plan with all intermediate representations
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Original query string
    pub query_text: String,
    /// Parsed AST
    pub ast: CypherQuery,
    /// Semantic analysis result
    pub semantic_result: SemanticResult,
    /// Logical plan
    pub logical_plan: LogicalOperator,
    /// DataFusion physical plan
    pub datafusion_plan: LogicalPlan,
}

impl QueryProcessor {
    pub fn new(config: GraphConfig) -> Self {
        Self { config }
    }

    /// Process a Cypher query through the complete pipeline
    pub fn process_query(&self, query_text: &str) -> Result<QueryPlan> {
        // Phase 1: Parse - Convert text to AST
        let ast = parse_cypher_query(query_text)?;

        // Phase 2: Semantic Analysis - Validate and enrich AST
        let mut semantic_analyzer = SemanticAnalyzer::new(self.config.clone());
        let semantic_result = semantic_analyzer.analyze(&ast)?;

        // Check for semantic errors
        if !semantic_result.errors.is_empty() {
            return Err(GraphError::PlanError {
                message: format!("Semantic errors: {}", semantic_result.errors.join(", ")),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Phase 3: Logical Planning - Convert AST to logical operators
        let mut logical_planner = LogicalPlanner::new();
        let logical_plan = logical_planner.plan(&ast)?;

        // Phase 4: Physical Planning - Convert logical plan to DataFusion plan
        let df_planner = DataFusionPlanner::new(self.config.clone());
        let datafusion_plan = df_planner.plan(&logical_plan)?;

        Ok(QueryPlan {
            query_text: query_text.to_string(),
            ast,
            semantic_result,
            logical_plan,
            datafusion_plan,
        })
    }

    /// Process and explain a query (for debugging)
    pub fn explain_query(&self, query_text: &str) -> Result<String> {
        let plan = self.process_query(query_text)?;

        let mut explanation = String::new();
        explanation.push_str("=== Query Processing Pipeline ===\n\n");
        explanation.push_str(&format!("Original Query:\n{}\n\n", plan.query_text));

        explanation.push_str("=== Phase 1: Parsing ===\n");
        explanation.push_str(&format!("AST: {:#?}\n\n", plan.ast));

        explanation.push_str("=== Phase 2: Semantic Analysis ===\n");
        explanation.push_str(&format!(
            "Variables: {:#?}\n",
            plan.semantic_result.variables
        ));
        if !plan.semantic_result.warnings.is_empty() {
            explanation.push_str(&format!("Warnings: {:?}\n", plan.semantic_result.warnings));
        }
        explanation.push('\n');

        explanation.push_str("=== Phase 3: Logical Planning ===\n");
        explanation.push_str(&format!("Logical Plan: {:#?}\n\n", plan.logical_plan));

        explanation.push_str("=== Phase 4: DataFusion Planning ===\n");
        explanation.push_str(&format!("DataFusion Plan: {:#?}\n\n", plan.datafusion_plan));

        // No legacy SQL generation layer; DataFusion plan is the physical output.

        Ok(explanation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> GraphConfig {
        GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap()
    }

    #[test]
    fn test_simple_query_pipeline() {
        let config = create_test_config();
        let processor = QueryProcessor::new(config);

        let query = "MATCH (n:Person) RETURN n.name";
        let plan = processor.process_query(query).unwrap();

        // Verify we have all phases
        // DataFusion plan is present (placeholder or concrete)
        let _ = plan.datafusion_plan;
        assert_eq!(plan.query_text, query);
        assert!(!plan.semantic_result.variables.is_empty());
    }

    #[test]
    fn test_query_explanation() {
        let config = create_test_config();
        let processor = QueryProcessor::new(config);

        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let explanation = processor.explain_query(query).unwrap();

        assert!(explanation.contains("Query Processing Pipeline"));
        assert!(explanation.contains("Phase 1: Parsing"));
        assert!(explanation.contains("Phase 2: Semantic Analysis"));
        assert!(explanation.contains("Phase 3: Logical Planning"));
        assert!(explanation.contains("Phase 4: DataFusion Planning"));
    }

    #[test]
    fn test_semantic_error_detection() {
        let config = create_test_config();
        let processor = QueryProcessor::new(config);

        // Query with undefined variable
        let query = "MATCH (n:Person) RETURN m.name"; // 'm' is not defined
        let result = processor.process_query(query);

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Undefined variable"));
    }

    #[test]
    fn test_new_pipeline_vs_old() {
        let config = create_test_config();
        let processor = QueryProcessor::new(config);

        // Test the new pipeline
        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let new_plan = processor.process_query(query).unwrap();

        // The new pipeline should produce a DataFusion plan
        let _ = new_plan.datafusion_plan;

        // Verify we have logical plan structure
        match &new_plan.logical_plan {
            LogicalOperator::Limit { input, .. }
            | LogicalOperator::Sort { input, .. }
            | LogicalOperator::Project { input, .. } => {
                // We should have nested structure
                assert!(matches!(**input, LogicalOperator::Filter { .. }));
            }
            _ => panic!("Expected nested logical plan structure"),
        }

        // Verify semantic analysis found variables
        assert!(new_plan.semantic_result.variables.contains_key("n"));
    }
}
