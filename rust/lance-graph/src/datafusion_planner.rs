// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion-based physical planner for graph queries
//!
//! This module implements the proper graph-to-relational mapping:
//! - Nodes as Tables: Each node label becomes a table
//! - Relationships as Tables: Each relationship type becomes a linking table
//! - Cypher traversal becomes SQL joins
//!
//! Uses DataFusion's LogicalPlan and optimizer for world-class query optimization.

use crate::error::Result;
use crate::logical_plan::*;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{
    col, lit, BinaryExpr, EmptyRelation, Expr, LogicalPlan, LogicalPlanBuilder, Operator,
};
use std::sync::Arc;

/// Planner abstraction for graph-to-physical planning
pub trait GraphPhysicalPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan>;
}

/// DataFusion-based physical planner
/// TODO: Fix DataFusion API compatibility issues
pub struct DataFusionPlanner {
    #[allow(dead_code)]
    config: crate::config::GraphConfig,
    catalog: Option<Arc<dyn GraphSourceCatalog>>,
}

impl DataFusionPlanner {
    pub fn new(config: crate::config::GraphConfig) -> Self {
        Self {
            config,
            catalog: None,
        }
    }

    pub fn with_catalog(
        config: crate::config::GraphConfig,
        catalog: Arc<dyn GraphSourceCatalog>,
    ) -> Self {
        Self {
            config,
            catalog: Some(catalog),
        }
    }
}

impl GraphPhysicalPlanner for DataFusionPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        use std::collections::HashMap;
        let mut var_labels: HashMap<String, String> = HashMap::new();
        self.plan_operator_with_ctx(logical_plan, &mut var_labels)
    }
}

impl DataFusionPlanner {
    fn empty_plan(&self) -> LogicalPlanBuilder {
        let schema = Arc::new(DFSchema::empty());
        LogicalPlanBuilder::from(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema,
        }))
    }
    fn plan_operator_with_ctx(
        &self,
        op: &LogicalOperator,
        var_labels: &mut std::collections::HashMap<String, String>,
    ) -> Result<LogicalPlan> {
        match op {
            LogicalOperator::ScanByLabel {
                variable,
                label,
                properties,
                ..
            } => {
                // Track variable -> label mapping
                var_labels.insert(variable.clone(), label.clone());
                if let Some(cat) = &self.catalog {
                    if let Some(source) = cat.node_source(label) {
                        let mut builder = LogicalPlanBuilder::scan(label, source, None).unwrap();
                        for (k, v) in properties.iter() {
                            let lit_expr = self
                                .to_df_value_expr(&crate::ast::ValueExpression::Literal(v.clone()));
                            let filter_expr = Expr::BinaryExpr(BinaryExpr {
                                left: Box::new(col(k)),
                                op: Operator::Eq,
                                right: Box::new(lit_expr),
                            });
                            builder = builder.filter(filter_expr).unwrap();
                        }
                        return Ok(builder.build().unwrap());
                    }
                }
                Ok(self.empty_plan().build().unwrap())
            }
            LogicalOperator::Filter { input, predicate } => {
                if self.catalog.is_none() {
                    return self.plan_operator_with_ctx(input, var_labels);
                }
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                let expr = self.to_df_boolean_expr(predicate);
                Ok(LogicalPlanBuilder::from(input_plan)
                    .filter(expr)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Project { input, projections } => {
                if self.catalog.is_none() {
                    return self.plan_operator_with_ctx(input, var_labels);
                }
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                let exprs: Vec<Expr> = projections
                    .iter()
                    .map(|p| self.to_df_value_expr(&p.expression))
                    .collect();
                Ok(LogicalPlanBuilder::from(input_plan)
                    .project(exprs)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Distinct { input } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .distinct()
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Sort { input, .. } => {
                // Schema-less placeholder: skip sort for now
                self.plan_operator_with_ctx(input, var_labels)
            }
            LogicalOperator::Limit { input, count } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .limit(0, Some((*count) as usize))
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                ..
            }
            | LogicalOperator::VariableLengthExpand {
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                ..
            } => {
                let left_plan = self.plan_operator_with_ctx(input, var_labels)?;
                // TODO(two-hop+): Support chaining multiple hops in the physical plan.
                // For single hop we scan the relationship table and filter with an ON expression.
                // For two-hop (e.g., a-[:R1]->m-[:R2]->b), we should:
                //   1) Join a with R1 (as done here)
                //   2) Join the result with R2
                //   3) Join the result with the b node scan
                // Ensure we maintain/propagate variable->label mapping (var_labels) and
                // project/qualify columns to avoid ambiguity across joins.
                // For VariableLengthExpand with bounds, consider unrolling small fixed bounds
                // (e.g., *1..2) into a UNION ALL of 1-hop and 2-hop plans.
                // Attempt first hop: source node -> relationship table
                if let (Some(cat), Some(rel_type)) = (&self.catalog, relationship_types.first()) {
                    if let Some(rel_map) = self.config.relationship_mappings.get(rel_type) {
                        if let Some(src_label) = var_labels.get(source_variable) {
                            if let Some(node_map) = self.config.node_mappings.get(src_label) {
                                if let Some(rel_source) =
                                    cat.relationship_source(&rel_map.relationship_type)
                                {
                                    let rel_scan = LogicalPlanBuilder::scan(
                                        &rel_map.relationship_type,
                                        rel_source,
                                        None,
                                    )
                                    .unwrap()
                                    .build()
                                    .unwrap();
                                    let mut builder = LogicalPlanBuilder::from(left_plan)
                                        .cross_join(rel_scan)
                                        .unwrap();
                                    let (left_key, right_key) = match direction {
                                        crate::ast::RelationshipDirection::Outgoing => {
                                            (&node_map.id_field, &rel_map.source_id_field)
                                        }
                                        crate::ast::RelationshipDirection::Incoming => {
                                            (&node_map.id_field, &rel_map.target_id_field)
                                        }
                                        crate::ast::RelationshipDirection::Undirected => {
                                            (&node_map.id_field, &rel_map.source_id_field)
                                        }
                                    };
                                    let on_expr = Expr::BinaryExpr(BinaryExpr {
                                        left: Box::new(col(left_key)),
                                        op: Operator::Eq,
                                        right: Box::new(col(right_key)),
                                    });
                                    builder = builder.filter(on_expr).unwrap();
                                    // Track target variable placeholder label for downstream
                                    var_labels
                                        .entry(target_variable.clone())
                                        .or_insert_with(|| "Node".to_string());
                                    return Ok(builder.build().unwrap());
                                }
                            }
                        }
                    }
                }
                // Fallback: pass-through
                var_labels
                    .entry(target_variable.clone())
                    .or_insert_with(|| "Node".to_string());
                Ok(self.plan_operator_with_ctx(input, var_labels)?)
            }
            LogicalOperator::Join { left, .. } => {
                // Not yet implemented: explicit join. For now, use left branch
                self.plan_operator_with_ctx(left, var_labels)
            }
        }
    }

    fn to_df_boolean_expr(&self, expr: &crate::ast::BooleanExpression) -> Expr {
        use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO};
        match expr {
            BE::Comparison {
                left,
                operator,
                right,
            } => {
                let l = self.to_df_value_expr(left);
                let r = self.to_df_value_expr(right);
                let op = match operator {
                    CO::Equal => Operator::Eq,
                    CO::NotEqual => Operator::NotEq,
                    CO::LessThan => Operator::Lt,
                    CO::LessThanOrEqual => Operator::LtEq,
                    CO::GreaterThan => Operator::Gt,
                    CO::GreaterThanOrEqual => Operator::GtEq,
                };
                Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(l),
                    op,
                    right: Box::new(r),
                })
            }
            BE::And(l, r) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(self.to_df_boolean_expr(l)),
                op: Operator::And,
                right: Box::new(self.to_df_boolean_expr(r)),
            }),
            BE::Or(l, r) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(self.to_df_boolean_expr(l)),
                op: Operator::Or,
                right: Box::new(self.to_df_boolean_expr(r)),
            }),
            BE::Not(inner) => Expr::Not(Box::new(self.to_df_boolean_expr(inner))),
            BE::Exists(prop) => Expr::IsNotNull(Box::new(
                self.to_df_value_expr(&crate::ast::ValueExpression::Property(prop.clone())),
            )),
            _ => lit(true),
        }
    }

    fn to_df_value_expr(&self, expr: &crate::ast::ValueExpression) -> Expr {
        use crate::ast::{PropertyValue as PV, ValueExpression as VE};
        match expr {
            VE::Property(prop) => col(&prop.property),
            VE::Variable(v) => col(v),
            VE::Literal(PV::String(s)) => lit(s.clone()),
            VE::Literal(PV::Integer(i)) => lit(*i),
            VE::Literal(PV::Float(f)) => lit(*f),
            VE::Literal(PV::Boolean(b)) => lit(*b),
            VE::Literal(PV::Null) => {
                datafusion::logical_expr::Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
            }
            VE::Literal(PV::Parameter(_)) => lit(0),
            VE::Literal(PV::Property(prop)) => col(&prop.property),
            VE::Function { .. } | VE::Arithmetic { .. } => lit(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BooleanExpression, ComparisonOperator, PropertyRef, PropertyValue, ValueExpression,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem};
    use crate::source_catalog::{InMemoryCatalog, SimpleTableSource};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn person_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]))
    }

    fn make_catalog() -> Arc<dyn crate::source_catalog::GraphSourceCatalog> {
        let person_src = Arc::new(SimpleTableSource::new(person_schema()));
        let knows_schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::Int64, false),
            Field::new("dst_person_id", DataType::Int64, false),
        ]));
        let knows_src = Arc::new(SimpleTableSource::new(knows_schema));
        Arc::new(
            InMemoryCatalog::new()
                .with_node_source("Person", person_src)
                .with_relationship_source("KNOWS", knows_src),
        )
    }

    #[test]
    fn test_df_planner_scan_filter_project() {
        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let pred = BooleanExpression::Comparison {
            left: ValueExpression::Property(PropertyRef {
                variable: "n".to_string(),
                property: "age".to_string(),
            }),
            operator: ComparisonOperator::GreaterThan,
            right: ValueExpression::Literal(PropertyValue::Integer(30)),
        };

        let filter = LogicalOperator::Filter {
            input: Box::new(scan),
            predicate: pred,
        };

        let project = LogicalOperator::Project {
            input: Box::new(filter),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(s.contains("Projection"), "plan missing Projection: {}", s);
        assert!(s.contains("Filter"), "plan missing Filter: {}", s);
        assert!(s.contains("TableScan"), "plan missing TableScan: {}", s);
        assert!(
            s.contains("Person") || s.contains("person"),
            "plan missing table name: {}",
            s
        );
    }

    #[test]
    fn test_df_planner_property_pushdown_filter() {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: props,
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&scan).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "plan missing Filter: {}", s);
        assert!(s.contains("TableScan"), "plan missing TableScan: {}", s);
        assert!(
            s.contains("Person") || s.contains("person"),
            "plan missing table name: {}",
            s
        );
    }

    #[test]
    fn test_df_planner_expand_creates_join_filter() {
        // MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(expand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "b".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("CrossJoin") || s.contains("Join("),
            "plan missing CrossJoin/Join: {}",
            s
        );
        assert!(s.contains("Filter"), "plan missing Filter (ON): {}", s);
        assert!(
            s.contains("TableScan") && s.contains("person"),
            "plan missing person scan: {}",
            s
        );
        assert!(
            s.contains("TableScan") && (s.contains("KNOWS") || s.contains("knows")),
            "plan missing relationship scan: {}",
            s
        );
    }
}

/*
TODO: Re-implement DataFusion integration after fixing API compatibility issues.

The main issues to fix:
1. Column import path: Use datafusion::common::Column instead of datafusion::logical_expr::Column
2. TableSource trait: Need to use LogicalTableSource or create proper table sources
3. ScalarValue::Null needs Option<FieldMetadata> parameter
4. SortExpr type issues with DataFusion's Expr system

Reference implementation should be here when these issues are resolved.
*/
