// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Native physical planner (placeholder)
//!
//! This planner is intended to compile logical graph plans into a physical
//! execution plan that leverages Lance's native scan and filter engine.
//!
//! For now, this is a placeholder implementation that conforms to the
//! `GraphPhysicalPlanner` trait and returns an empty DataFusion logical plan
//! until the native pipeline is wired up.

use crate::config::GraphConfig;
use crate::datafusion_planner::GraphPhysicalPlanner;
use crate::error::Result;
use crate::logical_plan::LogicalOperator;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{EmptyRelation, LogicalPlan};
use std::sync::Arc;

/// Placeholder Lance-native planner
pub struct LanceNativePlanner {
    #[allow(dead_code)]
    config: GraphConfig,
}

impl LanceNativePlanner {
    pub fn new(config: GraphConfig) -> Self {
        Self { config }
    }
}

impl GraphPhysicalPlanner for LanceNativePlanner {
    fn plan(&self, _logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        // Placeholder: return an empty relation. A future implementation will
        // produce a runnable pipeline using Lance's native execution engine.
        let schema = Arc::new(DFSchema::empty());
        Ok(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lance_native_planner_placeholder() {
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = LanceNativePlanner::new(cfg);
        // Minimal logical plan to feed into placeholder
        let lp = LogicalOperator::Distinct {
            input: Box::new(LogicalOperator::Limit {
                input: Box::new(LogicalOperator::Project {
                    input: Box::new(LogicalOperator::ScanByLabel {
                        variable: "n".to_string(),
                        label: "Person".to_string(),
                        properties: Default::default(),
                    }),
                    projections: vec![],
                }),
                count: 1,
            }),
        };
        let df_plan = planner.plan(&lp).unwrap();
        // Empty relation is acceptable as a placeholder
        match df_plan {
            LogicalPlan::EmptyRelation(_) => {}
            _ => panic!("expected empty relation placeholder"),
        }
    }
}
