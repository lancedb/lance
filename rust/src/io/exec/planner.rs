// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion::{
    common::DFSchema,
    error::Result,
    execution::context::SessionState,
    logical_expr::LogicalPlan,
    physical_plan::{ExecutionPlan, PhysicalExpr, PhysicalPlanner},
    prelude::Expr,
};

/// Lance Physical Planner
pub struct LancePhysicalPlanner {}

#[async_trait]
impl PhysicalPlanner for LancePhysicalPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    /// Create a physical expression from a logical expression
    /// suitable for evaluation
    fn create_physical_expr(
        &self,
        expr: &Expr,
        _input_dfschema: &DFSchema,
        input_schema: &Schema,
        _session_state: &SessionState,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if !matches!(expr, Expr::Literal(_) | Expr::BinaryExpr(_)) {
            return Err(datafusion::error::DataFusionError::Contex, ()))
        }
    }
}
