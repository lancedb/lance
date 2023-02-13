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
use sqlparser::ast::{Expr as SQLExpr, Value};

/// Resolve a [sqlparser::ast::Expr] to [LogicalPlan]
///
/// We do the resolving by hand to support nested fields.
/// We could add the function back to [datafusion] in long term.
// fn resolve_filter(filter: &SQLExpr) -> Result<LogicalPlan> {
//     match filter {
//         SQLExpr::Value(v) => {
//             match v => {
//                 Value::Number(val, _) => LogicalPlan::Values(())
//             }
//             LogicalPlan::Values(),
//         },
//         _ => {
//             return Err(datafusion::error::DataFusionError::Execution(format!(
//                 "Lance does not support filter: {}",
//                 filter
//             )))
//         }
//     }

//     todo!()
// }

// ///
// pub fn create_physical_filter_expr(expr: &Expr, schema: &Schema) -> Result<Arc<dyn PhysicalExpr>> {
//     if !matches!(expr, Expr::Literal(_) | Expr::BinaryExpr(_)) {
//         return Err(datafusion::error::DataFusionError::Execution(format!(
//             "Lance only supports literal or binary expression, but got {}",
//             expr
//         )));
//     }

//     todo!()
// }


#[cfg(test)]
mod tests {

}