// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::datatypes::Schema;
use arrow_schema::DataType;
use datafusion::error::DataFusionError;
use datafusion::error::Result;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_physical_expr::PhysicalExpr;
use lance_core::datatypes::Field;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_table::rowids::RowIdIndex;

use super::RowAddrExpr;

/// Make a DataFusion projection node from a Lance Schema.
pub fn project(
    input: Arc<dyn ExecutionPlan>,
    row_id_index: &Option<Arc<RowIdIndex>>,
    projection: &Schema,
) -> Result<ProjectionExec> {
    // Use input schema and projection schema to create a list of physical expressions.
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(projection.fields.len());

    let input_schema = input.schema();

    for field in &projection.fields {
        // TODO: is there something we need to do special here to make sure _rowid
        // is not optimized out of earlier nodes?
        if field.name == "_rowaddr" {
            let rowid_pos = input
                .schema()
                .fields
                .iter()
                .position(|f| f.name() == ROW_ID)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "Projection: _rowaddr requested but _rowid missing in input schema"
                    ))
                })?;

            exprs.push((
                Arc::new(RowAddrExpr::new(row_id_index.clone(), rowid_pos)),
                ROW_ADDR.to_string(),
            ));
        } else {
            match field.data_type() {
                DataType::Struct(_) => {
                    let (expr, name) = struct_project_expr(field)?;
                    exprs.push((expr, name));
                }
                DataType::List(_) => {
                    todo!()
                }
                _ => {
                    // This is a simple field, so we can just use the field name.
                    let expr =
                        expressions::Column::new_with_schema(&field.name, input_schema.as_ref())?;
                    exprs.push((Arc::new(expr), field.name.clone()));
                }
            }
        }
    }

    ProjectionExec::try_new(exprs, input)
}

fn struct_project_expr(field: &Field) -> Result<(Arc<dyn PhysicalExpr>, String)> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: what if we already projected rowaddr?

    #[tokio::test]
    async fn test_project_node() {

        // Create a nested schema

        // Project that schema with nested fields selected

        // Make execution plan for it

        // Execute the plan on a batch
    }

    #[test]
    fn test_validates_input_schema() {
        // Errors if top-level field missing

        // Errors if _rowaddr requested but _rowid missing

        // Errors if nested field missing
    }

    #[tokio::test]
    async fn test_project_node_with_rowaddr() {
        // Create a batch with row id
    }
}
