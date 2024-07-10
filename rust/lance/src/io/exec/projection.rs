// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, OnceLock};

use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::ScalarUDFImpl;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};

fn get_field_func() -> Arc<dyn ScalarUDFImpl> {
    static GET_FIELD_FUNC: OnceLock<Arc<dyn ScalarUDFImpl>> = OnceLock::new();
    GET_FIELD_FUNC
        .get_or_init(|| Arc::new(GetFieldFunc::new()))
        .clone()
}
fn get_make_struct_func() -> Arc<dyn ScalarUDFImpl> {
    static MAKE_STRUCT_FUNC: OnceLock<Arc<dyn ScalarUDFImpl>> = OnceLock::new();
    MAKE_STRUCT_FUNC
        .get_or_init(|| Arc::new(NamedStructFunc::new()))
        .clone()
}

/// Make a DataFusion projection node from a Lance Schema.
pub fn project(input: Arc<dyn ExecutionPlan>, projection: &ArrowSchema) -> Result<ProjectionExec> {
    // Use input schema and projection schema to create a list of physical expressions.
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(projection.fields.len());

    let input_schema = input.schema();

    let selections = compute_projection(input_schema.fields(), &projection.fields)?;

    for selection in selections {
        let expr = selection_as_expr(&selection, input_schema.fields(), None);
        exprs.push((expr, todo!()));
    }

    ProjectionExec::try_new(exprs, input)
}

fn selection_as_expr(
    selection: &Selection,
    parent_fields: &Fields,
    parent_expr: Option<Arc<dyn PhysicalExpr>>,
) -> Arc<dyn PhysicalExpr> {
    match selection {
        Selection::FullField(field_name) => {
            let (field_index, field) = &parent_fields.find(field_name).unwrap();
            if let Some(expr) = parent_expr {
                // We are extracting a child field.
                sub_field(expr, field.as_ref(), field_name)
            } else {
                // This is a top-level field
                Arc::new(Column::new(field.name(), *field_index))
            }
        }
        Selection::StructProjection(i, sub_selections) => {
            todo!()
        }
    }
}

fn sub_field(
    parent_expr: Arc<dyn PhysicalExpr>,
    field: &Field,
    field_name: &str,
) -> Arc<dyn PhysicalExpr> {
    match field.data_type() {
        DataType::Struct(_) => {
            // TODO: global reference to this?
            let function = Arc::new(GetFieldFunc::new());
            Arc::new(ScalarFunctionExpr::new(
                function.name(),
                function.clone(),
                vec![
                    parent_expr,
                    Arc::new(Literal::new(ScalarValue::Int64(field_index as i64))),
                ],
                field.data_type().clone(),
                function.monotonicity(),
                function.signature().type_signature.supports_zero_argument(),
            ))
        }
        _ => Arc::new(Column::new(field.name(), field_index)),
    }
}

/// Represents selection of fields from a struct / schema.
pub enum Selection<'a> {
    /// Selects this fields and all subfields
    FullField(&'a str),
    /// For a struct, selections of subfields
    StructProjection(&'a str, Vec<Selection<'a>>),
}

/// Resolves the projection into a list of selections.
pub fn compute_projection<'a, 'b>(
    schema: &'b Fields,
    projection: &'a Fields,
) -> Result<Vec<Selection<'a>>> {
    let mut selections = Vec::with_capacity(projection.len());
    for projected_field in projection {
        match projected_field.data_type() {
            DataType::Struct(fields) => {
                let original_field = schema
                    .iter()
                    .find(|f| &f.name() == &projected_field.name())
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "compute_projection: projected field {} not found in schema {:?}",
                            projected_field.name(),
                            schema
                        ))
                    })?;
                let input_fields = if let DataType::Struct(fields) = original_field.data_type() {
                    fields
                } else {
                    return Err(DataFusionError::Internal(
                        "compute_projection: expected struct".to_string(),
                    )
                    .into());
                };

                let sub_selections = compute_projection(&input_fields, fields)?;
                if fields.len() == input_fields.len()
                    && sub_selections
                        .iter()
                        .all(|s| matches!(s, Selection::FullField(_)))
                {
                    selections.push(Selection::FullField(projected_field.name()));
                } else {
                    selections.push(Selection::StructProjection(
                        projected_field.name(),
                        sub_selections,
                    ));
                }
            }
            _ => {
                selections.push(Selection::FullField(projected_field.name()));
            }
        }
    }
    Ok(selections)
}

/// Basically has two paths:
///  * if we are selecting all of it, then `Column(name)`
///  * if we are selecting subset of it, then
/// ```
/// MakeStruct(
///     GetFieldAccess(Column(name), field1),
///     GetFieldAccess(Column(name), field2),
///     ...
/// )
/// ```
fn struct_project_expr(
    field: &Field,
    access_expr: Arc<dyn PhysicalExpr>,
) -> Result<(Arc<dyn PhysicalExpr>, String)> {
    debug_assert!(matches!(field.data_type(), DataType::Struct(_)));
    // For each sub field, get the field access
    // Then use create struct
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
