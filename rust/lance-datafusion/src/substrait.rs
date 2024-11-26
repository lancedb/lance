// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::Schema;
use datafusion::{
    datasource::empty::EmptyTable, execution::context::SessionContext, logical_expr::Expr,
};
use datafusion_common::{
    tree_node::{Transformed, TreeNode},
    Column, DataFusionError, TableReference,
};
use datafusion_substrait::substrait::proto::{
    expression::field_reference::{ReferenceType, RootType},
    expression::reference_segment,
    expression::RexType,
    expression_reference::ExprType,
    extensions::{simple_extension_declaration::MappingType, SimpleExtensionDeclaration},
    function_argument::ArgType,
    plan_rel::RelType,
    r#type::{Kind, Struct},
    read_rel::{NamedTable, ReadType},
    rel, Expression, ExtendedExpression, NamedStruct, Plan, PlanRel, ProjectRel, ReadRel, Rel,
    RelRoot,
};
use lance_core::{Error, Result};
use prost::Message;
use snafu::{location, Location};
use std::collections::HashMap;
use std::sync::Arc;

/// Convert a DF Expr into a Substrait ExtendedExpressions message
pub fn encode_substrait(expr: Expr, schema: Arc<Schema>) -> Result<Vec<u8>> {
    use datafusion::logical_expr::{builder::LogicalTableSource, logical_plan, LogicalPlan};
    use datafusion_substrait::substrait::proto::{plan_rel, ExpressionReference, NamedStruct};

    let table_source = Arc::new(LogicalTableSource::new(schema.clone()));

    // DF doesn't handled ExtendedExpressions and so we need to create
    // a dummy plan with a single filter node
    let plan = LogicalPlan::Filter(logical_plan::Filter::try_new(
        expr,
        Arc::new(LogicalPlan::TableScan(logical_plan::TableScan::try_new(
            "dummy",
            table_source,
            None,
            vec![],
            None,
        )?)),
    )?);

    let session_context = SessionContext::new();

    let substrait_plan =
        datafusion_substrait::logical_plan::producer::to_substrait_plan(&plan, &session_context)?;

    if let Some(plan_rel::RelType::Root(root)) = &substrait_plan.relations[0].rel_type {
        if let Some(rel::RelType::Filter(filt)) = &root.input.as_ref().unwrap().rel_type {
            let expr = filt.condition.as_ref().unwrap().clone();
            let schema = NamedStruct {
                names: schema.fields().iter().map(|f| f.name().clone()).collect(),
                r#struct: None,
            };
            let envelope = ExtendedExpression {
                advanced_extensions: substrait_plan.advanced_extensions.clone(),
                base_schema: Some(schema),
                expected_type_urls: substrait_plan.expected_type_urls.clone(),
                extension_uris: substrait_plan.extension_uris.clone(),
                extensions: substrait_plan.extensions.clone(),
                referred_expr: vec![ExpressionReference {
                    output_names: vec![],
                    expr_type: Some(ExprType::Expression(*expr)),
                }],
                version: substrait_plan.version.clone(),
            };
            Ok(envelope.encode_to_vec())
        } else {
            unreachable!()
        }
    } else {
        unreachable!()
    }
}

fn remove_extension_types(
    substrait_schema: &NamedStruct,
    arrow_schema: Arc<Schema>,
) -> Result<(NamedStruct, Arc<Schema>, HashMap<usize, usize>)> {
    let fields = substrait_schema.r#struct.as_ref().unwrap();
    if fields.types.len() != arrow_schema.fields.len() {
        return Err(Error::InvalidInput {
            source: "the number of fields in the provided substrait schema did not match the number of fields in the input schema.".into(),
            location: location!(),
        });
    }
    let mut kept_substrait_fields = Vec::with_capacity(fields.types.len());
    let mut kept_arrow_fields = Vec::with_capacity(arrow_schema.fields.len());
    let mut index_mapping = HashMap::with_capacity(arrow_schema.fields.len());
    let mut field_counter = 0;
    for (field_index, (substrait_field, arrow_field)) in fields
        .types
        .iter()
        .zip(arrow_schema.fields.iter())
        .enumerate()
    {
        if !matches!(
            substrait_field.kind.as_ref().unwrap(),
            Kind::UserDefined(_) | Kind::UserDefinedTypeReference(_)
        ) {
            kept_substrait_fields.push(substrait_field.clone());
            kept_arrow_fields.push(arrow_field.clone());
            index_mapping.insert(field_index, field_counter);
            field_counter += 1;
        }
    }
    let new_arrow_schema = Arc::new(Schema::new(kept_arrow_fields));
    let new_substrait_schema = NamedStruct {
        names: vec![],
        r#struct: Some(Struct {
            nullability: fields.nullability,
            type_variation_reference: fields.type_variation_reference,
            types: kept_substrait_fields,
        }),
    };
    Ok((new_substrait_schema, new_arrow_schema, index_mapping))
}

fn remove_type_extensions(
    declarations: &[SimpleExtensionDeclaration],
) -> Vec<SimpleExtensionDeclaration> {
    declarations
        .iter()
        .filter(|d| matches!(d.mapping_type, Some(MappingType::ExtensionFunction(_))))
        .cloned()
        .collect()
}

fn remap_expr_references(expr: &mut Expression, mapping: &HashMap<usize, usize>) -> Result<()> {
    match expr.rex_type.as_mut().unwrap() {
        // Simple, no field references possible
        RexType::Literal(_) | RexType::Nested(_) | RexType::Enum(_) => Ok(()),
        // Complex operators not supported in filters
        RexType::WindowFunction(_) | RexType::Subquery(_) => Err(Error::invalid_input(
            "Window functions or subqueries not allowed in filter expression",
            location!(),
        )),
        // Pass through operators, nested children may have field references
        RexType::ScalarFunction(ref mut func) => {
            #[allow(deprecated)]
            for arg in &mut func.args {
                remap_expr_references(arg, mapping)?;
            }
            for arg in &mut func.arguments {
                match arg.arg_type.as_mut().unwrap() {
                    ArgType::Value(expr) => remap_expr_references(expr, mapping)?,
                    ArgType::Enum(_) | ArgType::Type(_) => {}
                }
            }
            Ok(())
        }
        RexType::IfThen(ref mut ifthen) => {
            for clause in ifthen.ifs.iter_mut() {
                remap_expr_references(clause.r#if.as_mut().unwrap(), mapping)?;
                remap_expr_references(clause.then.as_mut().unwrap(), mapping)?;
            }
            remap_expr_references(ifthen.r#else.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::SwitchExpression(ref mut switch) => {
            for clause in switch.ifs.iter_mut() {
                remap_expr_references(clause.then.as_mut().unwrap(), mapping)?;
            }
            remap_expr_references(switch.r#else.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::SingularOrList(ref mut orlist) => {
            for opt in orlist.options.iter_mut() {
                remap_expr_references(opt, mapping)?;
            }
            remap_expr_references(orlist.value.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::MultiOrList(ref mut orlist) => {
            for opt in orlist.options.iter_mut() {
                for field in opt.fields.iter_mut() {
                    remap_expr_references(field, mapping)?;
                }
            }
            for val in orlist.value.iter_mut() {
                remap_expr_references(val, mapping)?;
            }
            Ok(())
        }
        RexType::Cast(ref mut cast) => {
            remap_expr_references(cast.input.as_mut().unwrap(), mapping)?;
            Ok(())
        }
        RexType::Selection(ref mut sel) => {
            // Finally, the selection, which might actually have field references
            let root_type = sel.root_type.as_mut().unwrap();
            // These types of references do not reference input fields so no remap needed
            if matches!(
                root_type,
                RootType::Expression(_) | RootType::OuterReference(_)
            ) {
                return Ok(());
            }
            match sel.reference_type.as_mut().unwrap() {
                ReferenceType::DirectReference(direct) => {
                    match direct.reference_type.as_mut().unwrap() {
                        reference_segment::ReferenceType::ListElement(_)
                        | reference_segment::ReferenceType::MapKey(_) => Err(Error::invalid_input(
                            "map/list nested references not supported in pushdown filters",
                            location!(),
                        )),
                        reference_segment::ReferenceType::StructField(field) => {
                            if field.child.is_some() {
                                Err(Error::invalid_input(
                                    "nested references in pushdown filters not yet supported",
                                    location!(),
                                ))
                            } else {
                                if let Some(new_index) = mapping.get(&(field.field as usize)) {
                                    field.field = *new_index as i32;
                                } else {
                                    return Err(Error::invalid_input("pushdown filter referenced a field that is not yet supported by Substrait conversion", location!()));
                                }
                                Ok(())
                            }
                        }
                    }
                }
                ReferenceType::MaskedReference(_) => Err(Error::invalid_input(
                    "masked references not yet supported in filter expressions",
                    location!(),
                )),
            }
        }
    }
}

/// Convert a Substrait ExtendedExpressions message into a DF Expr
///
/// The ExtendedExpressions message must contain a single scalar expression
pub async fn parse_substrait(expr: &[u8], input_schema: Arc<Schema>) -> Result<Expr> {
    let envelope = ExtendedExpression::decode(expr)?;
    if envelope.referred_expr.is_empty() {
        return Err(Error::InvalidInput {
            source: "the provided substrait expression is empty (contains no expressions)".into(),
            location: location!(),
        });
    }
    if envelope.referred_expr.len() > 1 {
        return Err(Error::InvalidInput {
            source: format!(
                "the provided substrait expression had {} expressions when only 1 was expected",
                envelope.referred_expr.len()
            )
            .into(),
            location: location!(),
        });
    }
    let mut expr = match &envelope.referred_expr[0].expr_type {
        None => Err(Error::InvalidInput {
            source: "the provided substrait had an expression but was missing an expr_type".into(),
            location: location!(),
        }),
        Some(ExprType::Expression(expr)) => Ok(expr.clone()),
        _ => Err(Error::InvalidInput {
            source: "the provided substrait was not a scalar expression".into(),
            location: location!(),
        }),
    }?;

    let (substrait_schema, input_schema) =
        if envelope.base_schema.as_ref().unwrap().r#struct.is_some() {
            let (substrait_schema, input_schema, index_mapping) = remove_extension_types(
                envelope.base_schema.as_ref().unwrap(),
                input_schema.clone(),
            )?;

            if substrait_schema.r#struct.as_ref().unwrap().types.len()
                != envelope
                    .base_schema
                    .as_ref()
                    .unwrap()
                    .r#struct
                    .as_ref()
                    .unwrap()
                    .types
                    .len()
            {
                remap_expr_references(&mut expr, &index_mapping)?;
            }

            (substrait_schema, input_schema)
        } else {
            (envelope.base_schema.as_ref().unwrap().clone(), input_schema)
        };

    // Datafusion's substrait consumer only supports Plan (not ExtendedExpression) and so
    // we need to create a dummy plan with a single project node
    let plan = Plan {
        version: None,
        extensions: remove_type_extensions(&envelope.extensions),
        advanced_extensions: envelope.advanced_extensions.clone(),
        expected_type_urls: vec![],
        extension_uris: vec![],
        relations: vec![PlanRel {
            rel_type: Some(RelType::Root(RelRoot {
                input: Some(Rel {
                    rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                        common: None,
                        input: Some(Box::new(Rel {
                            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                                common: None,
                                base_schema: Some(substrait_schema),
                                filter: None,
                                best_effort_filter: None,
                                projection: None,
                                advanced_extension: None,
                                read_type: Some(ReadType::NamedTable(NamedTable {
                                    names: vec!["dummy".to_string()],
                                    advanced_extension: None,
                                })),
                            }))),
                        })),
                        expressions: vec![expr],
                        advanced_extension: None,
                    }))),
                }),
                // Not technically accurate but pretty sure DF ignores this
                names: vec![],
            })),
        }],
    };

    let session_context = SessionContext::new();
    let dummy_table = Arc::new(EmptyTable::new(input_schema));
    session_context.register_table(
        TableReference::Bare {
            table: "dummy".into(),
        },
        dummy_table,
    )?;
    let df_plan =
        datafusion_substrait::logical_plan::consumer::from_substrait_plan(&session_context, &plan)
            .await?;

    let expr = df_plan.expressions().pop().unwrap();

    // When DF parses the above plan it turns column references into qualified references
    // into `dummy` (e.g. we get `WHERE dummy.x < 0` instead of `WHERE x < 0`)  We want
    // these to be unqualified references instead and so we need a quick transformation pass

    let expr = expr.transform(&|node| match node {
        Expr::Column(column) => {
            if let Some(relation) = column.relation {
                match relation {
                    TableReference::Bare { table } => {
                        if table.as_ref() == "dummy" {
                            Ok(Transformed::yes(Expr::Column(Column {
                                relation: None,
                                name: column.name,
                            })))
                        } else {
                            // This should not be possible
                            Err(DataFusionError::Substrait(format!(
                                "Unexpected reference to table {} found when parsing filter",
                                table
                            )))
                        }
                    }
                            // This should not be possible
                            _ => Err(DataFusionError::Substrait("Unexpected partially or fully qualified table reference encountered when parsing filter".into()))
                }
            } else {
                Ok(Transformed::no(Expr::Column(column)))
            }
        }
        _ => Ok(Transformed::no(node)),
    })?;
    Ok(expr.data)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        logical_expr::{BinaryExpr, Operator},
        prelude::Expr,
    };
    use datafusion_common::{Column, ScalarValue};
    use prost::Message;
    use substrait_expr::functions::functions_comparison::FunctionsComparisonExt;
    use substrait_expr::{
        builder::{schema::SchemaBuildersExt, BuilderParams, ExpressionsBuilder},
        helpers::{literals::literal, schema::SchemaInfo},
    };

    use crate::substrait::parse_substrait;

    #[tokio::test]
    async fn test_substrait_conversion() {
        let schema = SchemaInfo::new_full()
            .field("x", substrait_expr::helpers::types::i32(true))
            .build();
        let expr_builder = ExpressionsBuilder::new(schema, BuilderParams::default());
        expr_builder
            .add_expression(
                "filter_mask",
                expr_builder
                    .functions()
                    .lt(
                        expr_builder.fields().resolve_by_name("x").unwrap(),
                        literal(0_i32),
                    )
                    .build()
                    .unwrap(),
            )
            .unwrap();
        let expr = expr_builder.build();
        let expr_bytes = expr.encode_to_vec();

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));

        let df_expr = parse_substrait(expr_bytes.as_slice(), schema)
            .await
            .unwrap();

        let expected = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column(Column {
                relation: None,
                name: "x".to_string(),
            })),
            op: Operator::Lt,
            right: Box::new(Expr::Literal(ScalarValue::Int32(Some(0)))),
        });
        assert_eq!(df_expr, expected);
    }
}
