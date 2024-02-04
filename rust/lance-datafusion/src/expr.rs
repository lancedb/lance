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

//! Utilities for working with datafusion expressions

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{cast::AsArray, ArrayRef};
use arrow_schema::{DataType, Schema};
use datafusion::{
    datasource::empty::EmptyTable, execution::context::SessionContext, logical_expr::Expr,
};
use datafusion_common::{
    tree_node::{Transformed, TreeNode},
    Column, DataFusionError, ScalarValue, TableReference,
};
use prost::Message;
use snafu::{location, Location};

use lance_core::{Error, Result};
use substrait::proto::{
    expression_reference::ExprType,
    plan_rel::RelType,
    read_rel::{NamedTable, ReadType},
    rel, ExtendedExpression, Plan, PlanRel, ProjectRel, ReadRel, Rel, RelRoot,
};

// This is slightly tedious but when we convert expressions from SQL strings to logical
// datafusion expressions there is no type coercion that happens.  In other words "x = 7"
// will always yield "x = 7_u64" regardless of the type of the column "x".  As a result, we
// need to do that literal coercion ourselves.
pub fn safe_coerce_scalar(value: &ScalarValue, ty: &DataType) -> Option<ScalarValue> {
    match value {
        ScalarValue::Int8(val) => match ty {
            DataType::Int8 => Some(value.clone()),
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(i16::from(v)))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => Some(value.clone()),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => Some(value.clone()),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // These conversions are inherently lossy as the full range of i32 cannot
            // be represented in f32.  However, there is no f32::TryFrom(i32) and its not
            // clear users would want that anyways
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Int64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => Some(value.clone()),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt8(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(v.into()))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => Some(value.clone()),
            DataType::UInt16 => val.map(|v| ScalarValue::UInt16(Some(u16::from(v)))),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => Some(value.clone()),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => Some(value.clone()),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => {
                val.and_then(|v| i64::try_from(v).map(|v| ScalarValue::Int64(Some(v))).ok())
            }
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => Some(value.clone()),
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Float32(val) => match ty {
            DataType::Float32 => Some(value.clone()),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Float64(val) => match ty {
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Utf8(val) => match ty {
            DataType::Utf8 => Some(value.clone()),
            DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(val.clone())),
            _ => None,
        },
        ScalarValue::Boolean(_) => match ty {
            DataType::Boolean => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Null => Some(value.clone()),
        ScalarValue::List(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::LargeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::FixedSizeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        _ => None,
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
    let expr = match &envelope.referred_expr[0].expr_type {
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

    // Datafusion's substrait consumer only supports Plan (not ExtendedExpression) and so
    // we need to create a dummy plan with a single project node
    let plan = Plan {
        version: None,
        extensions: envelope.extensions.clone(),
        advanced_extensions: envelope.advanced_extensions.clone(),
        expected_type_urls: envelope.expected_type_urls.clone(),
        extension_uris: envelope.extension_uris.clone(),
        relations: vec![PlanRel {
            rel_type: Some(RelType::Root(RelRoot {
                input: Some(Rel {
                    rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                        common: None,
                        input: Some(Box::new(Rel {
                            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                                common: None,
                                base_schema: envelope.base_schema.clone(),
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
    // these to be unqualified references instead and so we need a quick trasnformation pass

    let expr = expr.transform(&|node| match node {
        Expr::Column(column) => {
            if let Some(relation) = column.relation {
                match relation {
                    TableReference::Bare { table } => {
                        if table == "dummy" {
                            Ok(Transformed::Yes(Expr::Column(Column {
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
                Ok(Transformed::No(Expr::Column(column)))
            }
        }
        _ => Ok(Transformed::No(node)),
    })?;
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::Field;
    use datafusion::logical_expr::{BinaryExpr, Operator};
    use datafusion_common::Column;
    use prost::Message;
    use substrait_expr::{
        builder::{schema::SchemaBuildersExt, BuilderParams, ExpressionsBuilder},
        functions::functions_comparison::FunctionsComparisonExt,
        helpers::{literals::literal, schema::SchemaInfo},
    };

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
