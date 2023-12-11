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

//! Extends logical expression.

use arrow_schema::DataType;

use datafusion::logical_expr::{
    expr::ScalarFunction, BinaryExpr, BuiltinScalarFunction, GetFieldAccess, GetIndexedField,
    Operator,
};
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use lance_arrow::DataTypeExt;
use lance_datafusion::expr::safe_coerce_scalar;

use crate::datatypes::Schema;
use crate::{Error, Result};
use snafu::{location, Location};
/// Resolve a Value
fn resolve_value(expr: &Expr, data_type: &DataType) -> Result<Expr> {
    match expr {
        Expr::Literal(scalar_value) => {
            Ok(Expr::Literal(safe_coerce_scalar(scalar_value, data_type)?.ok_or_else(|| Error::IO {
                message: format!("Received literal {expr} and could not convert to literal of type '{data_type:?}'"),
                location: location!(),
            })?))
        }
        _ => Err(Error::IO {
            message: format!("Expected a literal of type '{data_type:?}' but received: {expr}"),
            location: location!(),
        }),
    }
}

/// Given a Expr::Column or Expr::GetIndexedField, get the data type of referenced
/// field in the schema.
///
/// If the column is not found in the schema, return None. If the expression is
/// not a field reference, also returns None.
pub fn resolve_column_type(expr: &Expr, schema: &Schema) -> Option<DataType> {
    let mut field_path = Vec::new();
    let mut current_expr = expr;
    // We are looping from outer-most reference to inner-most.
    loop {
        match current_expr {
            Expr::Column(c) => {
                field_path.push(c.name.as_str());
                break;
            }
            Expr::GetIndexedField(GetIndexedField { expr, field }) => {
                if let GetFieldAccess::NamedStructField {
                    name: ScalarValue::Utf8(Some(name)),
                } = field
                {
                    field_path.push(name);
                } else {
                    // We don't support other kinds of access right now.
                    return None;
                }
                current_expr = expr.as_ref();
            }
            _ => return None,
        }
    }

    let mut path_iter = field_path.iter().rev();
    let mut field = schema.field(path_iter.next()?)?;
    for name in path_iter {
        if field.data_type().is_struct() {
            field = field.children.iter().find(|f| &f.name == name)?;
        } else {
            return None;
        }
    }
    Some(field.data_type())
}

/// Resolve logical expression `expr`.
///
/// Parameters
///
/// - *expr*: a datafusion logical expression
/// - *schema*: lance schema.
pub fn resolve_expr(expr: &Expr, schema: &Schema) -> Result<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if matches!(op, Operator::And | Operator::Or) {
                return Ok(Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(resolve_expr(left.as_ref(), schema)?),
                    op: *op,
                    right: Box::new(resolve_expr(right.as_ref(), schema)?),
                }));
            }
            match (left.as_ref(), right.as_ref()) {
                (Expr::Column(_) | Expr::GetIndexedField(_), Expr::Literal(_)) => {
                    if let Some(resolved_type) = resolve_column_type(left.as_ref(), schema) {
                        Ok(Expr::BinaryExpr(BinaryExpr {
                            left: left.clone(),
                            op: *op,
                            right: Box::new(resolve_value(right.as_ref(), &resolved_type)?),
                        }))
                    } else {
                        Ok(expr.clone())
                    }
                }
                (Expr::Literal(_), Expr::Column(_) | Expr::GetIndexedField(_)) => {
                    if let Some(resolved_type) = resolve_column_type(right.as_ref(), schema) {
                        Ok(Expr::BinaryExpr(BinaryExpr {
                            left: Box::new(resolve_value(right.as_ref(), &resolved_type)?),
                            op: *op,
                            right: left.clone(),
                        }))
                    } else {
                        Ok(expr.clone())
                    }
                }
                // For cases complex expressions (not just literals) on right hand side like x = 1 + 1 + -2*2
                (Expr::Column(_) | Expr::GetIndexedField(_), Expr::BinaryExpr(r)) => {
                    if let Some(resolved_type) = resolve_column_type(left.as_ref(), schema) {
                        Ok(Expr::BinaryExpr(BinaryExpr {
                            left: left.clone(),
                            op: *op,
                            right: Box::new(Expr::BinaryExpr(BinaryExpr {
                                left: coerce_expr(&r.left, &resolved_type).map(Box::new)?,
                                op: r.op,
                                right: coerce_expr(&r.right, &resolved_type).map(Box::new)?,
                            })),
                        }))
                    } else {
                        Ok(expr.clone())
                    }
                }
                _ => Ok(expr.clone()),
            }
        }
        Expr::InList(in_list) => {
            if matches!(
                in_list.expr.as_ref(),
                Expr::Column(_) | Expr::GetIndexedField(_)
            ) {
                if let Some(resolved_type) = resolve_column_type(in_list.expr.as_ref(), schema) {
                    let resolved_values = in_list
                        .list
                        .iter()
                        .map(|val| coerce_expr(val, &resolved_type))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Expr::in_list(
                        in_list.expr.as_ref().clone(),
                        resolved_values,
                        in_list.negated,
                    ))
                } else {
                    Ok(expr.clone())
                }
            } else {
                Ok(expr.clone())
            }
        }
        _ => {
            // Passthrough
            Ok(expr.clone())
        }
    }
}

/// Coerce expression of literals to column type.
///
/// Parameters
///
/// - *expr*: a datafusion logical expression
/// - *dtype*: a lance data type
pub fn coerce_expr(expr: &Expr, dtype: &DataType) -> Result<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => Ok(Expr::BinaryExpr(BinaryExpr {
            left: Box::new(coerce_expr(left, dtype)?),
            op: *op,
            right: Box::new(coerce_expr(right, dtype)?),
        })),
        Expr::Literal(l) => Ok(resolve_value(&Expr::Literal(l.clone()), dtype)?),
        _ => Ok(expr.clone()),
    }
}

/// Coerce logical expression for filters to boolean.
///
/// Parameters
///
/// - *expr*: a datafusion logical expression
pub fn coerce_filter_type_to_boolean(expr: Expr) -> Result<Expr> {
    match expr {
        // TODO: consider making this dispatch more generic, i.e. fun.output_type -> coerce
        // instead of hardcoding coerce method for each function
        Expr::ScalarFunction(ScalarFunction {
            fun: BuiltinScalarFunction::RegexpMatch,
            args: _,
        }) => Ok(Expr::IsNotNull(Box::new(expr))),

        _ => Ok(expr),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    use arrow_schema::{Field, Schema as ArrowSchema};
    use datafusion::scalar::ScalarValue;

    #[test]
    fn test_resolve_large_utf8() {
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::LargeUtf8, false)]);
        let expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column("a".to_string().into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(ScalarValue::Utf8(Some("a".to_string())))),
        });

        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();
        match resolved {
            Expr::BinaryExpr(be) => {
                assert_eq!(
                    be.right.as_ref(),
                    &Expr::Literal(ScalarValue::LargeUtf8(Some("a".to_string())))
                )
            }
            _ => unreachable!("Expected BinaryExpr"),
        };
    }

    #[test]
    fn test_resolve_binary_expr_on_right() {
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::Float64, false)]);
        let expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(Expr::Column("a".to_string().into())),
            op: Operator::Eq,
            right: Box::new(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Expr::Literal(ScalarValue::Int64(Some(2)))),
                op: Operator::Minus,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(-1)))),
            })),
        });
        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();

        match resolved {
            Expr::BinaryExpr(be) => match be.right.as_ref() {
                Expr::BinaryExpr(r_be) => {
                    assert_eq!(
                        r_be.left.as_ref(),
                        &Expr::Literal(ScalarValue::Float64(Some(2.0)))
                    );
                    assert_eq!(
                        r_be.right.as_ref(),
                        &Expr::Literal(ScalarValue::Float64(Some(-1.0)))
                    );
                }
                _ => panic!("Expected BinaryExpr"),
            },
            _ => panic!("Expected BinaryExpr"),
        }
    }

    #[test]
    fn test_resolve_in_expr() {
        // Type coersion should apply for `A IN (0)` or `A NOT IN (0)`
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::Float32, false)]);
        let expr = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float64(Some(0.0)))],
            false,
        );
        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();
        let expected = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float32(Some(0.0)))],
            false,
        );
        assert_eq!(resolved, expected);

        let expr = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float64(Some(0.0)))],
            true,
        );
        let resolved = resolve_expr(&expr, &Schema::try_from(&arrow_schema).unwrap()).unwrap();
        let expected = Expr::in_list(
            Expr::Column("a".to_string().into()),
            vec![Expr::Literal(ScalarValue::Float32(Some(0.0)))],
            true,
        );
        assert_eq!(resolved, expected);
    }

    #[test]
    fn test_resolve_column_type() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("int", DataType::Int32, true),
            Field::new(
                "st",
                DataType::Struct(
                    vec![
                        Field::new("str", DataType::Utf8, true),
                        Field::new(
                            "st",
                            DataType::Struct(
                                vec![Field::new("float", DataType::Float64, true)].into(),
                            ),
                            true,
                        ),
                    ]
                    .into(),
                ),
                true,
            ),
        ]));
        let schema = Schema::try_from(schema.as_ref()).unwrap();

        assert_eq!(
            resolve_column_type(&col("int"), &schema),
            Some(DataType::Int32)
        );
        assert_eq!(
            resolve_column_type(&col("st").field("str"), &schema),
            Some(DataType::Utf8)
        );
        assert_eq!(
            resolve_column_type(&col("st").field("st").field("float"), &schema),
            Some(DataType::Float64)
        );

        assert_eq!(resolve_column_type(&col("x"), &schema), None);
        assert_eq!(resolve_column_type(&col("str"), &schema), None);
        assert_eq!(resolve_column_type(&col("float"), &schema), None);
        assert_eq!(
            resolve_column_type(&col("st").field("str").eq(lit("x")), &schema),
            None
        );
    }
}
