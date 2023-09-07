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
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{BuiltinScalarFunction, Operator};
use datafusion::scalar::ScalarValue;
use datafusion::{logical_expr::BinaryExpr, prelude::*};

use crate::datatypes::Schema;
use crate::{Error, Result};

/// Resolve a Value
fn resolve_value(expr: &Expr, data_type: &DataType) -> Result<Expr> {
    match expr {
        Expr::Literal(ScalarValue::Int64(v)) => match data_type {
            DataType::Int8 => Ok(Expr::Literal(ScalarValue::Int8(v.map(|v| v as i8)))),
            DataType::Int16 => Ok(Expr::Literal(ScalarValue::Int16(v.map(|v| v as i16)))),
            DataType::Int32 => Ok(Expr::Literal(ScalarValue::Int32(v.map(|v| v as i32)))),
            DataType::Int64 => Ok(Expr::Literal(ScalarValue::Int64(*v))),
            DataType::UInt8 => Ok(Expr::Literal(ScalarValue::UInt8(v.map(|v| v as u8)))),
            DataType::UInt16 => Ok(Expr::Literal(ScalarValue::UInt16(v.map(|v| v as u16)))),
            DataType::UInt32 => Ok(Expr::Literal(ScalarValue::UInt32(v.map(|v| v as u32)))),
            DataType::UInt64 => Ok(Expr::Literal(ScalarValue::UInt64(v.map(|v| v as u64)))),
            DataType::Float32 => Ok(Expr::Literal(ScalarValue::Float32(v.map(|v| v as f32)))),
            DataType::Float64 => Ok(Expr::Literal(ScalarValue::Float64(v.map(|v| v as f64)))),
            _ => Err(Error::IO {
                message: format!("DataType '{data_type:?}' does not match to the value: {expr}"),
            }),
        },
        Expr::Literal(ScalarValue::Float64(v)) => match data_type {
            DataType::Float32 => Ok(Expr::Literal(ScalarValue::Float32(v.map(|v| v as f32)))),
            DataType::Float64 => Ok(Expr::Literal(ScalarValue::Float64(*v))),
            _ => Err(Error::IO {
                message: format!("DataType '{data_type:?}' does not match to the value: {expr}"),
            }),
        },
        Expr::Literal(ScalarValue::Utf8(v)) => match data_type {
            DataType::Utf8 => Ok(expr.clone()),
            DataType::LargeUtf8 => Ok(Expr::Literal(ScalarValue::LargeUtf8(v.clone()))),
            _ => Err(Error::IO {
                message: format!("DataType '{data_type:?}' does not match to the value: {expr}"),
            }),
        },
        Expr::Literal(ScalarValue::Boolean(_)) | Expr::Literal(ScalarValue::Null) => {
            Ok(expr.clone())
        }
        _ => Err(Error::IO {
            message: format!("DataType '{data_type:?}' does not match to the value: {expr}"),
        }),
    }
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
                (Expr::Column(l), Expr::Literal(_)) => {
                    let Some(field) = schema.field(&l.flat_name()) else {
                        return Err(Error::IO {
                            message: format!(
                                "Column {} does not exist in the dataset.",
                                l.flat_name()
                            ),
                        });
                    };
                    Ok(Expr::BinaryExpr(BinaryExpr {
                        left: left.clone(),
                        op: *op,
                        right: Box::new(resolve_value(right.as_ref(), &field.data_type())?),
                    }))
                }
                (Expr::Literal(_), Expr::Column(l)) => {
                    let Some(field) = schema.field(&l.flat_name()) else {
                        return Err(Error::IO {
                            message: format!(
                                "Column {} does not exist in the dataset.",
                                l.flat_name()
                            ),
                        });
                    };
                    Ok(Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(resolve_value(right.as_ref(), &field.data_type())?),
                        op: *op,
                        right: right.clone(),
                    }))
                }
                // For cases complex expressions (not just literals) on right hand side like x = 1 + 1 + -2*2
                (Expr::Column(l), Expr::BinaryExpr(r)) => {
                    let Some(field) = schema.field(&l.flat_name()) else {
                        return Err(Error::IO {
                            message: format!(
                                "Column {} does not exist in the dataset.",
                                l.flat_name()
                            ),
                        });
                    };
                    Ok(Expr::BinaryExpr(BinaryExpr {
                        left: left.clone(),
                        op: *op,
                        right: Box::new(Expr::BinaryExpr(BinaryExpr {
                            left: coerce_expr(&r.left, &field.data_type()).map(Box::new)?,
                            op: r.op,
                            right: coerce_expr(&r.right, &field.data_type()).map(Box::new)?,
                        })),
                    }))
                }

                _ => Ok(expr.clone()),
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

/// Coerce logical expression for filters to bollean.
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
    use super::*;

    use arrow_schema::{Field, Schema as ArrowSchema};

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
}
