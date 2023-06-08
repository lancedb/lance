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

//! Exec plan planner

use std::sync::Arc;

use arrow_cast::DEFAULT_CAST_OPTIONS;
use arrow_schema::{DataType as ArrowDataType, SchemaRef, TimeUnit};
use datafusion::{
    logical_expr::{col, BinaryExpr, BuiltinScalarFunction, Like, Operator},
    physical_expr::execution_props::ExecutionProps,
    physical_plan::{
        expressions::{
            CastExpr, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
        },
        functions, PhysicalExpr,
    },
    prelude::Expr,
    scalar::ScalarValue,
};
use sqlparser::ast::{
    BinaryOperator, DataType as SQLDataType, Expr as SQLExpr, Function, FunctionArg,
    FunctionArgExpr, Ident, TimezoneInfo, UnaryOperator, Value,
};

use crate::datafusion::logical_expr::coerce_filter_type_to_boolean;
use crate::{
    datafusion::logical_expr::resolve_expr, datatypes::Schema, utils::sql::parse_sql_filter, Error,
    Result,
};

pub struct Planner {
    schema: SchemaRef,
}

impl Planner {
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }

    fn column(&self, idents: &[Ident]) -> Result<Expr> {
        Ok(col(idents
            .iter()
            .map(|id| id.value.clone())
            .collect::<Vec<_>>()
            .join(".")))
    }

    fn binary_op(&self, op: &BinaryOperator) -> Result<Operator> {
        Ok(match op {
            BinaryOperator::Plus => Operator::Plus,
            BinaryOperator::Minus => Operator::Minus,
            BinaryOperator::Multiply => Operator::Multiply,
            BinaryOperator::Divide => Operator::Divide,
            BinaryOperator::Modulo => Operator::Modulo,
            BinaryOperator::StringConcat => Operator::StringConcat,
            BinaryOperator::Gt => Operator::Gt,
            BinaryOperator::Lt => Operator::Lt,
            BinaryOperator::GtEq => Operator::GtEq,
            BinaryOperator::LtEq => Operator::LtEq,
            BinaryOperator::Eq => Operator::Eq,
            BinaryOperator::NotEq => Operator::NotEq,
            BinaryOperator::And => Operator::And,
            BinaryOperator::Or => Operator::Or,
            _ => {
                return Err(Error::IO {
                    message: format!("Operator {op} is not supported"),
                })
            }
        })
    }

    fn binary_expr(&self, left: &SQLExpr, op: &BinaryOperator, right: &SQLExpr) -> Result<Expr> {
        Ok(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(self.parse_sql_expr(left)?),
            self.binary_op(op)?,
            Box::new(self.parse_sql_expr(right)?),
        )))
    }

    fn unary_expr(&self, op: &UnaryOperator, expr: &SQLExpr) -> Result<Expr> {
        Ok(match op {
            UnaryOperator::Not | UnaryOperator::PGBitwiseNot => {
                Expr::Not(Box::new(self.parse_sql_expr(expr)?))
            }
            _ => {
                return Err(Error::IO {
                    message: format!("Unary operator '{:?}' is not supported", op),
                })
            }
        })
    }

    // See datafusion `sqlToRel::parse_sql_number()`
    fn number(&self, value: &str) -> Result<Expr> {
        use datafusion::logical_expr::lit;
        if let Ok(n) = value.parse::<i64>() {
            Ok(lit(n))
        } else {
            value.parse::<f64>().map(lit).map_err(|_| Error::IO {
                message: format!("'{value}' is not supported number value."),
            })
        }
    }

    fn value(&self, value: &Value) -> Result<Expr> {
        Ok(match value {
            Value::Number(v, _) => self.number(v.as_str())?,
            Value::SingleQuotedString(s) => Expr::Literal(ScalarValue::Utf8(Some(s.clone()))),
            Value::DollarQuotedString(_) => todo!(),
            Value::EscapedStringLiteral(_) => todo!(),
            Value::NationalStringLiteral(_) => todo!(),
            Value::HexStringLiteral(_) => todo!(),
            Value::DoubleQuotedString(s) => Expr::Literal(ScalarValue::Utf8(Some(s.clone()))),
            Value::Boolean(v) => Expr::Literal(ScalarValue::Boolean(Some(*v))),
            Value::Null => Expr::Literal(ScalarValue::Null),
            Value::Placeholder(_) => todo!(),
            Value::UnQuotedString(_) => todo!(),
            Value::SingleQuotedByteStringLiteral(_) => todo!(),
            Value::DoubleQuotedByteStringLiteral(_) => todo!(),
            Value::RawStringLiteral(_) => todo!(),
        })
    }

    fn parse_function_args(&self, func_args: &FunctionArg) -> Result<Expr> {
        match func_args {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => self.parse_sql_expr(expr),
            _ => Err(Error::IO {
                message: format!("Unsupported function args: {:?}", func_args),
            }),
        }
    }

    fn parse_function(&self, func: &Function) -> Result<Expr> {
        if func.name.to_string() == "is_valid" {
            if func.args.len() != 1 {
                return Err(Error::IO {
                    message: format!("is_valid only support 1 args, got {}", func.args.len()),
                });
            }
            return Ok(Expr::IsNotNull(Box::new(
                self.parse_function_args(&func.args[0])?,
            )));
        } else if func.name.to_string() == "regexp_match" {
            if func.args.len() != 2 {
                return Err(Error::IO {
                    message: format!("regexp_match only supports 2 args, got {}", func.args.len()),
                });
            }

            let args_vec: Vec<Expr> = func
                .args
                .iter()
                .map(|arg| self.parse_function_args(arg).unwrap())
                .collect::<Vec<_>>();

            return Ok(Expr::ScalarFunction {
                fun: BuiltinScalarFunction::RegexpMatch,
                args: args_vec,
            });
        }
        Err(Error::IO {
            message: format!("function '{}' is not supported", func.name),
        })
    }

    fn parse_type(&self, data_type: &SQLDataType) -> Result<ArrowDataType> {
        match data_type {
            SQLDataType::String => Ok(ArrowDataType::Utf8),
            SQLDataType::Binary(_) => Ok(ArrowDataType::Binary),
            SQLDataType::Float(_) => Ok(ArrowDataType::Float32),
            SQLDataType::Double => Ok(ArrowDataType::Float64),
            SQLDataType::Boolean => Ok(ArrowDataType::Boolean),
            SQLDataType::TinyInt(_) => Ok(ArrowDataType::Int8),
            SQLDataType::SmallInt(_) => Ok(ArrowDataType::Int16),
            SQLDataType::Int(_) | SQLDataType::Integer(_) => Ok(ArrowDataType::Int32),
            SQLDataType::BigInt(_) => Ok(ArrowDataType::Int64),
            SQLDataType::UnsignedTinyInt(_) => Ok(ArrowDataType::UInt8),
            SQLDataType::UnsignedSmallInt(_) => Ok(ArrowDataType::UInt16),
            SQLDataType::UnsignedInt(_) => Ok(ArrowDataType::UInt32),
            SQLDataType::UnsignedBigInt(_) => Ok(ArrowDataType::UInt64),
            SQLDataType::Date => Ok(ArrowDataType::Date32),
            SQLDataType::Timestamp(resolution, tz) => {
                match tz {
                    TimezoneInfo::None => {},
                    _ => {
                        return Err(Error::IO {
                            message: format!("Timezone not supported in timestamp"),
                        })
                    }
                };
                let time_unit = match resolution {
                    // Default to microsecond to match PyArrow
                    None => TimeUnit::Microsecond,
                    Some(0) => TimeUnit::Second,
                    Some(3) => TimeUnit::Millisecond,
                    Some(6) => TimeUnit::Microsecond,
                    Some(9) => TimeUnit::Nanosecond,
                    _ => {
                        return Err(Error::IO {
                            message: format!("Unsupported datetime resolution: {:?}", resolution),
                        })
                    }
                };
                Ok(ArrowDataType::Timestamp(time_unit, None))
            }
            SQLDataType::Datetime(resolution) => {
                let time_unit = match resolution {
                    None => TimeUnit::Microsecond,
                    Some(0) => TimeUnit::Second,
                    Some(3) => TimeUnit::Millisecond,
                    Some(6) => TimeUnit::Microsecond,
                    Some(9) => TimeUnit::Nanosecond,
                    _ => {
                        return Err(Error::IO {
                            message: format!("Unsupported datetime resolution: {:?}", resolution),
                        })
                    }
                };
                Ok(ArrowDataType::Timestamp(time_unit, None))
            }
            _ => Err(Error::IO {
                message: format!("Unsupported data type: {:?}. Supports: float, double, boolean, int, integer, date, datetime", data_type),
            }),
        }
    }

    fn parse_sql_expr(&self, expr: &SQLExpr) -> Result<Expr> {
        match expr {
            SQLExpr::Identifier(id) => {
                if id.quote_style == Some('"') {
                    Ok(Expr::Literal(ScalarValue::Utf8(Some(id.value.clone()))))
                } else {
                    self.column(vec![id.clone()].as_slice())
                }
            }
            SQLExpr::CompoundIdentifier(ids) => self.column(ids.as_slice()),
            SQLExpr::BinaryOp { left, op, right } => self.binary_expr(left, op, right),
            SQLExpr::UnaryOp { op, expr } => self.unary_expr(op, expr),
            SQLExpr::Value(value) => self.value(value),
            // For example, DATE '2020-01-01'
            SQLExpr::TypedString { data_type, value } => {
                Ok(Expr::Cast(datafusion::logical_expr::Cast {
                    expr: Box::new(Expr::Literal(ScalarValue::Utf8(Some(value.clone())))),
                    data_type: self.parse_type(data_type)?,
                }))
            }
            SQLExpr::IsFalse(expr) => Ok(Expr::IsFalse(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotFalse(_) => Ok(Expr::IsNotFalse(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsTrue(expr) => Ok(Expr::IsTrue(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotTrue(expr) => Ok(Expr::IsNotTrue(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNull(expr) => Ok(Expr::IsNull(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotNull(expr) => Ok(Expr::IsNotNull(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::InList {
                expr,
                list,
                negated,
            } => {
                let value_expr = self.parse_sql_expr(expr)?;
                let list_exprs = list
                    .iter()
                    .map(|e| self.parse_sql_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(value_expr.in_list(list_exprs, *negated))
            }
            SQLExpr::Nested(inner) => self.parse_sql_expr(inner.as_ref()),
            SQLExpr::Function(func) => self.parse_function(func),
            SQLExpr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => Ok(Expr::Like(Like::new(
                *negated,
                Box::new(self.parse_sql_expr(expr)?),
                Box::new(self.parse_sql_expr(pattern)?),
                *escape_char,
            ))),
            SQLExpr::Cast { expr, data_type } => Ok(Expr::Cast(datafusion::logical_expr::Cast {
                expr: Box::new(self.parse_sql_expr(expr)?),
                data_type: self.parse_type(data_type)?,
            })),
            _ => {
                return Err(Error::IO {
                    message: format!("Expression '{expr}' is not supported as filter in lance"),
                })
            }
        }
    }

    /// Create Logical [Expr] from a SQL filter clause.
    pub fn parse_filter(&self, filter: &str) -> Result<Expr> {
        // Allow sqlparser to parse filter as part of ONE SQL statement.

        let ast_expr = parse_sql_filter(filter)?;
        let expr = self.parse_sql_expr(&ast_expr)?;
        let schema = Schema::try_from(self.schema.as_ref())?;
        let resolved = resolve_expr(&expr, &schema)?;
        coerce_filter_type_to_boolean(resolved)
    }

    /// Create the [`PhysicalExpr`] from a logical [`Expr`]
    pub fn create_physical_expr(&self, expr: &Expr) -> Result<Arc<dyn PhysicalExpr>> {
        use crate::datafusion::physical_expr::Column;
        use datafusion::physical_expr::expressions::BinaryExpr;

        Ok(match expr {
            Expr::Column(c) => Arc::new(Column::new(c.flat_name())),
            Expr::Literal(v) => Arc::new(Literal::new(v.clone())),
            Expr::BinaryExpr(expr) => Arc::new(BinaryExpr::new(
                self.create_physical_expr(expr.left.as_ref())?,
                expr.op,
                self.create_physical_expr(expr.right.as_ref())?,
            )),
            Expr::IsNotNull(expr) => Arc::new(IsNotNullExpr::new(self.create_physical_expr(expr)?)),
            Expr::IsNull(expr) => Arc::new(IsNullExpr::new(self.create_physical_expr(expr)?)),
            Expr::IsTrue(expr) => self.create_physical_expr(expr)?,
            Expr::IsFalse(expr) => Arc::new(NotExpr::new(self.create_physical_expr(expr)?)),
            Expr::InList {
                expr,
                list,
                negated,
            } => Arc::new(InListExpr::new(
                self.create_physical_expr(expr)?,
                list.iter()
                    .map(|e| self.create_physical_expr(e))
                    .collect::<Result<Vec<_>>>()?,
                *negated,
                self.schema.as_ref(),
            )),
            Expr::Like(expr) => Arc::new(LikeExpr::new(
                expr.negated,
                true,
                self.create_physical_expr(expr.expr.as_ref())?,
                self.create_physical_expr(expr.pattern.as_ref())?,
            )),
            Expr::Not(expr) => Arc::new(NotExpr::new(self.create_physical_expr(expr)?)),
            Expr::Cast(datafusion::logical_expr::Cast { expr, data_type }) => {
                let expr = self.create_physical_expr(expr.as_ref())?;
                Arc::new(CastExpr::new(expr, data_type.clone(), DEFAULT_CAST_OPTIONS))
            }
            Expr::ScalarFunction { fun, args } => {
                if fun != &BuiltinScalarFunction::RegexpMatch {
                    return Err(Error::IO {
                        message: format!("Scalar function '{:?}' is not supported", fun),
                    });
                }
                let execution_props = ExecutionProps::new();
                let args_vec = args
                    .iter()
                    .map(|e| self.create_physical_expr(e).unwrap())
                    .collect::<Vec<_>>();
                if args_vec.len() != 2 {
                    return Err(Error::IO {
                        message: format!(
                            "Scalar function '{:?}' only supports 2 args, got {}",
                            fun,
                            args_vec.len()
                        ),
                    });
                }

                let args_array: [Arc<dyn PhysicalExpr>; 2] =
                    [args_vec[0].clone(), args_vec[1].clone()];

                let physical_expr = functions::create_physical_expr(
                    fun,
                    &args_array,
                    self.schema.as_ref(),
                    &execution_props,
                );
                physical_expr?
            }
            _ => {
                return Err(Error::IO {
                    message: format!("Expression '{expr}' is not supported as filter in lance"),
                })
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Float32Array, Int32Array, RecordBatch, StringArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields, Schema};
    use datafusion::logical_expr::{col, lit};

    #[test]
    fn test_parse_filter_simple() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, true),
            Field::new(
                "st",
                DataType::Struct(Fields::from(vec![
                    Field::new("x", DataType::Float32, false),
                    Field::new("y", DataType::Float32, false),
                ])),
                true,
            ),
        ]));

        let planner = Planner::new(schema.clone());

        let expected = col("i")
            .gt(lit(3_i32))
            .and(col("st.x").lt_eq(lit(5.0_f32)))
            .and(
                col("s")
                    .eq(lit("str-4"))
                    .or(col("s").in_list(vec![lit("str-4"), lit("str-5")], false)),
            );

        // double quotes
        let expr = planner
            .parse_filter("i > 3 AND st.x <= 5.0 AND (s == 'str-4' OR s in ('str-4', 'str-5'))")
            .unwrap();
        assert_eq!(expr, expected);

        // single quote
        let expr = planner
            .parse_filter("i > 3 AND st.x <= 5.0 AND (s = 'str-4' OR s in ('str-4', 'str-5'))")
            .unwrap();

        let physical_expr = planner.create_physical_expr(&expr).unwrap();
        println!("Physical expr: {:#?}", physical_expr);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..10)) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..10).map(|v| format!("str-{}", v)),
                )),
                Arc::new(StructArray::from(vec![
                    (
                        Field::new("x", DataType::Float32, false),
                        Arc::new(Float32Array::from_iter_values((0..10).map(|v| v as f32)))
                            as ArrayRef,
                    ),
                    (
                        Field::new("y", DataType::Float32, false),
                        Arc::new(Float32Array::from_iter_values(
                            (0..10).map(|v| (v * 10) as f32),
                        )),
                    ),
                ])),
            ],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, true, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_sql_like() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").like(lit("str-4"));
        // single quote
        let expr = planner.parse_filter("s LIKE 'str-4'").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, false, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_not_like() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").not_like(lit("str-4"));
        // single quote
        let expr = planner.parse_filter("s NOT LIKE 'str-4'").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                true, true, true, true, false, true, true, true, true, true
            ])
        );
    }

    #[test]
    fn test_sql_is_in() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").in_list(vec![lit("str-4"), lit("str-5")], false);
        // single quote
        let expr = planner.parse_filter("s IN ('str-4', 'str-5')").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..10).map(|v| format!("str-{}", v)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, false, false, false, true, true, false, false, false, false
            ])
        );
    }

    #[test]
    fn test_sql_is_null() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));

        let planner = Planner::new(schema.clone());

        let expected = col("s").is_null();
        let expr = planner.parse_filter("s IS NULL").unwrap();
        assert_eq!(expr, expected);
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter((0..10).map(|v| {
                if v % 3 == 0 {
                    Some(format!("str-{}", v))
                } else {
                    None
                }
            })))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, true, true, false, true, true, false, true, true, false
            ])
        );

        let expr = planner.parse_filter("s IS NOT NULL").unwrap();
        let physical_expr = planner.create_physical_expr(&expr).unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                true, false, false, true, false, false, true, false, false, true,
            ])
        );
    }

    #[test]
    fn test_sql_invert() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Boolean, true)]));

        let planner = Planner::new(schema.clone());

        let expr = planner.parse_filter("NOT s").unwrap();
        let physical_expr = planner.create_physical_expr(&expr).unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BooleanArray::from_iter(
                (0..10).map(|v| Some(v % 3 == 0)),
            ))],
        )
        .unwrap();
        let predicates = physical_expr.evaluate(&batch).unwrap();
        assert_eq!(
            predicates.into_array(0).as_ref(),
            &BooleanArray::from(vec![
                false, true, true, false, true, true, false, true, true, false
            ])
        );
    }
}
