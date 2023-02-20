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
use datafusion::logical_expr::Operator;
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
            _ => {
                return Err(Error::IO(format!(
                    "DataType '{data_type:?}' does not match to the value: {expr}"
                )));
            }
        },
        Expr::Literal(ScalarValue::Float64(v)) => match data_type {
            DataType::Float32 => Ok(Expr::Literal(ScalarValue::Float32(v.map(|v| v as f32)))),
            DataType::Float64 => Ok(Expr::Literal(ScalarValue::Float64(*v))),
            _ => {
                return Err(Error::IO(format!(
                    "DataType '{data_type:?}' does not match to the value: {expr}"
                )));
            }
        },
        Expr::Literal(ScalarValue::Utf8(_))
        | Expr::Literal(ScalarValue::Boolean(_))
        | Expr::Literal(ScalarValue::Null) => Ok(expr.clone()),
        _ => Err(Error::IO(format!(
            "DataType '{data_type:?}' does not match to the value: {expr}"
        ))),
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
                (Expr::Column(l), Expr::Literal(r)) => {
                    let Some(field) = schema.field(&l.flat_name()) else {
                        return Err(Error::IO(format!("Column {} does not exist in the dataset.", l.flat_name())))
                    };
                    return Ok(Expr::BinaryExpr(BinaryExpr {
                        left: left.clone(),
                        op: *op,
                        right: Box::new(resolve_value(right.as_ref(), &field.data_type())?),
                    }));
                }
                (Expr::Literal(_), Expr::Column(l)) => {
                    let Some(field) = schema.field(&l.flat_name()) else {
                        return Err(Error::IO(format!("Column {} does not exist in the dataset.", l.flat_name())))
                    };
                    return Ok(Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(resolve_value(right.as_ref(), &field.data_type())?),
                        op: *op,
                        right: right.clone(),
                    }));
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

#[cfg(test)]
mod tests {}
