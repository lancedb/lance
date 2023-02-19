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

use arrow_schema::SchemaRef;
use datafusion::{
    logical_expr::{col, BinaryExpr, Operator},
    physical_plan::{
        expressions::{IsNotNullExpr, IsNullExpr, Literal, NotExpr},
        PhysicalExpr,
    },
    prelude::Expr,
    scalar::ScalarValue,
};
use sqlparser::{
    ast::{BinaryOperator, Expr as SQLExpr, Ident, SetExpr, Statement, Value},
    dialect::GenericDialect,
    parser::Parser,
};

use crate::{datatypes::Schema, Error, Result};

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
            _ => return Err(Error::IO(format!("Operator {op} is not supported"))),
        })
    }

    fn binary_expr(&self, left: &SQLExpr, op: &BinaryOperator, right: &SQLExpr) -> Result<Expr> {
        Ok(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(self.parse_sql_expr(left)?),
            self.binary_op(op)?,
            Box::new(self.parse_sql_expr(right)?),
        )))
    }

    // See datafusion `sqlToRel::parse_sql_number()`
    fn number(&self, value: &str) -> Result<Expr> {
        use datafusion::logical_expr::lit;
        if let Ok(n) = value.parse::<i64>() {
            Ok(lit(n))
        } else {
            value
                .parse::<f64>()
                .map(lit)
                .map_err(|_| Error::IO(format!("'{value}' is not supported number value.")))
        }
    }

    fn value(&self, value: &Value) -> Result<Expr> {
        use datafusion::scalar::ScalarValue;

        Ok(match value {
            Value::Number(v, _) => self.number(v.as_str())?,
            Value::SingleQuotedString(s) => Expr::Literal(ScalarValue::Utf8(Some(s.clone()))),
            Value::DollarQuotedString(s) => todo!(),
            Value::EscapedStringLiteral(_) => todo!(),
            Value::NationalStringLiteral(_) => todo!(),
            Value::HexStringLiteral(_) => todo!(),
            Value::DoubleQuotedString(_) => todo!(),
            Value::Boolean(v) => Expr::Literal(ScalarValue::Boolean(Some(*v))),
            Value::Null => Expr::Literal(ScalarValue::Null),
            Value::Placeholder(_) => todo!(),
            Value::UnQuotedString(_) => todo!(),
        })
    }

    /// True value
    ///
    /// For the expression that Lance does not support, just return True, and
    /// let the upper runtime engine to filter.
    fn true_value(&self) -> Result<Expr> {
        Ok(Expr::Literal(ScalarValue::Boolean(Some(true))))
    }

    fn parse_sql_expr(&self, expr: &SQLExpr) -> Result<Expr> {
        match expr {
            SQLExpr::Identifier(id) => self.column(vec![id.clone()].as_slice()),
            SQLExpr::CompoundIdentifier(ids) => self.column(ids.as_slice()),
            SQLExpr::BinaryOp { left, op, right } => self.binary_expr(left, op, right),
            SQLExpr::Value(value) => self.value(value),
            SQLExpr::IsFalse(expr) => Ok(Expr::IsFalse(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsNotFalse(_) => Ok(Expr::IsNotFalse(Box::new(self.parse_sql_expr(expr)?))),
            SQLExpr::IsTrue(expr) => Ok(Expr::IsTrue(Box::new(self.parse_sql_expr(expr)?))),
            sqlparser::ast::Expr::IsNotTrue(expr) => {
                Ok(Expr::IsNotTrue(Box::new(self.parse_sql_expr(expr)?)))
            }
            sqlparser::ast::Expr::IsNull(expr) => {
                Ok(Expr::IsNull(Box::new(self.parse_sql_expr(expr)?)))
            }
            sqlparser::ast::Expr::IsNotNull(_) => {
                Ok(Expr::IsNotNull(Box::new(self.parse_sql_expr(expr)?)))
            }
            _ => {
                return Err(Error::IO(format!(
                    "Expression '{expr}' is not supported as filter in lance"
                )))
            }
        }
    }

    /// Create Logical [Expr] from a SQL filter clause.
    pub fn parse_filter(&self, filter: &str) -> Result<Expr> {
        let sql = format!("SELECT 1 FROM t WHERE {filter}");

        let dialect = GenericDialect {};
        let stmts = Parser::parse_sql(&dialect, sql.as_str())?;
        if stmts.len() != 1 {
            return Err(Error::IO(format!("Filter is not valid: {filter}")));
        }
        let selection = if let Statement::Query(query) = &stmts[0] {
            if let SetExpr::Select(s) = query.body.as_ref() {
                s.selection.as_ref()
            } else {
                None
            }
        } else {
            None
        };
        let expr = selection.ok_or_else(|| Error::IO(format!("Filter is not valid: {filter}")))?;

        self.parse_sql_expr(expr)
    }

    /// Create the [`PhysicalExpr`] from a logical [`Expr`]
    pub fn create_physical_expr(&self, expr: &Expr) -> Result<Arc<dyn PhysicalExpr>> {
        use crate::datafusion::physical_expr::Column;
        use datafusion::physical_expr::expressions::BinaryExpr;

        Ok(match expr {
            Expr::Column(c) => {
                println!("Matched to column: col={:#?}", c);
                Arc::new(Column::new(c.flat_name()))
            }
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
            _ => {
                return Err(Error::IO(format!(
                    "Expression '{expr}' is not supported as filter in lance"
                )))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float32Array, Int32Array, RecordBatch, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        logical_expr::{col, lit},
        physical_expr::expressions::{binary, lit as phy_lit, BinaryExpr},
    };

    use crate::datafusion::physical_expr::col as phy_col;

    #[test]
    fn test_parse_filter_simple() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, true),
            Field::new(
                "st",
                DataType::Struct(vec![
                    Field::new("x", DataType::Float32, false),
                    Field::new("y", DataType::Float32, false),
                ]),
                true,
            ),
        ]));

        let planner = Planner::new(schema.clone());

        let expr = planner
            .parse_filter("i > 10 AND st.x = 2.5 AND s = 'abc'")
            .unwrap();
        println!("Expr: {:#?}", expr);
        assert_eq!(
            expr,
            col("i")
                .gt(lit(10_i64))
                .and(col("st.x").eq(lit(2.5)))
                .and(col("s").eq(lit("abc")))
        );

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
        let value = physical_expr.evaluate(&batch).unwrap();
        println!("Evaluate value: {:?}\n", value);
    }
}
