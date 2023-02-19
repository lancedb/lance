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
    prelude::Expr,
    scalar::ScalarValue,
};
use sqlparser::{
    ast::{BinaryOperator, Expr as SQLExpr, Ident, SetExpr, Statement, Value},
    dialect::GenericDialect,
    parser::Parser,
};

use crate::{Error, Result};

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
            Box::new(self.parse_expr(left)?),
            self.binary_op(op)?,
            Box::new(self.parse_expr(right)?),
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
            Value::DollarQuotedString(_) => todo!(),
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

    fn parse_expr(&self, expr: &SQLExpr) -> Result<Expr> {
        match expr {
            SQLExpr::Identifier(id) => self.column(vec![id.clone()].as_slice()),
            SQLExpr::CompoundIdentifier(ids) => self.column(ids.as_slice()),
            SQLExpr::BinaryOp { left, op, right } => self.binary_expr(left, op, right),
            SQLExpr::Value(value) => self.value(value),
            sqlparser::ast::Expr::CompositeAccess { expr, key } => todo!(),
            sqlparser::ast::Expr::IsFalse(_) => todo!(),
            sqlparser::ast::Expr::IsNotFalse(_) => todo!(),
            sqlparser::ast::Expr::IsTrue(expr) => {
                Ok(Expr::IsTrue(Box::new(self.parse_expr(expr)?)))
            }
            sqlparser::ast::Expr::IsNotTrue(_) => todo!(),
            sqlparser::ast::Expr::IsNull(expr) => {
                Ok(Expr::IsNull(Box::new(self.parse_expr(expr)?)))
            }
            sqlparser::ast::Expr::IsNotNull(_) => todo!(),
            sqlparser::ast::Expr::IsUnknown(_) => todo!(),
            sqlparser::ast::Expr::IsNotUnknown(_) => todo!(),
            sqlparser::ast::Expr::IsDistinctFrom(_, _) => todo!(),
            sqlparser::ast::Expr::IsNotDistinctFrom(_, _) => todo!(),
            sqlparser::ast::Expr::InList {
                expr,
                list,
                negated,
            } => todo!(),
            sqlparser::ast::Expr::Between {
                expr,
                negated,
                low,
                high,
            } => todo!(),
            sqlparser::ast::Expr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => todo!(),
            sqlparser::ast::Expr::ILike {
                negated,
                expr,
                pattern,
                escape_char,
            } => todo!(),
            sqlparser::ast::Expr::SimilarTo {
                negated,
                expr,
                pattern,
                escape_char,
            } => todo!(),
            sqlparser::ast::Expr::AnyOp(_) => todo!(),
            sqlparser::ast::Expr::AllOp(_) => todo!(),
            sqlparser::ast::Expr::UnaryOp { op, expr } => todo!(),
            sqlparser::ast::Expr::Cast { expr, data_type } => todo!(),
            sqlparser::ast::Expr::TryCast { expr, data_type } => todo!(),
            sqlparser::ast::Expr::SafeCast { expr, data_type } => todo!(),
            sqlparser::ast::Expr::AtTimeZone {
                timestamp,
                time_zone,
            } => todo!(),
            sqlparser::ast::Expr::Extract { field, expr } => todo!(),
            sqlparser::ast::Expr::Ceil { expr, field } => todo!(),
            sqlparser::ast::Expr::Floor { expr, field } => todo!(),
            sqlparser::ast::Expr::Position { expr, r#in } => todo!(),
            sqlparser::ast::Expr::Substring {
                expr,
                substring_from,
                substring_for,
            } => todo!(),
            sqlparser::ast::Expr::Collate { expr, collation } => todo!(),
            sqlparser::ast::Expr::Nested(_) => todo!(),

            sqlparser::ast::Expr::TypedString { data_type, value } => todo!(),
            sqlparser::ast::Expr::MapAccess { column, keys } => todo!(),
            sqlparser::ast::Expr::Function(_) => todo!(),
            sqlparser::ast::Expr::AggregateExpressionWithFilter { expr, filter } => todo!(),
            sqlparser::ast::Expr::Case {
                operand,
                conditions,
                results,
                else_result,
            } => todo!(),
            sqlparser::ast::Expr::Exists { subquery, negated } => todo!(),
            sqlparser::ast::Expr::Subquery(_) => todo!(),
            sqlparser::ast::Expr::ArraySubquery(_) => todo!(),
            sqlparser::ast::Expr::ListAgg(_) => todo!(),
            sqlparser::ast::Expr::ArrayAgg(_) => todo!(),
            sqlparser::ast::Expr::Tuple(_) => todo!(),
            sqlparser::ast::Expr::ArrayIndex { obj, indexes } => todo!(),
            sqlparser::ast::Expr::Array(_) => todo!(),
            sqlparser::ast::Expr::Interval {
                value,
                leading_field,
                leading_precision,
                last_field,
                fractional_seconds_precision,
            } => todo!(),
            sqlparser::ast::Expr::MatchAgainst {
                columns,
                match_value,
                opt_search_modifier,
            } => todo!(),
            _ => {
                return Err(Error::IO(format!(
                    "Expression '{expr}' is not supported in lance"
                )))
            }
        }
    }

    /// Create Logical [Expr] from a SQL filter statement
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

        self.parse_expr(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field, Schema};

    #[test]
    fn test_resolve_simple() {
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

        let planner = Planner::new(schema);

        let expr = planner.parse_filter("i > 10 AND st.x = 2.5 AND s = 'abc'").unwrap();
        println!("Expr: {}", expr);
    }
}
