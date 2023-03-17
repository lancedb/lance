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

//! SQL Parser utility

use sqlparser::{
    ast::{Expr, SetExpr, Statement},
    dialect::GenericDialect,
    parser::Parser,
    tokenizer::{Token, Tokenizer},
};

use crate::{Error, Result};

/// Parse sql filter to Expression.
pub(crate) fn parse_sql_filter(filter: &str) -> Result<Expr> {
    let sql = format!("SELECT 1 FROM t WHERE {filter}");

    let dialect = GenericDialect {};
    let mut tokenizer = Tokenizer::new(&dialect, &sql);
    let tokens = tokenizer
        .tokenize()?
        .iter()
        .map(|tok| {
            if tok == &Token::DoubleEq {
                Token::Eq
            } else {
                tok.clone()
            }
        })
        .collect();

    let stmts = Parser::new(&dialect)
        .with_tokens(tokens)
        .parse_statements()?;

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
    Ok(expr.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    use sqlparser::ast::{BinaryOperator, Ident, Value};

    #[test]
    fn test_double_equal() {
        let expr = parse_sql_filter("a == b").unwrap();
        assert_eq!(
            Expr::BinaryOp {
                left: Box::new(Expr::Identifier(Ident::new("a"))),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::Identifier(Ident::new("b")))
            },
            expr
        );
    }

    #[test]
    fn test_like() {
        let expr = parse_sql_filter("a LIKE 'abc%'").unwrap();
        assert_eq!(
            Expr::Like {
                negated: false,
                expr: Box::new(Expr::Identifier(Ident::new("a"))),
                pattern: Box::new(Expr::Value(Value::SingleQuotedString("abc%".to_string()))),
                escape_char: None
            },
            expr
        );
    }
}
