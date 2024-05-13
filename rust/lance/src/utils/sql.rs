// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SQL Parser utility

use datafusion::sql::sqlparser::{
    ast::{Expr, SelectItem, SetExpr, Statement},
    dialect::{Dialect, GenericDialect},
    parser::Parser,
    tokenizer::{Token, Tokenizer},
};

use crate::{Error, Result};
use snafu::{location, Location};
#[derive(Debug, Default)]
struct LanceDialect(GenericDialect);

impl LanceDialect {
    fn new() -> Self {
        Self(GenericDialect {})
    }
}

impl Dialect for LanceDialect {
    fn is_identifier_start(&self, ch: char) -> bool {
        self.0.is_identifier_start(ch)
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        self.0.is_identifier_part(ch)
    }

    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        ch == '`'
    }
}

/// Parse sql filter to Expression.
pub(crate) fn parse_sql_filter(filter: &str) -> Result<Expr> {
    let sql = format!("SELECT 1 FROM t WHERE {filter}");
    let statement = parse_statement(&sql)?;

    let selection = if let Statement::Query(query) = &statement {
        if let SetExpr::Select(s) = query.body.as_ref() {
            s.selection.as_ref()
        } else {
            None
        }
    } else {
        None
    };
    let expr = selection
        .ok_or_else(|| Error::io(format!("Filter is not valid: {filter}"), location!()))?;
    Ok(expr.clone())
}

/// Parse a SQL expression to Expression. This is more lenient than parse_sql_filter
/// as it can be used for projection expressions as well.
pub(crate) fn parse_sql_expr(expr: &str) -> Result<Expr> {
    let sql = format!("SELECT {expr} FROM t");
    let statement = parse_statement(&sql)?;

    let selection = if let Statement::Query(query) = &statement {
        if let SetExpr::Select(s) = query.body.as_ref() {
            if let SelectItem::UnnamedExpr(expr) = &s.projection[0] {
                Some(expr)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let expr = selection
        .ok_or_else(|| Error::io(format!("Expression is not valid: {expr}"), location!()))?;
    Ok(expr.clone())
}

fn parse_statement(statement: &str) -> Result<Statement> {
    let dialect = LanceDialect::new();

    // Hack to allow == as equals
    // This is used to parse PyArrow expressions from strings.
    // See: https://github.com/sqlparser-rs/sqlparser-rs/pull/815#issuecomment-1450714278
    let mut tokenizer = Tokenizer::new(&dialect, statement);
    let mut tokens = Vec::new();
    let mut token_iter = tokenizer.tokenize()?.into_iter();
    let mut prev_token = token_iter.next().unwrap();
    for next_token in token_iter {
        if let (Token::Eq, Token::Eq) = (&prev_token, &next_token) {
            continue; // skip second equals
        }
        let token = std::mem::replace(&mut prev_token, next_token);
        tokens.push(token);
    }
    tokens.push(prev_token);

    Ok(Parser::new(&dialect)
        .with_tokens(tokens)
        .parse_statement()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::sql::sqlparser::ast::{BinaryOperator, Ident, Value};

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

    #[test]
    fn test_quoted_ident() {
        // CUBE is a SQL keyword, so it must be quoted.
        let expr = parse_sql_filter("`a:Test_Something` == `CUBE`").unwrap();
        assert_eq!(
            Expr::BinaryOp {
                left: Box::new(Expr::Identifier(Ident::with_quote('`', "a:Test_Something"))),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::Identifier(Ident::with_quote('`', "CUBE")))
            },
            expr
        );

        let expr = parse_sql_filter("`outer field`.`inner field` == 1").unwrap();
        assert_eq!(
            Expr::BinaryOp {
                left: Box::new(Expr::CompoundIdentifier(vec![
                    Ident::with_quote('`', "outer field"),
                    Ident::with_quote('`', "inner field")
                ])),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::Value(Value::Number("1".to_string(), false))),
            },
            expr
        );
    }
}
