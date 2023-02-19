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
use datafusion::prelude::Expr;
use sqlparser::{
    ast::{SetExpr, Statement},
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
        todo!()
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

        let expr = planner.parse_filter("i > 10 AND st.x == 2.5").unwrap();
        println!("Expr: {}", expr);
    }
}
