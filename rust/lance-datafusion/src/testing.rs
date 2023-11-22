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

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion::{
    optimizer::simplify_expressions::SimplifyContext,
    sql::{
        planner::{ContextProvider, PlannerContext, SqlToRel},
        sqlparser::{dialect::PostgreSqlDialect, parser::Parser},
    },
};
use datafusion_common::{config::ConfigOptions, DFSchema, TableReference};
use datafusion_expr::{AggregateUDF, Expr, ScalarUDF, TableSource, WindowUDF};

use datafusion_physical_expr::execution_props::ExecutionProps;
use lance_core::Result;

struct MockContextProvider {}

// We're just compiling simple expressions (not entire statements) and so this is unused
impl ContextProvider for MockContextProvider {
    fn get_table_provider(
        &self,
        _: TableReference,
    ) -> datafusion_common::Result<std::sync::Arc<dyn TableSource>> {
        todo!()
    }

    fn get_function_meta(&self, _: &str) -> Option<std::sync::Arc<ScalarUDF>> {
        todo!()
    }

    fn get_aggregate_meta(&self, _: &str) -> Option<std::sync::Arc<AggregateUDF>> {
        todo!()
    }

    fn get_window_meta(&self, _: &str) -> Option<std::sync::Arc<WindowUDF>> {
        todo!()
    }

    fn get_variable_type(&self, _: &[String]) -> Option<DataType> {
        todo!()
    }

    fn options(&self) -> &ConfigOptions {
        todo!()
    }
}

pub struct TestingSqlParser {
    dialect: PostgreSqlDialect,
    schema: Arc<DFSchema>,
    planner_context: PlannerContext,
    context_provider: MockContextProvider,
}

impl TestingSqlParser {
    pub fn try_new(schema: Schema) -> Result<Self> {
        let df_schema = DFSchema::try_from(schema)?;

        Ok(Self {
            dialect: PostgreSqlDialect {},
            schema: Arc::new(df_schema),
            planner_context: PlannerContext::new(),
            context_provider: MockContextProvider {},
        })
    }
}

impl TestingSqlParser {
    pub fn parse_expr(&mut self, expr: &str) -> Result<Expr> {
        let mut parser = Parser::new(&self.dialect).try_with_sql(expr)?;
        let expr = parser.parse_expr()?;
        let planner = SqlToRel::new(&self.context_provider);
        Ok(planner.sql_to_expr(expr, &self.schema, &mut self.planner_context)?)
    }

    pub fn optimize_expr(&mut self, expr: Expr) -> Result<Expr> {
        let props = ExecutionProps::default();
        let simplify_context = SimplifyContext::new(&props).with_schema(self.schema.clone());
        let simplifier =
            datafusion::optimizer::simplify_expressions::ExprSimplifier::new(simplify_context);
        let expr = simplifier.simplify(expr.clone())?;
        let expr = simplifier.coerce(expr, self.schema.clone())?;

        Ok(expr)
    }
}
