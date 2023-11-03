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

use async_trait::async_trait;
use datafusion::{datasource::TableProvider, execution::context::SessionState, logical_expr::TableType, prelude::Expr, error::{Result as DataFusionResult, DataFusionError}, physical_plan::ExecutionPlan};

use crate::dataset::scanner::Scanner;

struct LanceTableProvider {
    scanner: Scanner,
}

impl LanceTableProvider {
    pub fn new(scanner: Scanner) -> Self {
        Self { scanner }
    }
}

#[async_trait]
impl TableProvider for LanceTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.scanner.schema().expect("cannot convert to arrow schema")
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &SessionState,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut scanner = self.scanner.clone();
        // TODO: need to switch expr to filter
        // scanner.filter(filter)

        // TODO: Use this to configure partitions in the table.
        let num_partitions = state.config().options().execution.target_partitions;

        if let Some(projection) = projection {
            let schema = scanner.schema().unwrap();
            let columns = schema.fields();
            let columns = columns
                .iter()
                .enumerate()
                .filter_map(|(i, f)| {
                    if projection.contains(&i) {
                        Some(f.name().clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            scanner.project(&columns)?;
        }

        scanner.limit(limit.map(|l| l as i64), None);

        scanner.create_plan().await.map_err(|err| DataFusionError::Execution(err.to_string()))
    }
}