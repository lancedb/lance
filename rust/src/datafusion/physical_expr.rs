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

//! Extends DataFusion Physical Expression


use std::any::Any;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use datafusion::{
    error::{DataFusionError, Result},
    physical_expr::PhysicalExpr,
    physical_plan::ColumnarValue,
};

use crate::datatypes::Schema;

/// Column expression.
///
/// The difference between it and [`datafusion::physical_expr::expressions::Column`] is that
/// this one supports nested column access.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct Column {
    /// Qualified column name.
    pub name: String,
}

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Column {
    /// Create a column with qualified column name.
    pub(crate) fn new(name: String) -> Self {
        Self { name }
    }
}

impl PhysicalExpr for Column {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, schema: &ArrowSchema) -> Result<DataType> {
        println!(
            "Fetch datatype: {:?}, {:?}",
            schema,
            Schema::try_from(schema)?.field(self.name.as_str())
        );

        Schema::try_from(schema)?
            .field(self.name.as_str())
            .map(|f| f.data_type())
            .ok_or_else(|| DataFusionError::Plan(format!("column {} does not exist", self.name)))
    }

    fn nullable(&self, schema: &ArrowSchema) -> Result<bool> {
        Schema::try_from(schema)?
            .field(self.name.as_str())
            .map(|f| f.nullable)
            .ok_or_else(|| DataFusionError::Plan(format!("column {} does not exist", self.name)))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        todo!()
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        todo!()
    }
}

impl PartialEq<dyn Any> for Column {
    fn eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .map(|x| self == x)
            .unwrap_or(false)
    }
}

/// Create a column expression.
pub fn col(name: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name.to_string()))
}

#[cfg(test)]
mod tests {}
