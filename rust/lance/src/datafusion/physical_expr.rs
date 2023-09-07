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

use crate::arrow::*;
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

impl PartialEq<dyn Any> for Column {
    fn eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .map(|x| self == x)
            .unwrap_or(false)
    }
}

impl PhysicalExpr for Column {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, schema: &ArrowSchema) -> Result<DataType> {
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
        let array = batch
            .column_by_qualified_name(&self.name)
            .ok_or_else(|| DataFusionError::Plan(format!("column {} does not exist", self.name)))?;
        Ok(ColumnarValue::Array(array.clone()))
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

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        use std::hash::Hash;
        let mut s = state;
        self.hash(&mut s);
    }
}

struct ColumnVisitor {
    columns: Vec<String>,
}

impl ColumnVisitor {
    fn new() -> Self {
        Self { columns: vec![] }
    }

    fn visit(&mut self, expr: &dyn PhysicalExpr) {
        if let Some(c) = expr.as_any().downcast_ref::<Column>() {
            self.columns.push(c.name.clone())
        }

        expr.children().iter().for_each(|e| self.visit(e.as_ref()))
    }
}

/// Collect the columns in the physical expression.
pub fn column_names_in_expr(expr: &dyn PhysicalExpr) -> Vec<String> {
    let mut visitor = ColumnVisitor::new();
    visitor.visit(expr);
    visitor.columns
}

#[cfg(test)]
mod tests {

    use super::*;

    use arrow_array::{ArrayRef, Float32Array, Int32Array, StringArray, StructArray};
    use arrow_schema::{Field, Fields};

    #[test]
    fn test_simple_column() {
        let schema = Arc::new(ArrowSchema::new(vec![
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

        let column = Column::new("i".to_string());
        assert_eq!(column.data_type(schema.as_ref()).unwrap(), DataType::Int32);
        assert!(!column.nullable(schema.as_ref()).unwrap());

        let column = Column::new("s".to_string());
        assert_eq!(column.data_type(schema.as_ref()).unwrap(), DataType::Utf8);
        assert!(column.nullable(schema.as_ref()).unwrap());

        let column = Column::new("st.x".to_string());
        assert_eq!(
            column.data_type(schema.as_ref()).unwrap(),
            DataType::Float32
        );
        assert!(!column.nullable(schema.as_ref()).unwrap());
    }

    #[test]
    fn test_column_evaluate() {
        let schema = Arc::new(ArrowSchema::new(vec![
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

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(0..10)) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..10).map(|v| format!("str-{}", v)),
                )),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(Field::new("x", DataType::Float32, false)),
                        Arc::new(Float32Array::from_iter_values((0..10).map(|v| v as f32)))
                            as ArrayRef,
                    ),
                    (
                        Arc::new(Field::new("y", DataType::Float32, false)),
                        Arc::new(Float32Array::from_iter_values(
                            (0..10).map(|v| (v * 10) as f32),
                        )),
                    ),
                ])),
            ],
        )
        .unwrap();

        let column = Column::new("i".to_string());
        assert_eq!(
            column.evaluate(&batch).unwrap().into_array(0).as_ref(),
            &Int32Array::from_iter_values(0..10)
        );

        let column = Column::new("s".to_string());
        assert_eq!(
            column.evaluate(&batch).unwrap().into_array(0).as_ref(),
            &StringArray::from_iter_values((0..10).map(|v| format!("str-{}", v)))
        );

        let column = Column::new("st.x".to_string());
        assert_eq!(
            column.evaluate(&batch).unwrap().into_array(0).as_ref(),
            &Float32Array::from_iter_values((0..10).map(|v| v as f32))
        );
    }
}
