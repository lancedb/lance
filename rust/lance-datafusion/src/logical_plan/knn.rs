// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{self, Debug, Display};
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion_common::{DFSchema, DFSchemaRef, Result};
use lance_linalg::distance::DistanceType;

/// Logical plan node for K-Nearest Neighbor search
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KnnNode {
    /// Input logical plan
    pub input: LogicalPlan,
    /// Column name to search on
    pub column: String,
    /// Query vector expression (typically a literal, but can be a parameter or subquery)
    pub query: Expr,
    /// Number of nearest neighbors to find
    pub k: usize,
    /// Whether to use an index
    pub use_index: bool,
    /// Distance metric to use
    pub distance_metric: DistanceType,
    /// The output schema (includes _distance column)
    output_schema: DFSchemaRef,
}

impl KnnNode {
    pub fn try_new(
        input: LogicalPlan,
        column: String,
        query_vector: Expr,
        k: usize,
        use_index: bool,
        distance_metric: DistanceType,
    ) -> Result<Self> {
        // Validate that the column exists in the input schema
        let input_schema = input.schema();
        let column_field = input_schema.field_with_name(None, &column)?;

        // Validate that the column is a vector type (FixedSizeList)
        match column_field.data_type() {
            DataType::FixedSizeList(inner_field, _) => {
                // Validate the inner type is numeric
                if !inner_field.data_type().is_numeric() {
                    return Err(datafusion_common::DataFusionError::Plan(format!(
                        "KNN search column {} must contain numeric values",
                        column
                    )));
                }
            }
            _ => {
                return Err(datafusion_common::DataFusionError::Plan(format!(
                    "KNN search column {} must be a FixedSizeList (vector)",
                    column
                )));
            }
        }

        // Create output schema by adding distance column
        let mut qualified_fields = vec![];

        // Copy all fields from input
        for (qualifier, field) in input_schema.iter() {
            qualified_fields.push((qualifier.cloned(), field.clone()));
        }

        // Add _distance column
        let distance_field = Field::new("_distance", DataType::Float32, true);
        qualified_fields.push((None, Arc::new(distance_field)));

        let output_schema = Arc::new(DFSchema::new_with_metadata(
            qualified_fields,
            input_schema.inner().metadata().clone(),
        )?);

        Ok(Self {
            input,
            column,
            query: query_vector,
            k,
            use_index,
            distance_metric,
            output_schema,
        })
    }
}

impl PartialOrd for KnnNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Use column name for ordering
        Some(self.column.cmp(&other.column))
    }
}

impl UserDefinedLogicalNodeCore for KnnNode {
    fn name(&self) -> &str {
        "KnnSearch"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.output_schema
    }

    fn expressions(&self) -> Vec<Expr> {
        // Return the query vector expression
        vec![self.query.clone()]
    }

    fn prevent_predicate_push_down_columns(&self) -> std::collections::HashSet<String> {
        // We don't want predicates pushed down past the KNN node
        std::collections::HashSet::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "KnnSearch: column={}, k={}, metric={}, use_index={}",
            self.column, self.k, self.distance_metric, self.use_index
        )
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if inputs.len() != 1 {
            return Err(datafusion_common::DataFusionError::Plan(
                "KnnNode must have exactly one input".to_string(),
            ));
        }
        if exprs.len() != 1 {
            return Err(datafusion_common::DataFusionError::Plan(
                "KnnNode must have exactly one expression".to_string(),
            ));
        }

        Ok(Self {
            input: inputs.into_iter().next().unwrap(),
            column: self.column.clone(),
            query: exprs.into_iter().next().unwrap(),
            k: self.k,
            use_index: self.use_index,
            distance_metric: self.distance_metric,
            output_schema: self.output_schema.clone(),
        })
    }
}

impl Display for KnnNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_for_explain(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array;
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
    use datafusion::logical_expr::EmptyRelation;
    use datafusion_common::ScalarValue;
    use std::sync::Arc;

    fn create_test_schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                false,
            ),
            ArrowField::new("label", DataType::Utf8, true),
        ]))
    }

    #[test]
    fn test_knn_node_creation() {
        let schema = create_test_schema();
        let df_schema = DFSchema::try_from_qualified_schema("test", &schema).unwrap();
        let scan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        });

        // Create a literal expression for the query vector
        let query_vector_expr = Expr::Literal(ScalarValue::List(Arc::new(
            arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                vec![Some(vec![Some(1.0); 128])],
            ),
        )));

        let knn_node = KnnNode::try_new(
            scan,
            "vector".to_string(),
            query_vector_expr,
            10,
            true,
            DistanceType::L2,
        )
        .unwrap();

        assert_eq!(knn_node.column, "vector");
        assert_eq!(knn_node.k, 10);
        assert_eq!(knn_node.distance_metric, DistanceType::L2);
        assert!(knn_node.use_index);

        // Check that output schema has distance column
        let output_schema = knn_node.schema();
        assert!(output_schema.field_with_name(None, "_distance").is_ok());
    }

    #[test]
    fn test_knn_node_invalid_column() {
        let schema = create_test_schema();
        let df_schema = DFSchema::try_from_qualified_schema("test", &schema).unwrap();
        let scan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        });

        let query_vector_expr = Expr::Literal(ScalarValue::List(Arc::new(
            arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                vec![Some(vec![Some(1.0); 128])],
            ),
        )));

        // Test with non-existent column
        let result = KnnNode::try_new(
            scan.clone(),
            "nonexistent".to_string(),
            query_vector_expr.clone(),
            10,
            true,
            DistanceType::L2,
        );
        assert!(result.is_err());

        // Test with non-vector column
        let result = KnnNode::try_new(
            scan,
            "id".to_string(),
            query_vector_expr,
            10,
            true,
            DistanceType::L2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_knn_node_equality() {
        let schema = create_test_schema();
        let df_schema = DFSchema::try_from_qualified_schema("test", &schema).unwrap();
        let scan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        });

        let query_vector_expr1 = Expr::Literal(ScalarValue::List(Arc::new(
            arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                vec![Some(vec![Some(1.0); 128])],
            ),
        )));

        let query_vector_expr2 = Expr::Literal(ScalarValue::List(Arc::new(
            arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                vec![Some(vec![Some(2.0); 128])],
            ),
        )));

        let knn_node1 = KnnNode::try_new(
            scan.clone(),
            "vector".to_string(),
            query_vector_expr1.clone(),
            10,
            true,
            DistanceType::L2,
        )
        .unwrap();

        let knn_node2 = KnnNode::try_new(
            scan.clone(),
            "vector".to_string(),
            query_vector_expr1,
            10,
            true,
            DistanceType::L2,
        )
        .unwrap();

        let knn_node3 = KnnNode::try_new(
            scan,
            "vector".to_string(),
            query_vector_expr2,
            10,
            true,
            DistanceType::L2,
        )
        .unwrap();

        // Same query vectors should be equal
        assert_eq!(knn_node1, knn_node2);

        // Different query vectors should not be equal
        assert_ne!(knn_node1, knn_node3);
    }

    #[test]
    fn test_knn_node_display() {
        let schema = create_test_schema();
        let df_schema = DFSchema::try_from_qualified_schema("test", &schema).unwrap();
        let scan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(df_schema),
        });

        let query_vector_expr = Expr::Literal(ScalarValue::List(Arc::new(
            arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                vec![Some(vec![Some(1.0); 128])],
            ),
        )));

        let knn_node = KnnNode::try_new(
            scan,
            "vector".to_string(),
            query_vector_expr,
            10,
            false,
            DistanceType::Cosine,
        )
        .unwrap();

        let display = format!("{}", knn_node);
        assert_eq!(
            display,
            "KnnSearch: column=vector, k=10, metric=cosine, use_index=false"
        );
    }
}
