// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::StringArray;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::DisplayAs;
use datafusion::physical_plan::DisplayFormatType;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::PlanProperties;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::PhysicalExpr;
use lance_core::ROW_ID;
use lance_core::ROW_ID_FIELD;

use super::MergeInsertParams;
use super::Result;
use super::WhenMatched;
use super::WhenNotMatchedBySource;

// const UPDATED_FIELD: LazyLock<Arc<Field>> = LazyLock::new(|| {
//     Arc::new(Field::new("__update", DataType::Boolean, false))
// });
// const DELETED_FIELD: LazyLock<Arc<Field>> = LazyLock::new(|| {
//     Arc::new(Field::new("__delete", DataType::Boolean, false))
// });

// const UPDATED_FIELD: LazyLock<Arc<Field>> = LazyLock::new(|| {
//     Arc::new(Field::new("__update", DataType::Boolean, false))
// });
const ACTION_FIELD: LazyLock<Arc<Field>> = LazyLock::new(|| {
    Arc::new(Field::new(
        "__action",
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        false,
    ))
});

#[repr(i8)]
enum Action {
    Nothing = 0,
    Update = 1,
    Insert = 2,
    Delete = 3,
}

const ACTION_DICT: LazyLock<Arc<StringArray>> = LazyLock::new(|| {
    let keys = vec!["nothing", "update", "insert", "delete"];
    Arc::new(StringArray::from(keys))
});

/// Performs the core logic of the merge insert operation.
///
/// This expects an input schema of:
/// - target: struct
/// - source: struct
/// - _rowid: u64
/// - _rowaddr (optional) : u64
///
/// It does two things to the output:
/// 1. Computes an `__action` column, which is a dictionary column with the following values:
///    - 0: nothing
///    - 1: update
///    - 2: insert
///    - 3: delete
/// 2. Transforms the schema by:
///     a. Flattening the target and source structs to the final data
///     b. Adding the two new columns
///
/// The output schema is:
/// - target field 1
/// - ...
/// - target field n
/// - _rowid: u64
/// - _rowaddr (optional) : u64
/// - __action: dictionary<i8, utf8>
#[derive(Debug)]
struct MergeInsertComputeExec {
    input: Arc<dyn ExecutionPlan>,
    params: MergeInsertParams,
    properties: datafusion::physical_plan::PlanProperties,

    output_schema: Arc<Schema>,
    action_expr: Arc<dyn PhysicalExpr>,

    target_pos: usize,
    source_pos: usize,
    row_id_pos: usize,
    row_addr_pos: Option<usize>,
}

impl MergeInsertComputeExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>, params: MergeInsertParams) -> Result<Self> {
        let input_schema = input.schema();
        let target_pos = input_schema
            .fields()
            .iter()
            .position(|f| f.name() == "target" && matches!(f.data_type(), DataType::Struct(_)))
            .expect("target field not found");
        let source_pos = input_schema
            .fields()
            .iter()
            .position(|f| f.name() == "source" && matches!(f.data_type(), DataType::Struct(_)))
            .expect("target field not found");
        let row_id_pos = input_schema
            .fields()
            .iter()
            .position(|f| f.name() == ROW_ID && matches!(f.data_type(), DataType::UInt64))
            .expect("rowid field not found");
        let row_addr_pos = input_schema
            .fields()
            .iter()
            .position(|f| f.name() == "_rowaddr" && matches!(f.data_type(), DataType::UInt64));

        let output_schema = Self::compute_output_schema(
            input_schema.as_ref(),
            target_pos,
            source_pos,
            row_id_pos,
            row_addr_pos,
        );

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            input.properties().partitioning.clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        let action_expr = Self::action_expr();

        Ok(Self {
            input,
            params,
            properties,

            output_schema,
            action_expr,

            target_pos,
            source_pos,
            row_id_pos,
            row_addr_pos,
        })
    }

    fn compute_output_schema(
        input_schema: &Schema,
        target_pos: usize,
        source_pos: usize,
        row_id_pos: usize,
        row_addr_pos: Option<usize>,
    ) -> SchemaRef {
        // Which fields do we output?
        let target_schema = input_schema.field(target_pos).data_type();

        let target_fields = if let DataType::Struct(target_fields) = target_schema {
            target_fields
        } else {
            panic!("target field is not a struct")
        };

        let num_meta_fields = if row_addr_pos.is_some() { 4 } else { 3 };
        let mut fields = Vec::with_capacity(target_fields.len() + num_meta_fields);

        for field in target_fields {
            fields.push(field.clone());
        }
        fields.push(Arc::new(ROW_ID_FIELD.clone()));
        if let Some(row_addr_pos) = row_addr_pos {
            fields.push(Arc::new(input_schema.field(row_addr_pos).clone()));
        }
        fields.push((*UPDATED_FIELD).clone());
        fields.push((*DELETED_FIELD).clone());

        Arc::new(Schema::new(fields))
    }

    fn action_expr() -> Arc<dyn PhysicalExpr> {
        todo!()
    }
}

impl DisplayAs for MergeInsertComputeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MergeInsertComputeExec: on_match=")?;
        match &self.params.when_matched {
            WhenMatched::UpdateAll => write!(f, "on_match=update_all, ")?,
            WhenMatched::UpdateIf(expr) => {
                let mut expr_str = format!("{}", expr);
                if !matches!(t, DisplayFormatType::Verbose) && expr_str.len() > 50 {
                    expr_str.truncate(50);
                    expr_str.push_str("...");
                }
                write!(f, "on_match=update_all_if({}), ", expr_str)?
            }
            WhenMatched::DoNothing => {}
        }

        if self.params.insert_not_matched {
            write!(f, "on_not_matched=insert")?;
        }

        match &self.params.delete_not_matched_by_source {
            WhenNotMatchedBySource::DeleteIf(expr) => {
                let mut expr_str = format!("{}", expr);
                if !matches!(t, DisplayFormatType::Verbose) && expr_str.len() > 50 {
                    expr_str.truncate(50);
                    expr_str.push_str("...");
                }
                write!(f, ", on_not_matched_by_source=delete_if({})", expr_str)?;
            }
            WhenNotMatchedBySource::Delete => write!(f, ", on_not_matched_by_source=delete_all")?,
            WhenNotMatchedBySource::Keep => {}
        }

        Ok(())
    }
}

impl ExecutionPlan for MergeInsertComputeExec {
    fn name(&self) -> &str {
        todo!()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        todo!()
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.properties
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "MergeInsertComputeExec wrong number of children".to_string(),
            ));
        }

        Ok(Arc::new(
            MergeInsertComputeExec::try_new(children[0].clone(), self.params.clone())
                .map_err(|e| datafusion::error::DataFusionError::Internal(e.to_string()))?,
        ))
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn cardinality_effect(&self) -> datafusion::physical_plan::execution_plan::CardinalityEffect {
        datafusion::physical_plan::execution_plan::CardinalityEffect::Equal
    }
}
