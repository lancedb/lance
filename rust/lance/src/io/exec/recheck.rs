//use std::sync::Arc;
//
//use arrow_schema::SchemaRef;
//use datafusion::{
//    common::Statistics,
//    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties},
//};
//use lance_index::scalar::expression::ScalarIndexExpr;
//
//use crate::Dataset;
//
//use super::scalar_index::MATERIALIZE_INDEX_SCHEMA;
//
//#[derive(Debug)]
//pub struct RecheckExec {
//    dataset: Arc<Dataset>,
//    expr: ScalarIndexExpr,
//    properties: PlanProperties,
//    input: Arc<dyn ExecutionPlan>,
//}
//
//impl DisplayAs for RecheckExec {
//    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//        match t {
//            DisplayFormatType::Default | DisplayFormatType::Verbose => {
//                write!(f, "RecheckExec: query={}", self.expr)
//            }
//        }
//    }
//}
//
//impl RecheckExec {
//    pub fn new(
//        dataset: Arc<Dataset>,
//        input: Arc<dyn ExecutionPlan>,
//        expr: ScalarIndexExpr,
//    ) -> Self {
//        // todo is that right
//        let properties = input.properties().clone();
//        Self {
//            dataset,
//            expr,
//            properties,
//            input,
//        }
//    }
//}
//
//impl ExecutionPlan for RecheckExec {
//    fn name(&self) -> &str {
//        "RecheckExec"
//    }
//
//    fn as_any(&self) -> &dyn std::any::Any {
//        self
//    }
//
//    fn schema(&self) -> SchemaRef {
//        MATERIALIZE_INDEX_SCHEMA.clone()
//    }
//
//    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
//        vec![&self.input]
//    }
//
//    fn with_new_children(
//        self: Arc<Self>,
//        _children: Vec<Arc<dyn ExecutionPlan>>,
//    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
//        if _children.len() != 1 {
//            return Err(DataFusionError::Internal(
//                "TakeExec wrong number of children".to_string(),
//            ));
//        }
//
//        let projection = self
//            .dataset
//            .empty_projection()
//            .union_schema(&self.original_projection);
//
//        let plan = Self::try_new(
//            self.dataset.clone(),
//            _children[0].clone(),
//            projection,
//            self.batch_readahead,
//        )?;
//
//        if let Some(plan) = plan {
//            Ok(Arc::new(plan))
//        } else {
//            // Is this legal or do we need to insert a no-op node?
//            Ok(_children[0].clone())
//        }
//    }
//
//    fn execute(
//        &self,
//        _partition: usize,
//        _context: Arc<datafusion::execution::context::TaskContext>,
//    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
//        todo!()
//    }
//
//    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
//        Ok(Statistics::new_unknown(&MATERIALIZE_INDEX_SCHEMA))
//    }
//
//    fn properties(&self) -> &PlanProperties {
//        &self.properties
//    }
//}
//
//impl RecheckExec {}
//
