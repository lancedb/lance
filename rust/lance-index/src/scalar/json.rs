// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    ops::Bound,
    sync::{Arc, Mutex},
};

use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::{
    execution::SendableRecordBatchStream,
    physical_plan::{projection::ProjectionExec, ExecutionPlan},
};
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Operator, ScalarUDF};
use datafusion_physical_expr::{
    expressions::{Column, Literal},
    PhysicalExpr, ScalarFunctionExpr,
};
use deepsize::DeepSizeOf;
use lance_datafusion::exec::{get_session_context, LanceExecutionOptions, OneShotExec};
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;

use lance_core::{cache::LanceCache, error::LanceOptionExt, Error, Result, ROW_ID};

use crate::{
    frag_reuse::FragReuseIndex,
    metrics::MetricsCollector,
    scalar::{
        expression::{IndexedExpression, ScalarIndexExpr, ScalarIndexSearch, ScalarQueryParser},
        registry::{
            ScalarIndexPlugin, ScalarIndexPluginRegistry, TrainingCriteria, TrainingRequest,
            VALUE_COLUMN_NAME,
        },
        AnyQuery, CreatedIndex, IndexStore, ScalarIndex, SearchResult, UpdateCriteria,
    },
    Index, IndexType,
};

const JSON_INDEX_VERSION: u32 = 0;

/// A JSON index that indexes a field in a JSON column
///
/// The underlying index can be any other type of scalar index
#[derive(Debug)]
pub struct JsonIndex {
    target_index: Arc<dyn ScalarIndex>,
    path: String,
}

impl JsonIndex {
    pub fn new(target_index: Arc<dyn ScalarIndex>, path: String) -> Self {
        Self { target_index, path }
    }
}

impl DeepSizeOf for JsonIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.target_index.deep_size_of_children(context) + self.path.deep_size_of_children(context)
    }
}

#[async_trait]
impl Index for JsonIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        unimplemented!()
    }

    fn index_type(&self) -> IndexType {
        // TODO: This causes the index to appear as btree in list_indices call.  Need better logic
        // in list_indices to use details instead of index_type.
        IndexType::Scalar
    }

    async fn prewarm(&self) -> Result<()> {
        self.target_index.prewarm().await
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        todo!()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        self.target_index.calculate_included_frags().await
    }
}

#[async_trait]
impl ScalarIndex for JsonIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<JsonQuery>().unwrap();
        self.target_index
            .search(query.target_query.as_ref(), metrics)
            .await
    }

    fn can_remap(&self) -> bool {
        self.target_index.can_remap()
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target_created = self.target_index.remap(mapping, dest_store).await?;
        let json_details = crate::pb::JsonIndexDetails {
            path: self.path.clone(),
            target_details: Some(target_created.index_details),
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&json_details)?,
            // TODO: We should store the target index version in the details
            index_version: JSON_INDEX_VERSION,
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target_created = self.target_index.update(new_data, dest_store).await?;
        let json_details = crate::pb::JsonIndexDetails {
            path: self.path.clone(),
            target_details: Some(target_created.index_details),
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&json_details)?,
            // TODO: We should store the target index version in the details
            index_version: JSON_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        self.target_index.update_criteria()
    }
}

/// Parameters for a [`JsonIndex`]
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonIndexParameters {
    target_index_type: String,
    target_index_parameters: Option<String>,
    path: String,
}

// TODO: Do we really need to wrap the query or could we just return the target query directly?
//
// I think the only thing we really gain is a different format impl (e.g. it shows up as a json query
// in the explain plan) but I don't know if that helps the user much.
#[derive(Debug, Clone)]
pub struct JsonQuery {
    target_query: Arc<dyn AnyQuery>,
    path: String,
}

impl JsonQuery {
    pub fn new(target_query: Arc<dyn AnyQuery>, path: String) -> Self {
        Self { target_query, path }
    }
}

impl PartialEq for JsonQuery {
    fn eq(&self, other: &Self) -> bool {
        self.target_query.dyn_eq(other.target_query.as_ref()) && self.path == other.path
    }
}

impl AnyQuery for JsonQuery {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn format(&self, col: &str) -> String {
        format!("Json({}->{})", self.target_query.format(col), self.path)
    }

    fn to_expr(&self, _col: String) -> Expr {
        todo!()
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

#[derive(Debug)]
pub struct JsonQueryParser {
    path: String,
    target_parser: Box<dyn ScalarQueryParser>,
}

impl JsonQueryParser {
    pub fn new(path: String, target_parser: Box<dyn ScalarQueryParser>) -> Self {
        Self {
            path,
            target_parser,
        }
    }

    fn wrap_search(&self, target_expr: IndexedExpression) -> IndexedExpression {
        if let Some(scalar_query) = target_expr.scalar_query {
            let scalar_query = match scalar_query {
                ScalarIndexExpr::Query(ScalarIndexSearch {
                    column,
                    index_name,
                    query,
                    needs_recheck,
                }) => ScalarIndexExpr::Query(ScalarIndexSearch {
                    column,
                    index_name,
                    query: Arc::new(JsonQuery::new(query, self.path.clone())),
                    needs_recheck,
                }),
                // This code path should only be hit on leaf expr
                _ => unreachable!(),
            };
            IndexedExpression {
                scalar_query: Some(scalar_query),
                refine_expr: target_expr.refine_expr,
            }
        } else {
            target_expr
        }
    }
}

impl ScalarQueryParser for JsonQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        self.target_parser
            .visit_between(column, low, high)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        self.target_parser
            .visit_in_list(column, in_list)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        self.target_parser
            .visit_is_bool(column, value)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        self.target_parser
            .visit_is_null(column)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        self.target_parser
            .visit_comparison(column, value, op)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        self.target_parser
            .visit_scalar_function(column, data_type, func, args)
            .map(|target_expr| self.wrap_search(target_expr))
    }

    fn is_valid_reference(&self, func: &Expr, _data_type: &DataType) -> Option<DataType> {
        match func {
            Expr::ScalarFunction(udf) => {
                if udf.name() != "json_extract" {
                    return None;
                }
                if udf.args.len() != 2 {
                    return None;
                }
                // We already know index 0 is a column reference to the column so we just need to
                // ensure that index 1 matches our path
                match &udf.args[1] {
                    Expr::Literal(ScalarValue::Utf8(Some(path)), _) => {
                        if path == &self.path {
                            // TODO: This may need to be flexible
                            Some(DataType::Utf8)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

pub struct JsonTrainingRequest {
    parameters: JsonIndexParameters,
    target_request: Box<dyn TrainingRequest>,
}

impl JsonTrainingRequest {
    pub fn new(parameters: JsonIndexParameters, target_request: Box<dyn TrainingRequest>) -> Self {
        Self {
            parameters,
            target_request,
        }
    }
}

impl TrainingRequest for JsonTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        self.target_request.criteria()
    }
}

/// Plugin implementation for a [`JsonIndex`]
#[derive(Default)]
pub struct JsonIndexPlugin {
    registry: Mutex<Option<Arc<ScalarIndexPluginRegistry>>>,
}

impl std::fmt::Debug for JsonIndexPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JsonIndexPlugin")
    }
}

impl JsonIndexPlugin {
    fn registry(&self) -> Result<Arc<ScalarIndexPluginRegistry>> {
        Ok(self.registry.lock().unwrap().as_ref().expect_ok()?.clone())
    }

    fn extract_json(
        data: SendableRecordBatchStream,
        path: String,
    ) -> Result<SendableRecordBatchStream> {
        let input = Arc::new(OneShotExec::new(data));
        let input_schema = input.schema();
        let value_column_idx = input_schema
            .column_with_name(VALUE_COLUMN_NAME)
            .expect_ok()?
            .0;
        // TODO: We should just copy over all non-value columns, not cherry-pick row id
        let row_id_column_idx = input_schema.column_with_name(ROW_ID).expect_ok()?.0;
        let exprs = vec![
            (
                Arc::new(ScalarFunctionExpr::try_new(
                    Arc::new(lance_datafusion::udf::json::json_extract_udf()),
                    vec![
                        Arc::new(Column::new(VALUE_COLUMN_NAME, value_column_idx)),
                        Arc::new(Literal::new(ScalarValue::Utf8(Some(path)))),
                    ],
                    &input_schema,
                )?) as Arc<dyn PhysicalExpr>,
                VALUE_COLUMN_NAME.to_string(),
            ),
            (
                Arc::new(Column::new(ROW_ID, row_id_column_idx)) as Arc<dyn PhysicalExpr>,
                ROW_ID.to_string(),
            ),
        ];
        let project = ProjectionExec::try_new(exprs, input)?;
        let ctx = get_session_context(&LanceExecutionOptions::default());
        project.execute(0, ctx.task_ctx()).map_err(Error::from)
    }
}

#[async_trait]
impl ScalarIndexPlugin for JsonIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if !matches!(field.data_type(), DataType::Binary | DataType::LargeBinary) {
            return Err(Error::InvalidInput {
                source: "A JSON index can only be created on a Binary or LargeBinary field.".into(),
                location: location!(),
            });
        }

        // TODO: How do we determine the target type?
        // TODO: How do we extract to a specific type?  Maybe try_cast?
        let target_type = DataType::Utf8;

        let params = serde_json::from_str::<JsonIndexParameters>(params)?;
        let registry = self.registry()?;
        let target_plugin = registry.get_plugin_by_name(&params.target_index_type)?;
        let target_request = target_plugin.new_training_request(
            params.target_index_parameters.as_deref().unwrap_or("{}"),
            &Field::new("", target_type, true),
        )?;

        Ok(Box::new(JsonTrainingRequest::new(params, target_request)))
    }

    fn provides_exact_answer(&self) -> bool {
        // TODO: Need to lookup target plugin via details to figure this out correctly
        true
    }

    fn attach_registry(&self, registry: Arc<ScalarIndexPluginRegistry>) {
        let mut reg_ref = self.registry.lock().unwrap();
        *reg_ref = Some(registry);
    }

    fn version(&self) -> u32 {
        JSON_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // TODO: Allow return Result here
        let registry = self.registry().unwrap();
        let json_details =
            crate::pb::JsonIndexDetails::decode(index_details.value.as_slice()).unwrap();
        let target_details = json_details.target_details.as_ref().expect_ok().unwrap();
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();
        // TODO: Use something like ${index_name}_${path} for the index name?  Don't have access to path here tho
        let target_parser = target_plugin.new_query_parser(index_name, index_details)?;
        Some(Box::new(JsonQueryParser::new(
            json_details.path.clone(),
            target_parser,
        )) as Box<dyn ScalarQueryParser>)
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<JsonTrainingRequest>()
            .unwrap();
        let path = request.parameters.path.clone();
        let registry = self.registry()?;
        let target_plugin = registry.get_plugin_by_name(&request.parameters.target_index_type)?;
        let data = Self::extract_json(data, path.clone())?;
        let target_index = target_plugin
            .train_index(data, index_store, request.target_request)
            .await?;

        let index_details = crate::pb::JsonIndexDetails {
            path,
            target_details: Some(target_index.index_details),
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&index_details)?,
            index_version: JSON_INDEX_VERSION,
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let registry = self.registry().unwrap();
        let json_details = crate::pb::JsonIndexDetails::decode(index_details.value.as_slice())?;
        let target_details = json_details.target_details.as_ref().expect_ok()?;
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();
        let target_index = target_plugin
            .load_index(index_store, target_details, frag_reuse_index, cache)
            .await?;
        Ok(Arc::new(JsonIndex::new(target_index, json_details.path)))
    }
}
