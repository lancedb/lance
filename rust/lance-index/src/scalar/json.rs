// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    ops::Bound,
    sync::{Arc, Mutex},
};

use arrow_array::{Array, LargeBinaryArray, RecordBatch, StructArray, UInt8Array};
use arrow_schema::{DataType, Field, Field as ArrowField, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::{
    execution::SendableRecordBatchStream,
    physical_plan::{projection::ProjectionExec, ExecutionPlan},
};
use datafusion_common::{config::ConfigOptions, ScalarValue};
use datafusion_expr::{Expr, Operator, ScalarUDF};
use datafusion_physical_expr::{
    expressions::{Column, Literal},
    PhysicalExpr, ScalarFunctionExpr,
};
use deepsize::DeepSizeOf;
use futures::StreamExt;
use lance_datafusion::exec::{get_session_context, LanceExecutionOptions, OneShotExec};
use lance_datafusion::udf::json::JsonbType;
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

    fn derive_index_params(&self) -> Result<super::ScalarIndexParams> {
        self.target_index.derive_index_params()
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

    // TODO: maybe we should address it by https://github.com/lancedb/lance/issues/4624
    fn is_valid_reference(&self, func: &Expr, _data_type: &DataType) -> Option<DataType> {
        match func {
            Expr::ScalarFunction(udf) => {
                // Support multiple JSON extraction functions
                let json_functions = [
                    "json_extract",
                    "json_get",
                    "json_get_int",
                    "json_get_float",
                    "json_get_bool",
                    "json_get_string",
                ];
                if !json_functions.contains(&udf.name()) {
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
                            // Return the appropriate type based on the function
                            match udf.name() {
                                "json_get_int" => Some(DataType::Int64),
                                "json_get_float" => Some(DataType::Float64),
                                "json_get_bool" => Some(DataType::Boolean),
                                "json_get_string" | "json_extract" => Some(DataType::Utf8),
                                _ => None,
                            }
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

    /// Extract JSON with type information using the new UDF
    async fn extract_json_with_type_info(
        data: SendableRecordBatchStream,
        path: String,
    ) -> Result<(SendableRecordBatchStream, DataType)> {
        let input = Arc::new(OneShotExec::new(data));
        let input_schema = input.schema();
        let value_column_idx = input_schema
            .column_with_name(VALUE_COLUMN_NAME)
            .expect_ok()?
            .0;
        let row_id_column_idx = input_schema.column_with_name(ROW_ID).expect_ok()?.0;

        // Call json_extract_with_type UDF
        let exprs = vec![
            (
                Arc::new(ScalarFunctionExpr::try_new(
                    Arc::new(lance_datafusion::udf::json::json_extract_with_type_udf()),
                    vec![
                        Arc::new(Column::new(VALUE_COLUMN_NAME, value_column_idx)),
                        Arc::new(Literal::new(ScalarValue::Utf8(Some(path)))),
                    ],
                    &input_schema,
                    Arc::new(ConfigOptions::default()),
                )?) as Arc<dyn PhysicalExpr>,
                "json_result".to_string(),
            ),
            (
                Arc::new(Column::new(ROW_ID, row_id_column_idx)) as Arc<dyn PhysicalExpr>,
                ROW_ID.to_string(),
            ),
        ];

        let project = ProjectionExec::try_new(exprs, input)?;
        let ctx = get_session_context(&LanceExecutionOptions::default());
        let mut stream = project.execute(0, ctx.task_ctx())?;

        // Collect batches and determine type from first non-null value
        let mut all_batches = Vec::new();
        let mut inferred_type: Option<DataType> = None;

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;

            // Determine type from first non-null value if not yet set
            if inferred_type.is_none() {
                if let Some(json_result_column) = batch.column_by_name("json_result") {
                    if let Some(struct_array) =
                        json_result_column.as_any().downcast_ref::<StructArray>()
                    {
                        if let Some(type_array) = struct_array.column_by_name("type_tag") {
                            if let Some(uint8_array) =
                                type_array.as_any().downcast_ref::<UInt8Array>()
                            {
                                // Find first non-null value to determine type
                                for i in 0..uint8_array.len() {
                                    if !uint8_array.is_null(i) {
                                        let type_tag = uint8_array.value(i);
                                        let jsonb_type =
                                            JsonbType::from_u8(type_tag).ok_or_else(|| {
                                                Error::InvalidInput {
                                                    source: format!(
                                                        "Invalid type tag: {}",
                                                        type_tag
                                                    )
                                                    .into(),
                                                    location: location!(),
                                                }
                                            })?;

                                        // Map JsonbType to Arrow DataType
                                        inferred_type = Some(match jsonb_type {
                                            JsonbType::Null => continue, // Skip null values
                                            JsonbType::Boolean => DataType::Boolean,
                                            JsonbType::Int64 => DataType::Int64,
                                            JsonbType::Float64 => DataType::Float64,
                                            JsonbType::String => DataType::Utf8,
                                            JsonbType::Array => DataType::LargeBinary,
                                            JsonbType::Object => DataType::LargeBinary,
                                        });
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            all_batches.push(batch);
        }

        // If no type was inferred (all nulls), default to String
        let inferred_type = inferred_type.unwrap_or(DataType::Utf8);

        // Recreate stream from collected batches
        let schema =
            all_batches
                .first()
                .map(|b| b.schema())
                .ok_or_else(|| Error::InvalidInput {
                    source: "No batches in stream".into(),
                    location: location!(),
                })?;

        let recreated_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(all_batches.into_iter().map(Ok)),
        )) as SendableRecordBatchStream;

        Ok((recreated_stream, inferred_type))
    }

    /// Convert the stream with JSONB values and type tags to properly typed values
    async fn convert_stream_by_type(
        data: SendableRecordBatchStream,
        target_type: DataType,
    ) -> Result<SendableRecordBatchStream> {
        let input = Arc::new(OneShotExec::new(data));
        let _input_schema = input.schema();
        let ctx = get_session_context(&LanceExecutionOptions::default());
        let mut stream = input.execute(0, ctx.task_ctx())?;

        let mut converted_batches = Vec::new();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;

            // Extract the struct column containing value and type_tag
            let json_result_column =
                batch
                    .column_by_name("json_result")
                    .ok_or_else(|| Error::InvalidInput {
                        source: "Missing json_result column".into(),
                        location: location!(),
                    })?;

            let struct_array = json_result_column
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| Error::InvalidInput {
                    source: "json_result is not a struct".into(),
                    location: location!(),
                })?;

            let value_array =
                struct_array
                    .column_by_name("value")
                    .ok_or_else(|| Error::InvalidInput {
                        source: "Missing value column in struct".into(),
                        location: location!(),
                    })?;

            let binary_array = value_array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| Error::InvalidInput {
                    source: "value is not LargeBinary".into(),
                    location: location!(),
                })?;

            // Convert based on target type using serde deserialization
            let converted_array: Arc<dyn Array> = match target_type {
                DataType::Boolean => {
                    let mut builder =
                        arrow_array::builder::BooleanBuilder::with_capacity(binary_array.len());
                    for i in 0..binary_array.len() {
                        if binary_array.is_null(i) {
                            builder.append_null();
                        } else if let Some(bytes) = binary_array.value(i).into() {
                            let raw_jsonb = jsonb::RawJsonb::new(bytes);
                            // Try to deserialize directly to bool
                            match jsonb::from_raw_jsonb::<bool>(&raw_jsonb) {
                                Ok(bool_val) => builder.append_value(bool_val),
                                Err(e) => {
                                    return Err(Error::InvalidInput {
                                        source: format!(
                                            "Failed to deserialize JSONB to bool at index {}: {}",
                                            i, e
                                        )
                                        .into(),
                                        location: location!(),
                                    });
                                }
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                    Arc::new(builder.finish())
                }
                DataType::Int64 => {
                    let mut builder =
                        arrow_array::builder::Int64Builder::with_capacity(binary_array.len());
                    for i in 0..binary_array.len() {
                        if binary_array.is_null(i) {
                            builder.append_null();
                        } else if let Some(bytes) = binary_array.value(i).into() {
                            let raw_jsonb = jsonb::RawJsonb::new(bytes);
                            // Try to deserialize directly to i64
                            match jsonb::from_raw_jsonb::<i64>(&raw_jsonb) {
                                Ok(int_val) => builder.append_value(int_val),
                                Err(e) => {
                                    return Err(Error::InvalidInput {
                                        source: format!(
                                            "Failed to deserialize JSONB to i64 at index {}: {}",
                                            i, e
                                        )
                                        .into(),
                                        location: location!(),
                                    });
                                }
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                    Arc::new(builder.finish())
                }
                DataType::Float64 => {
                    let mut builder =
                        arrow_array::builder::Float64Builder::with_capacity(binary_array.len());
                    for i in 0..binary_array.len() {
                        if binary_array.is_null(i) {
                            builder.append_null();
                        } else if let Some(bytes) = binary_array.value(i).into() {
                            let raw_jsonb = jsonb::RawJsonb::new(bytes);
                            // Try to deserialize directly to f64 (serde handles int->float conversion)
                            match jsonb::from_raw_jsonb::<f64>(&raw_jsonb) {
                                Ok(float_val) => builder.append_value(float_val),
                                Err(e) => {
                                    return Err(Error::InvalidInput {
                                        source: format!(
                                            "Failed to deserialize JSONB to f64 at index {}: {}",
                                            i, e
                                        )
                                        .into(),
                                        location: location!(),
                                    });
                                }
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                    Arc::new(builder.finish())
                }
                DataType::Utf8 => {
                    let mut builder = arrow_array::builder::StringBuilder::with_capacity(
                        binary_array.len(),
                        1024,
                    );
                    for i in 0..binary_array.len() {
                        if binary_array.is_null(i) {
                            builder.append_null();
                        } else if let Some(bytes) = binary_array.value(i).into() {
                            let raw_jsonb = jsonb::RawJsonb::new(bytes);
                            // Try to deserialize to String, or use to_string() for any type
                            match jsonb::from_raw_jsonb::<String>(&raw_jsonb) {
                                Ok(str_val) => builder.append_value(&str_val),
                                Err(_) => {
                                    // For non-string types, convert to string representation
                                    builder.append_value(raw_jsonb.to_string());
                                }
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                    Arc::new(builder.finish())
                }
                DataType::LargeBinary => {
                    // Keep as binary for array/object types
                    value_array.clone()
                }
                _ => {
                    return Err(Error::InvalidInput {
                        source: format!("Unsupported target type: {:?}", target_type).into(),
                        location: location!(),
                    });
                }
            };

            // Get row_id column
            let row_id_column = batch
                .column_by_name(ROW_ID)
                .ok_or_else(|| Error::InvalidInput {
                    source: "Missing row_id column".into(),
                    location: location!(),
                })?
                .clone();

            // Create new batch with converted values
            let new_schema = Arc::new(Schema::new(vec![
                ArrowField::new(VALUE_COLUMN_NAME, target_type.clone(), true),
                ArrowField::new(ROW_ID, DataType::UInt64, false),
            ]));

            let new_batch =
                RecordBatch::try_new(new_schema.clone(), vec![converted_array, row_id_column])?;

            converted_batches.push(new_batch);
        }

        // Create stream from converted batches
        let schema = converted_batches
            .first()
            .map(|b| b.schema())
            .ok_or_else(|| Error::InvalidInput {
                source: "No batches to convert".into(),
                location: location!(),
            })?;

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(converted_batches.into_iter().map(Ok)),
        )))
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

        // Initially use Utf8, will be refined during training with type inference
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
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<JsonTrainingRequest>()
            .unwrap();
        let path = request.parameters.path.clone();

        // Extract JSON with type information
        let (data_stream, inferred_type) =
            Self::extract_json_with_type_info(data, path.clone()).await?;

        // Convert the stream to properly typed values based on inferred type
        let converted_stream =
            Self::convert_stream_by_type(data_stream, inferred_type.clone()).await?;

        // Update the target request with inferred type
        let registry = self.registry()?;
        let target_plugin = registry.get_plugin_by_name(&request.parameters.target_index_type)?;

        // Create a new training request with the inferred type
        let target_request = target_plugin.new_training_request(
            request
                .parameters
                .target_index_parameters
                .as_deref()
                .unwrap_or("{}"),
            &Field::new("", inferred_type, true),
        )?;

        let target_index = target_plugin
            .train_index(converted_stream, index_store, target_request, fragment_ids)
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
        cache: &LanceCache,
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    // Note: The old test_detect_json_value_type test has been removed as we now use
    // JSONB's inherent type information instead of string-based type detection

    #[tokio::test]
    async fn test_json_extract_with_type_info() {
        use arrow_array::{LargeBinaryArray, UInt64Array};
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use futures::stream;

        // Create test JSONB data
        let json_data = vec![
            r#"{"name": "Alice", "age": 30, "active": true}"#,
            r#"{"name": "Bob", "age": 25, "active": false}"#,
            r#"{"name": "Charlie", "age": 35, "active": true}"#,
        ];

        // Convert JSON strings to JSONB binary format
        let mut jsonb_values = Vec::new();
        for json_str in &json_data {
            let owned_jsonb: jsonb::OwnedJsonb = json_str.parse().unwrap();
            jsonb_values.push(Some(owned_jsonb.to_vec()));
        }

        // Create test batch with JSONB data
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));

        let jsonb_array = LargeBinaryArray::from(
            jsonb_values
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        );
        let row_ids = UInt64Array::from(vec![1, 2, 3]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(jsonb_array) as ArrayRef,
                Arc::new(row_ids) as ArrayRef,
            ],
        )
        .unwrap();

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch)]),
        )) as SendableRecordBatchStream;

        // Test type inference for integer field
        let (_result_stream, inferred_type) =
            JsonIndexPlugin::extract_json_with_type_info(stream, "$.age".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Int64);

        // Create new test stream for boolean field
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    json_data[0]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[1]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[2]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                ])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
        )
        .unwrap();

        let stream2 = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch2)]),
        )) as SendableRecordBatchStream;

        // Test type inference for boolean field
        let (_, inferred_type) =
            JsonIndexPlugin::extract_json_with_type_info(stream2, "$.active".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Boolean);

        // Create test stream for string field
        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    json_data[0]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[1]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[2]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                ])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
        )
        .unwrap();

        let stream3 = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch3)]),
        )) as SendableRecordBatchStream;

        // Test type inference for string field
        let (_, inferred_type) =
            JsonIndexPlugin::extract_json_with_type_info(stream3, "$.name".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Utf8);
    }
}
