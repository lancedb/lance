// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::sync::Arc;

use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::prelude::Column;
use datafusion::scalar::ScalarValue;
use datafusion_expr::expr::InList;
use datafusion_expr::{Between, BinaryExpr, Expr, Operator};
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_index::metrics::MetricsCollector;
use lance_index::progress::IndexBuildProgress;
use lance_index::registry::IndexPluginRegistry;
use lance_index::scalar::expression::{
    IndexedExpression, ScalarIndexExpr, ScalarIndexSearch, ScalarQueryParser,
};
use lance_index::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use lance_index::scalar::{
    AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, ScalarIndex, ScalarIndexParams,
    SearchResult, UpdateCriteria,
};
use lance_index::{Index, IndexType};
use lance_index_plugin_abi::{
    ABI_VERSION, EXACTNESS_AT_LEAST, EXACTNESS_AT_MOST, EXACTNESS_EXACT,
    LanceIndexBuildProgressAbi, LanceIndexStoreAbi, LanceTrainingDataAbi, ORDERING_ADDRESSES,
    ORDERING_NONE, ORDERING_VALUES, OwnedLanceIndexBuildProgressAbi, OwnedLanceIndexStoreAbi,
    OwnedLanceTrainingDataAbi, QueryPlanAbi, SubstraitQueryAbi, TrainInputAbi,
    UPDATE_ONLY_NEW_DATA, UPDATE_REQUIRES_OLD_DATA, XabiLanceScalarIndexHandle,
    XabiLanceScalarIndexPluginHandle, ipc_to_record_batches, le_bytes_to_u32s, le_bytes_to_u64s,
    record_batch_to_ipc, record_batches_to_ipc, u32s_to_le_bytes, u64s_to_le_bytes,
};
use lance_select::RowAddrTreeMap;
use prost_types::Any as ProstAny;
use roaring::RoaringBitmap;
use serde_json::json;

use tokio::sync::Mutex;

#[derive(Debug)]
pub struct XabiScalarIndexPlugin {
    handle: Arc<XabiLanceScalarIndexPluginHandle>,
    name: String,
    version: u32,
    provides_exact_answer: bool,
}

impl XabiScalarIndexPlugin {
    fn new(
        handle: XabiLanceScalarIndexPluginHandle,
        name: String,
        version: u32,
        provides_exact_answer: bool,
    ) -> Self {
        Self {
            handle: Arc::new(handle),
            name,
            version,
            provides_exact_answer,
        }
    }
}

struct XabiTrainingRequest {
    criteria: TrainingCriteria,
}

impl TrainingRequest for XabiTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// Register every Lance scalar-index export from a trusted xabi module.
///
/// # Safety
///
/// `path` must point to trusted native code that follows the Lance index plugin
/// ABI. Loading an arbitrary dynamic library runs code in the current process.
pub unsafe fn register_xabi_scalar_index_plugin_library(
    registry: Arc<IndexPluginRegistry>,
    path: impl AsRef<Path>,
) -> Result<Vec<String>> {
    let module = unsafe { xabi::Module::load(path) }.map_err(xabi_to_lance)?;
    let mut registered = Vec::new();
    for (_export_name, handle) in
        XabiLanceScalarIndexPluginHandle::xabi_load_all(&module).map_err(xabi_to_lance)?
    {
        let name = handle.name().map_err(xabi_to_lance)?;
        let version = handle.version().map_err(xabi_to_lance)?;
        let provides_exact_answer = handle.provides_exact_answer().map_err(xabi_to_lance)?;
        let details_type_url = handle.details_type_url().map_err(xabi_to_lance)?;
        let plugin = Arc::new(XabiScalarIndexPlugin::new(
            handle,
            name.clone(),
            version,
            provides_exact_answer,
        ));
        plugin.attach_registry(registry.clone());
        registry.add_runtime_plugin(name.clone(), plugin.clone())?;
        registry.add_runtime_plugin_for_details(&details_type_url, plugin)?;
        registered.push(name);
    }
    Ok(registered)
}

fn xabi_to_lance(err: impl std::fmt::Display) -> Error {
    Error::index(format!("xabi index plugin error: {err}"))
}

fn block_on_xabi<F: std::future::Future>(future: F) -> F::Output {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
}

fn training_ordering_from_abi(ordering: u32) -> Result<TrainingOrdering> {
    match ordering {
        ORDERING_NONE => Ok(TrainingOrdering::None),
        ORDERING_VALUES => Ok(TrainingOrdering::Values),
        ORDERING_ADDRESSES => Ok(TrainingOrdering::Addresses),
        other => Err(Error::invalid_input(format!(
            "unknown xabi training ordering {other}"
        ))),
    }
}

fn training_criteria_from_abi(
    criteria: lance_index_plugin_abi::TrainingCriteriaAbi,
) -> Result<TrainingCriteria> {
    let ordering = training_ordering_from_abi(criteria.ordering)?;
    let needs_row_ids = criteria.needs_row_ids != 0;
    let needs_row_addrs = criteria.needs_row_addrs != 0;
    let mut criteria = TrainingCriteria::new(ordering);
    if needs_row_ids {
        criteria = criteria.with_row_id();
    }
    if needs_row_addrs {
        criteria = criteria.with_row_addr();
    }
    Ok(criteria)
}

fn update_criteria_from_abi(
    criteria: lance_index_plugin_abi::UpdateCriteriaAbi,
) -> Result<UpdateCriteria> {
    let data_criteria = training_criteria_from_abi(criteria.data_criteria)?;
    match criteria.mode {
        UPDATE_REQUIRES_OLD_DATA => Ok(UpdateCriteria::requires_old_data(data_criteria)),
        UPDATE_ONLY_NEW_DATA => Ok(UpdateCriteria::only_new_data(data_criteria)),
        other => Err(Error::invalid_input(format!(
            "unknown xabi update criteria mode {other}"
        ))),
    }
}

fn created_index_from_abi(created: lance_index_plugin_abi::CreatedIndexAbi) -> CreatedIndex {
    CreatedIndex {
        index_details: ProstAny {
            type_url: created.details_type_url,
            value: created.details,
        },
        index_version: created.index_version,
        files: Vec::new(),
    }
}

fn row_addr_map(row_ids: Vec<u64>) -> RowAddrTreeMap {
    let mut map = RowAddrTreeMap::new();
    for row_id in row_ids {
        map.insert(row_id);
    }
    map
}

fn search_result_from_abi(output: lance_index_plugin_abi::SearchOutputAbi) -> Result<SearchResult> {
    let row_ids = row_addr_map(le_bytes_to_u64s(&output.row_ids_le).map_err(xabi_to_lance)?);
    let null_row_ids =
        row_addr_map(le_bytes_to_u64s(&output.null_row_ids_le).map_err(xabi_to_lance)?);
    let result = match output.exactness {
        EXACTNESS_EXACT => SearchResult::exact(row_ids),
        EXACTNESS_AT_MOST => SearchResult::at_most(row_ids),
        EXACTNESS_AT_LEAST => SearchResult::at_least(row_ids),
        other => {
            return Err(Error::invalid_input(format!(
                "unknown xabi search exactness {other}"
            )));
        }
    };
    Ok(result.with_nulls(null_row_ids))
}

#[async_trait]
impl ScalarIndexPlugin for XabiScalarIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        let field_json = format!("{field:?}");
        let plan = self
            .handle
            .new_training_plan(params, &field_json)
            .map_err(xabi_to_lance)?;
        Ok(Box::new(XabiTrainingRequest {
            criteria: training_criteria_from_abi(plan.criteria)?,
        }))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let data = OwnedLanceTrainingDataAbi::new(HostTrainingData::new(data));
        let store = OwnedLanceIndexStoreAbi::new(HostIndexStore::new(index_store.clone_arc()));
        let progress = OwnedLanceIndexBuildProgressAbi::new(HostIndexBuildProgress::new(progress));
        let input = TrainInputAbi::new(
            "{}".to_string(),
            u32s_to_le_bytes(&fragment_ids.unwrap_or_default()),
            data.xabi_borrow(),
            store.xabi_borrow(),
            progress.xabi_borrow(),
        );
        let created = block_on_xabi(self.handle.train_index(input)).map_err(xabi_to_lance)?;
        let mut created = created_index_from_abi(created);
        created.files = index_store.list_files_with_sizes().await?;
        let _ = request;
        Ok(created)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn provides_exact_answer(&self) -> bool {
        self.provides_exact_answer
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &ProstAny,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(XabiScalarQueryParser {
            handle: self.handle.clone(),
            index_name,
            index_type: self.name.clone(),
            details: index_details.value.clone(),
        }))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &ProstAny,
        _frag_reuse_index: Option<Arc<lance_index::frag_reuse::FragReuseIndex>>,
        _cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let store = OwnedLanceIndexStoreAbi::new(HostIndexStore::new(index_store));
        let handle = block_on_xabi(
            self.handle
                .load_index(&index_details.value, store.xabi_borrow()),
        )
        .map_err(xabi_to_lance)?;
        let can_remap = handle.can_remap().map_err(xabi_to_lance)?;
        Ok(Arc::new(XabiScalarIndex {
            handle: Arc::new(handle),
            index_type: self.name.clone(),
            can_remap,
        }))
    }

    async fn load_statistics(
        &self,
        _index_store: Arc<dyn IndexStore>,
        index_details: &ProstAny,
    ) -> Result<Option<serde_json::Value>> {
        block_on_xabi(self.handle.load_statistics(&index_details.value))
            .map_err(xabi_to_lance)?
            .map(|json| serde_json::from_str(&json).map_err(Error::from))
            .transpose()
    }

    fn supports_load_statistics(&self) -> bool {
        true
    }

    fn details_as_json(&self, details: &ProstAny) -> Result<serde_json::Value> {
        let json = self
            .handle
            .details_as_json(&details.value)
            .map_err(xabi_to_lance)?;
        serde_json::from_str(&json).map_err(Error::from)
    }
}

#[derive(Debug)]
struct XabiScalarQuery {
    query: Vec<u8>,
    display: String,
}

impl AnyQuery for XabiScalarQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, _col: &str) -> String {
        self.display.clone()
    }

    fn to_expr(&self, _col: String) -> Expr {
        Expr::Literal(ScalarValue::Boolean(Some(true)), None)
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.query == other.query)
    }
}

struct XabiScalarQueryParser {
    handle: Arc<XabiLanceScalarIndexPluginHandle>,
    index_name: String,
    index_type: String,
    details: Vec<u8>,
}

fn bound_value(bound: &std::ops::Bound<ScalarValue>) -> Option<&ScalarValue> {
    match bound {
        std::ops::Bound::Included(value) | std::ops::Bound::Excluded(value) => Some(value),
        std::ops::Bound::Unbounded => None,
    }
}

impl Debug for XabiScalarQueryParser {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XabiScalarQueryParser")
            .field("index_name", &self.index_name)
            .field("index_type", &self.index_type)
            .finish()
    }
}

impl XabiScalarQueryParser {
    fn column_expr(column: &str) -> Expr {
        Expr::Column(Column::from_name(column))
    }

    #[cfg(feature = "substrait")]
    fn encode_substrait(expr: Expr, schema: arrow_schema::Schema) -> (Vec<u8>, Vec<u8>) {
        let schema = Arc::new(schema);
        let schema_ipc = record_batches_to_ipc(schema.clone(), &[]).unwrap_or_default();
        let state = datafusion::execution::context::SessionContext::new().state();
        let expr_bytes =
            lance_datafusion::substrait::encode_substrait(expr, schema, &state).unwrap_or_default();
        (schema_ipc, expr_bytes)
    }

    #[cfg(not(feature = "substrait"))]
    fn encode_substrait(_expr: Expr, _schema: arrow_schema::Schema) -> (Vec<u8>, Vec<u8>) {
        (Vec::new(), Vec::new())
    }

    fn plan(
        &self,
        column: &str,
        host_query_json: serde_json::Value,
        substrait: Option<(Expr, arrow_schema::Schema)>,
    ) -> Option<IndexedExpression> {
        let (schema_ipc, expr_bytes) = substrait
            .map(|(expr, schema)| Self::encode_substrait(expr, schema))
            .unwrap_or_default();
        let query = SubstraitQueryAbi::new(
            ABI_VERSION.to_string(),
            schema_ipc,
            expr_bytes,
            host_query_json.to_string(),
        );
        let QueryPlanAbi {
            accepted: _,
            query,
            needs_recheck,
        } = self
            .handle
            .plan_query(&self.details, query)
            .ok()
            .filter(|plan| plan.accepted != 0)?;
        Some(IndexedExpression {
            scalar_query: Some(ScalarIndexExpr::Query(ScalarIndexSearch {
                column: column.to_string(),
                index_name: self.index_name.clone(),
                index_type: self.index_type.clone(),
                query: Arc::new(XabiScalarQuery {
                    query,
                    display: host_query_json.to_string(),
                }),
                needs_recheck: needs_recheck != 0,
                fragment_bitmap: None,
            })),
            refine_expr: None,
        })
    }
}

impl ScalarQueryParser for XabiScalarQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: &std::ops::Bound<ScalarValue>,
        high: &std::ops::Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        self.plan(
            column,
            json!({
                "op": "between",
                "low": format!("{low:?}"),
                "high": format!("{high:?}"),
            }),
            match (bound_value(low), bound_value(high)) {
                (Some(low), Some(high)) => Some((
                    Expr::Between(Between::new(
                        Box::new(Self::column_expr(column)),
                        false,
                        Box::new(Expr::Literal(low.clone(), None)),
                        Box::new(Expr::Literal(high.clone(), None)),
                    )),
                    arrow_schema::Schema::new(vec![Field::new(column, low.data_type(), true)]),
                )),
                _ => None,
            },
        )
    }

    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        let substrait = in_list.first().map(|value| {
            (
                Expr::InList(InList::new(
                    Box::new(Self::column_expr(column)),
                    in_list
                        .iter()
                        .cloned()
                        .map(|value| Expr::Literal(value, None))
                        .collect(),
                    false,
                )),
                arrow_schema::Schema::new(vec![Field::new(column, value.data_type(), true)]),
            )
        });
        self.plan(
            column,
            json!({
                "op": "in",
                "values": in_list.iter().map(|v| v.to_string()).collect::<Vec<_>>(),
            }),
            substrait,
        )
    }

    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        self.plan(
            column,
            json!({"op": "eq", "value": value}),
            Some((
                Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(Self::column_expr(column)),
                    Operator::Eq,
                    Box::new(Expr::Literal(ScalarValue::Boolean(Some(value)), None)),
                )),
                arrow_schema::Schema::new(vec![Field::new(
                    column,
                    arrow_schema::DataType::Boolean,
                    true,
                )]),
            )),
        )
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        self.plan(
            column,
            json!({"op": "is_null"}),
            Some((
                Expr::IsNull(Box::new(Self::column_expr(column))),
                arrow_schema::Schema::new(vec![Field::new(
                    column,
                    arrow_schema::DataType::Null,
                    true,
                )]),
            )),
        )
    }

    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        self.plan(
            column,
            json!({
                "op": op.to_string(),
                "value": value.to_string(),
            }),
            Some((
                Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(Self::column_expr(column)),
                    *op,
                    Box::new(Expr::Literal(value.clone(), None)),
                )),
                arrow_schema::Schema::new(vec![Field::new(column, value.data_type(), true)]),
            )),
        )
    }

    fn visit_scalar_function(
        &self,
        _column: &str,
        _data_type: &arrow_schema::DataType,
        _func: &datafusion_expr::ScalarUDF,
        _args: &[Expr],
    ) -> Option<IndexedExpression> {
        None
    }
}

#[derive(Debug)]
struct XabiScalarIndex {
    handle: Arc<XabiLanceScalarIndexHandle>,
    index_type: String,
    can_remap: bool,
}

impl DeepSizeOf for XabiScalarIndex {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[async_trait]
impl Index for XabiScalarIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn lance_index::vector::VectorIndex>> {
        Err(Error::invalid_input(
            "xabi scalar index cannot be used as a vector index",
        ))
    }

    async fn prewarm(&self) -> Result<()> {
        block_on_xabi(self.handle.prewarm()).map_err(xabi_to_lance)
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let json = self.handle.statistics().map_err(xabi_to_lance)?;
        serde_json::from_str(&json).map_err(Error::from)
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let bytes = self.handle.calculate_included_frags();
        let bytes = block_on_xabi(bytes).map_err(xabi_to_lance)?;
        Ok(le_bytes_to_u32s(&bytes)
            .map_err(xabi_to_lance)?
            .into_iter()
            .collect())
    }
}

#[async_trait]
impl ScalarIndex for XabiScalarIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query
            .as_any()
            .downcast_ref::<XabiScalarQuery>()
            .ok_or_else(|| {
                Error::invalid_input("xabi scalar index received an incompatible query")
            })?;
        let output = block_on_xabi(self.handle.search(&query.query)).map_err(xabi_to_lance)?;
        search_result_from_abi(output)
    }

    fn can_remap(&self) -> bool {
        self.can_remap
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let mappings = mapping.keys().copied().collect::<Vec<_>>();
        let mapped_row_ids = mapping
            .values()
            .map(|value| value.unwrap_or_default())
            .collect::<Vec<_>>();
        let has_mapped_row_id = mapping
            .values()
            .map(|value| u8::from(value.is_some()))
            .collect::<Vec<_>>();
        let store = OwnedLanceIndexStoreAbi::new(HostIndexStore::new(dest_store.clone_arc()));
        let input = lance_index_plugin_abi::RemapInputAbi::new(
            u64s_to_le_bytes(&mappings),
            u64s_to_le_bytes(&mapped_row_ids),
            has_mapped_row_id,
            store.xabi_borrow(),
        );
        let created = block_on_xabi(self.handle.remap(input)).map_err(xabi_to_lance)?;
        let mut created = created_index_from_abi(created);
        created.files = dest_store.list_files_with_sizes().await?;
        Ok(created)
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        _old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let data = OwnedLanceTrainingDataAbi::new(HostTrainingData::new(new_data));
        let store = OwnedLanceIndexStoreAbi::new(HostIndexStore::new(dest_store.clone_arc()));
        let input =
            lance_index_plugin_abi::UpdateInputAbi::new(data.xabi_borrow(), store.xabi_borrow());
        let created = block_on_xabi(self.handle.update(input)).map_err(xabi_to_lance)?;
        let mut created = created_index_from_abi(created);
        created.files = dest_store.list_files_with_sizes().await?;
        Ok(created)
    }

    fn update_criteria(&self) -> UpdateCriteria {
        self.handle
            .update_criteria()
            .map_err(xabi_to_lance)
            .and_then(update_criteria_from_abi)
            .unwrap_or_else(|_| {
                UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None))
            })
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params_json = self
            .handle
            .derive_index_params_json()
            .map_err(xabi_to_lance)?;
        let value = serde_json::from_str::<serde_json::Value>(&params_json)?;
        let mut params = ScalarIndexParams::new(
            value
                .get("index_type")
                .and_then(|value| value.as_str())
                .unwrap_or(&self.index_type)
                .to_string(),
        );
        if let Some(plugin_params) = value.get("params") {
            params.params = Some(plugin_params.to_string());
        }
        Ok(params)
    }
}

struct HostTrainingData {
    inner: Mutex<SendableRecordBatchStream>,
}

impl HostTrainingData {
    fn new(inner: SendableRecordBatchStream) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

impl LanceTrainingDataAbi for HostTrainingData {
    async fn next_batch_ipc(&self) -> xabi::Result<Option<Vec<u8>>> {
        let mut stream = self.inner.lock().await;
        match stream.next().await {
            Some(Ok(batch)) => record_batch_to_ipc(&batch).map(Some),
            Some(Err(err)) => Err(xabi::Error::Export(err.to_string())),
            None => Ok(None),
        }
    }
}

struct HostIndexStore {
    inner: Arc<dyn IndexStore>,
}

impl HostIndexStore {
    fn new(inner: Arc<dyn IndexStore>) -> Self {
        Self { inner }
    }
}

impl LanceIndexStoreAbi for HostIndexStore {
    async fn write_record_batches(&self, name: &str, batches_ipc: &[u8]) -> xabi::Result<Vec<u8>> {
        let batches = ipc_to_record_batches(batches_ipc)?;
        let Some(first) = batches.first() else {
            return Err(xabi::Error::Export(
                "write_record_batches requires at least one batch".to_string(),
            ));
        };
        let mut writer = self
            .inner
            .new_index_file(name, first.schema())
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        for batch in batches {
            writer
                .write_record_batch(batch)
                .await
                .map_err(|err| xabi::Error::Export(err.to_string()))?;
        }
        let file = writer
            .finish()
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        serde_json::to_vec(&json!({
            "path": file.path,
            "size_bytes": file.size_bytes,
        }))
        .map_err(|err| xabi::Error::Export(err.to_string()))
    }

    async fn read_record_batches(&self, name: &str) -> xabi::Result<Vec<u8>> {
        let reader = self
            .inner
            .open_index_file(name)
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        let batch = reader
            .read_range(0..reader.num_rows(), None)
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        record_batches_to_ipc(batch.schema(), &[batch])
    }
}

struct HostIndexBuildProgress {
    inner: Arc<dyn IndexBuildProgress>,
}

impl HostIndexBuildProgress {
    fn new(inner: Arc<dyn IndexBuildProgress>) -> Self {
        Self { inner }
    }
}

impl LanceIndexBuildProgressAbi for HostIndexBuildProgress {
    async fn stage_start(&self, name: &str, total: u64, unit: &str) -> xabi::Result<()> {
        self.inner
            .stage_start(name, Some(total), unit)
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))
    }

    async fn stage_progress(&self, name: &str, amount: u64) -> xabi::Result<()> {
        self.inner
            .stage_progress(name, amount)
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))
    }

    async fn stage_complete(&self, name: &str) -> xabi::Result<()> {
        self.inner
            .stage_complete(name)
            .await
            .map_err(|err| xabi::Error::Export(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_index::scalar::ScalarIndexParams;

    use crate::Dataset;
    use crate::dataset::builder::DatasetBuilder;
    use crate::index::DatasetIndexExt;
    use crate::session::Session;

    fn mock_btree_plugin_path() -> std::path::PathBuf {
        if let Some(path) = std::env::var_os("LANCE_XABI_MOCK_BTREE_PLUGIN") {
            return path.into();
        }

        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("lance crate should be under rust/lance");
        let manifest_path = workspace
            .join("rust")
            .join("lance")
            .join("tests")
            .join("xabi_plugins")
            .join("mock_btree")
            .join("Cargo.toml");
        let target_dir = workspace.join("target").join("xabi-mock-btree-test");
        let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
            .current_dir(workspace)
            .args([
                "build",
                "--manifest-path",
                manifest_path.to_str().unwrap(),
                "--target-dir",
            ])
            .arg(&target_dir)
            .status()
            .expect("failed to build mock xabi btree plugin");
        assert!(status.success(), "failed to build mock xabi btree plugin");

        let library_name = if cfg!(target_os = "windows") {
            "lance_index_plugin_mock_btree.dll"
        } else if cfg!(target_os = "macos") {
            "liblance_index_plugin_mock_btree.dylib"
        } else {
            "liblance_index_plugin_mock_btree.so"
        };
        target_dir.join("debug").join(library_name)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_xabi_mock_btree_scalar_index_end_to_end() {
        let plugin_path = mock_btree_plugin_path();
        assert!(
            plugin_path.exists(),
            "mock xabi btree plugin does not exist at {}",
            plugin_path.display()
        );

        let mut session = Session::default();
        let registered = unsafe {
            session
                .register_xabi_index_plugin_library(&plugin_path)
                .unwrap()
        };
        assert_eq!(registered, vec!["mock-btree".to_string()]);
        let session = Arc::new(session);

        let tempdir = tempfile::tempdir().unwrap();
        let uri = tempdir.path().to_str().unwrap().to_string();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Int32Array::from(vec![10, 20, 10, 30, 10])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        Dataset::write(reader, &uri, None).await.unwrap();

        let mut dataset = DatasetBuilder::from_uri(&uri)
            .with_session(session.clone())
            .load()
            .await
            .unwrap();
        let params = ScalarIndexParams::new("mock-btree".to_string());
        dataset
            .create_index(
                &["value"],
                lance_index::IndexType::Scalar,
                Some("value_mock_btree_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        let dataset = DatasetBuilder::from_uri(&uri)
            .with_session(session)
            .load()
            .await
            .unwrap();
        let filtered = dataset
            .scan()
            .filter("value = 10")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = filtered["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values();
        assert_eq!(ids, &[1, 3, 5]);

        let descriptions = dataset.describe_indices(None).await.unwrap();
        assert_eq!(descriptions.len(), 1);
        let description = &descriptions[0];
        assert_eq!(description.name(), "value_mock_btree_idx");
        assert_eq!(description.index_type(), "mock-btree");
        assert_eq!(
            description.type_url(),
            "type.googleapis.com/lance.index.plugin.MockBTreeIndexDetails"
        );

        let stats: serde_json::Value = serde_json::from_str(
            &dataset
                .index_statistics("value_mock_btree_idx")
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stats["indices"][0]["plugin"], "mock-btree");
    }
}
