// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance_index_plugin_abi::{
    CreatedIndexAbi, EXACTNESS_EXACT, LanceScalarIndexAbi, LanceScalarIndexPluginAbi,
    ORDERING_VALUES, QueryPlanAbi, SearchOutputAbi, SubstraitQueryAbi, TrainInputAbi,
    TrainingCriteriaAbi, TrainingPlanAbi, UPDATE_ONLY_NEW_DATA, UpdateCriteriaAbi,
    ipc_to_record_batches, record_batches_to_ipc, u32s_to_le_bytes, u64s_to_le_bytes,
};
use serde_json::Value;

const PLUGIN_NAME: &str = "mock-btree";
const DETAILS_TYPE_URL: &str = "type.googleapis.com/lance.index.plugin.MockBTreeIndexDetails";
const INDEX_FILE: &str = "mock_btree.lance";

#[derive(Default)]
struct MockBTreePlugin;

struct MockBTreeIndex {
    rows: Vec<(i32, u64)>,
}

impl MockBTreeIndex {
    async fn load(store: lance_index_plugin_abi::BorrowedLanceIndexStoreAbi) -> xabi::Result<Self> {
        let bytes = store
            .read_record_batches(INDEX_FILE)
            .await
            .map_err(xabi_call_to_error)?;
        let rows = rows_from_batches(&ipc_to_record_batches(&bytes)?)?;
        Ok(Self { rows })
    }
}

impl MockBTreePlugin {
    fn name(&self) -> String {
        PLUGIN_NAME.to_string()
    }

    fn version(&self) -> u32 {
        1
    }

    fn details_type_url(&self) -> String {
        DETAILS_TYPE_URL.to_string()
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn new_training_plan(
        &self,
        _params_json: &str,
        _field_json: &str,
    ) -> xabi::Result<TrainingPlanAbi> {
        Ok(TrainingPlanAbi::new(TrainingCriteriaAbi::new(
            ORDERING_VALUES,
            1,
            0,
        )))
    }

    async fn train_index(&self, input: TrainInputAbi) -> xabi::Result<CreatedIndexAbi> {
        let mut rows = Vec::new();
        while let Some(batch_ipc) = input
            .data
            .next_batch_ipc()
            .await
            .map_err(xabi_call_to_error)?
        {
            rows.extend(rows_from_batches(&ipc_to_record_batches(&batch_ipc)?)?);
        }
        rows.sort_unstable_by_key(|(value, row_id)| (*value, *row_id));

        let batch = batch_from_rows(&rows)?;
        let bytes = record_batches_to_ipc(batch.schema(), &[batch])?;
        input
            .store
            .write_record_batches(INDEX_FILE, &bytes)
            .await
            .map_err(xabi_call_to_error)?;

        Ok(CreatedIndexAbi::new(
            DETAILS_TYPE_URL.to_string(),
            b"mock-btree-v1".to_vec(),
            1,
        ))
    }

    async fn load_index(
        &self,
        _details: &[u8],
        store: lance_index_plugin_abi::BorrowedLanceIndexStoreAbi,
    ) -> xabi::Result<impl LanceScalarIndexAbi + 'static> {
        MockBTreeIndex::load(store).await
    }

    fn plan_query(&self, _details: &[u8], query: SubstraitQueryAbi) -> xabi::Result<QueryPlanAbi> {
        let host_query: Value = serde_json::from_str(&query.host_query_json)
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        let accepted = matches!(
            host_query.get("op").and_then(Value::as_str),
            Some("=") | Some("Eq")
        );
        if accepted && (query.schema_ipc.is_empty() || query.expr_bytes.is_empty()) {
            return Err(xabi::Error::Export(
                "accepted mock-btree queries must include Substrait payload".to_string(),
            ));
        }
        Ok(QueryPlanAbi::new(
            u8::from(accepted),
            query.host_query_json.into_bytes(),
            0,
        ))
    }

    async fn load_statistics(&self, _details: &[u8]) -> xabi::Result<Option<String>> {
        Ok(Some(r#"{"plugin":"mock-btree"}"#.to_string()))
    }

    fn details_as_json(&self, details: &[u8]) -> xabi::Result<String> {
        Ok(format!(
            r#"{{"plugin":"mock-btree","details_size":{}}}"#,
            details.len()
        ))
    }
}

impl LanceScalarIndexAbi for MockBTreeIndex {
    async fn prewarm(&self) -> xabi::Result<()> {
        Ok(())
    }

    fn statistics(&self) -> xabi::Result<String> {
        Ok(format!(r#"{{"rows":{}}}"#, self.rows.len()))
    }

    async fn calculate_included_frags(&self) -> xabi::Result<Vec<u8>> {
        let mut fragments = self
            .rows
            .iter()
            .map(|(_, row_id)| (*row_id >> 32) as u32)
            .collect::<Vec<_>>();
        fragments.sort_unstable();
        fragments.dedup();
        Ok(u32s_to_le_bytes(&fragments))
    }

    async fn search(&self, query: &[u8]) -> xabi::Result<SearchOutputAbi> {
        let query: Value =
            serde_json::from_slice(query).map_err(|err| xabi::Error::Export(err.to_string()))?;
        let needle = query
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(|| xabi::Error::Export("query is missing string value".to_string()))?
            .parse::<i32>()
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        let row_ids = self
            .rows
            .iter()
            .filter_map(|(value, row_id)| (*value == needle).then_some(*row_id))
            .collect::<Vec<_>>();
        Ok(SearchOutputAbi::new(
            EXACTNESS_EXACT,
            u64s_to_le_bytes(&row_ids),
            Vec::new(),
        ))
    }

    fn can_remap(&self) -> bool {
        false
    }

    async fn remap(
        &self,
        _input: lance_index_plugin_abi::RemapInputAbi,
    ) -> xabi::Result<CreatedIndexAbi> {
        Err(xabi::Error::Export(
            "mock-btree remap is not implemented".to_string(),
        ))
    }

    async fn update(
        &self,
        _input: lance_index_plugin_abi::UpdateInputAbi,
    ) -> xabi::Result<CreatedIndexAbi> {
        Err(xabi::Error::Export(
            "mock-btree update is not implemented".to_string(),
        ))
    }

    fn update_criteria(&self) -> xabi::Result<UpdateCriteriaAbi> {
        Ok(UpdateCriteriaAbi::new(
            UPDATE_ONLY_NEW_DATA,
            TrainingCriteriaAbi::new(ORDERING_VALUES, 1, 0),
        ))
    }

    fn derive_index_params_json(&self) -> xabi::Result<String> {
        Ok(r#"{"index_type":"mock-btree"}"#.to_string())
    }
}

fn rows_from_batches(batches: &[RecordBatch]) -> xabi::Result<Vec<(i32, u64)>> {
    let mut rows = Vec::new();
    for batch in batches {
        let values = batch
            .column_by_name("value")
            .ok_or_else(|| {
                xabi::Error::Export("training batch is missing value column".to_string())
            })?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| xabi::Error::Export("value column must be Int32".to_string()))?;
        let row_ids = batch
            .column_by_name("_rowid")
            .ok_or_else(|| {
                xabi::Error::Export("training batch is missing _rowid column".to_string())
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| xabi::Error::Export("_rowid column must be UInt64".to_string()))?;
        for row_idx in 0..batch.num_rows() {
            if values.is_null(row_idx) || row_ids.is_null(row_idx) {
                continue;
            }
            rows.push((values.value(row_idx), row_ids.value(row_idx)));
        }
    }
    Ok(rows)
}

fn batch_from_rows(rows: &[(i32, u64)]) -> xabi::Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, true),
        Field::new("_rowid", DataType::UInt64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|(value, _)| *value),
            )),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|(_, row_id)| *row_id),
            )),
        ],
    )
    .map_err(|err| xabi::Error::Export(err.to_string()))
}

fn xabi_call_to_error(err: impl std::fmt::Display) -> xabi::Error {
    xabi::Error::Export(err.to_string())
}

#[xabi::module]
mod exports {
    use super::*;

    #[xabi::xabi(name = "mock-btree", version = 1)]
    impl LanceScalarIndexPluginAbi for MockBTreePlugin {
        fn name(&self) -> String {
            MockBTreePlugin::name(self)
        }

        fn version(&self) -> u32 {
            MockBTreePlugin::version(self)
        }

        fn details_type_url(&self) -> String {
            MockBTreePlugin::details_type_url(self)
        }

        fn provides_exact_answer(&self) -> bool {
            MockBTreePlugin::provides_exact_answer(self)
        }

        fn new_training_plan(
            &self,
            params_json: &str,
            field_json: &str,
        ) -> xabi::Result<TrainingPlanAbi> {
            MockBTreePlugin::new_training_plan(self, params_json, field_json)
        }

        async fn train_index(&self, input: TrainInputAbi) -> xabi::Result<CreatedIndexAbi> {
            MockBTreePlugin::train_index(self, input).await
        }

        async fn load_index(
            &self,
            details: &[u8],
            store: lance_index_plugin_abi::BorrowedLanceIndexStoreAbi,
        ) -> xabi::Result<impl LanceScalarIndexAbi + 'static> {
            MockBTreePlugin::load_index(self, details, store).await
        }

        fn plan_query(
            &self,
            details: &[u8],
            query: SubstraitQueryAbi,
        ) -> xabi::Result<QueryPlanAbi> {
            MockBTreePlugin::plan_query(self, details, query)
        }

        async fn load_statistics(&self, details: &[u8]) -> xabi::Result<Option<String>> {
            MockBTreePlugin::load_statistics(self, details).await
        }

        fn details_as_json(&self, details: &[u8]) -> xabi::Result<String> {
            MockBTreePlugin::details_as_json(self, details)
        }
    }
}
