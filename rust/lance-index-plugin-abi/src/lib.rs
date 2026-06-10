// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable xabi contracts for Lance index plugins.
//!
//! This crate is the ABI boundary between Lance and dynamically loaded index
//! plugins. It intentionally avoids depending on Lance internals. Data crosses
//! the boundary as xabi data, Arrow IPC record batches, and Substrait bytes.

use std::io::Cursor;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;

pub const SCALAR_PLUGIN_TRAIT_ID: &str = "lance.scalar-index-plugin";
pub const SCALAR_INDEX_TRAIT_ID: &str = "lance.scalar-index";
pub const INDEX_STORE_TRAIT_ID: &str = "lance.index-store";
pub const TRAINING_DATA_TRAIT_ID: &str = "lance.training-data";
pub const BUILD_PROGRESS_TRAIT_ID: &str = "lance.index-build-progress";
pub const ABI_VERSION: u32 = 1;

pub const ORDERING_NONE: u32 = 0;
pub const ORDERING_VALUES: u32 = 1;
pub const ORDERING_ADDRESSES: u32 = 2;

pub const EXACTNESS_EXACT: u32 = 0;
pub const EXACTNESS_AT_MOST: u32 = 1;
pub const EXACTNESS_AT_LEAST: u32 = 2;

pub const UPDATE_REQUIRES_OLD_DATA: u32 = 0;
pub const UPDATE_ONLY_NEW_DATA: u32 = 1;

#[xabi::data]
#[derive(Debug, Clone, Copy)]
pub struct TrainingCriteriaAbi {
    pub ordering: u32,
    pub needs_row_ids: u8,
    pub needs_row_addrs: u8,
}

#[xabi::data]
#[derive(Debug, Clone, Copy)]
pub struct UpdateCriteriaAbi {
    pub mode: u32,
    pub data_criteria: TrainingCriteriaAbi,
}

#[xabi::data]
#[derive(Debug, Clone)]
pub struct TrainingPlanAbi {
    pub criteria: TrainingCriteriaAbi,
}

#[xabi::data]
#[derive(Debug, Clone)]
pub struct CreatedIndexAbi {
    pub details_type_url: String,
    pub details: Vec<u8>,
    pub index_version: u32,
}

#[xabi::data]
#[derive(Debug, Clone)]
pub struct QueryPlanAbi {
    pub accepted: u8,
    pub query: Vec<u8>,
    pub needs_recheck: u8,
}

#[xabi::data]
#[derive(Debug, Clone)]
pub struct SearchOutputAbi {
    pub exactness: u32,
    pub row_ids_le: Vec<u8>,
    pub null_row_ids_le: Vec<u8>,
}

#[xabi::data]
#[derive(Clone)]
pub struct TrainInputAbi {
    pub params_json: String,
    pub fragment_ids_le: Vec<u8>,
    pub data: BorrowedLanceTrainingDataAbi,
    pub store: BorrowedLanceIndexStoreAbi,
    pub progress: BorrowedLanceIndexBuildProgressAbi,
}

unsafe impl Send for TrainInputAbi {}
unsafe impl Sync for TrainInputAbi {}

#[xabi::data]
#[derive(Debug, Clone)]
pub struct SubstraitQueryAbi {
    pub version: String,
    pub schema_ipc: Vec<u8>,
    pub expr_bytes: Vec<u8>,
    pub host_query_json: String,
}

#[xabi::data]
#[derive(Clone)]
pub struct RemapInputAbi {
    pub mappings_le: Vec<u8>,
    pub mapped_row_ids_le: Vec<u8>,
    pub has_mapped_row_id: Vec<u8>,
    pub store: BorrowedLanceIndexStoreAbi,
}

unsafe impl Send for RemapInputAbi {}
unsafe impl Sync for RemapInputAbi {}

#[xabi::data]
#[derive(Clone)]
pub struct UpdateInputAbi {
    pub new_data: BorrowedLanceTrainingDataAbi,
    pub store: BorrowedLanceIndexStoreAbi,
}

unsafe impl Send for UpdateInputAbi {}
unsafe impl Sync for UpdateInputAbi {}

#[xabi::xabi(id = TRAINING_DATA_TRAIT_ID, version = ABI_VERSION)]
pub trait LanceTrainingDataAbi {
    async fn next_batch_ipc(&self) -> xabi::Result<Option<Vec<u8>>>;
}

#[xabi::xabi(id = INDEX_STORE_TRAIT_ID, version = ABI_VERSION)]
pub trait LanceIndexStoreAbi {
    async fn write_record_batches(&self, name: &str, batches_ipc: &[u8]) -> xabi::Result<Vec<u8>>;

    async fn read_record_batches(&self, name: &str) -> xabi::Result<Vec<u8>>;
}

#[xabi::xabi(id = BUILD_PROGRESS_TRAIT_ID, version = ABI_VERSION)]
pub trait LanceIndexBuildProgressAbi {
    async fn stage_start(&self, name: &str, total: u64, unit: &str) -> xabi::Result<()>;

    async fn stage_progress(&self, name: &str, amount: u64) -> xabi::Result<()>;

    async fn stage_complete(&self, name: &str) -> xabi::Result<()>;
}

#[xabi::xabi(id = SCALAR_INDEX_TRAIT_ID, version = ABI_VERSION)]
pub trait LanceScalarIndexAbi {
    async fn prewarm(&self) -> xabi::Result<()>;

    fn statistics(&self) -> xabi::Result<String>;

    async fn calculate_included_frags(&self) -> xabi::Result<Vec<u8>>;

    async fn search(&self, query: &[u8]) -> xabi::Result<SearchOutputAbi>;

    fn can_remap(&self) -> bool;

    async fn remap(&self, input: RemapInputAbi) -> xabi::Result<CreatedIndexAbi>;

    async fn update(&self, input: UpdateInputAbi) -> xabi::Result<CreatedIndexAbi>;

    fn update_criteria(&self) -> xabi::Result<UpdateCriteriaAbi>;

    fn derive_index_params_json(&self) -> xabi::Result<String>;
}

#[xabi::xabi(id = SCALAR_PLUGIN_TRAIT_ID, version = ABI_VERSION)]
pub trait LanceScalarIndexPluginAbi {
    fn name(&self) -> String;

    fn version(&self) -> u32;

    fn details_type_url(&self) -> String;

    fn provides_exact_answer(&self) -> bool;

    fn new_training_plan(
        &self,
        params_json: &str,
        field_json: &str,
    ) -> xabi::Result<TrainingPlanAbi>;

    async fn train_index(&self, input: TrainInputAbi) -> xabi::Result<CreatedIndexAbi>;

    async fn load_index(
        &self,
        details: &[u8],
        store: BorrowedLanceIndexStoreAbi,
    ) -> xabi::Result<impl LanceScalarIndexAbi + 'static>;

    fn plan_query(&self, details: &[u8], query: SubstraitQueryAbi) -> xabi::Result<QueryPlanAbi>;

    async fn load_statistics(&self, details: &[u8]) -> xabi::Result<Option<String>>;

    fn details_as_json(&self, details: &[u8]) -> xabi::Result<String>;
}

pub use XabiV1BorrowedTraitLanceIndexBuildProgressAbi as BorrowedLanceIndexBuildProgressAbi;
pub use XabiV1BorrowedTraitLanceIndexStoreAbi as BorrowedLanceIndexStoreAbi;
pub use XabiV1BorrowedTraitLanceTrainingDataAbi as BorrowedLanceTrainingDataAbi;
pub use XabiV1HandleTraitLanceScalarIndexAbi as XabiLanceScalarIndexHandle;
pub use XabiV1HandleTraitLanceScalarIndexPluginAbi as XabiLanceScalarIndexPluginHandle;
pub use XabiV1OwnedTraitLanceIndexBuildProgressAbi as OwnedLanceIndexBuildProgressAbi;
pub use XabiV1OwnedTraitLanceIndexStoreAbi as OwnedLanceIndexStoreAbi;
pub use XabiV1OwnedTraitLanceTrainingDataAbi as OwnedLanceTrainingDataAbi;

pub fn record_batch_to_ipc(batch: &RecordBatch) -> xabi::Result<Vec<u8>> {
    record_batches_to_ipc(batch.schema(), std::slice::from_ref(batch))
}

pub fn record_batches_to_ipc(schema: SchemaRef, batches: &[RecordBatch]) -> xabi::Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, schema.as_ref())
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
        for batch in batches {
            writer
                .write(batch)
                .map_err(|err| xabi::Error::Export(err.to_string()))?;
        }
        writer
            .finish()
            .map_err(|err| xabi::Error::Export(err.to_string()))?;
    }
    Ok(out)
}

pub fn ipc_to_record_batches(bytes: &[u8]) -> xabi::Result<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|err| xabi::Error::Export(err.to_string()))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| xabi::Error::Export(err.to_string()))
}

pub fn u32s_to_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn le_bytes_to_u32s(bytes: &[u8]) -> xabi::Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(xabi::Error::Export(format!(
            "u32 byte payload length {} is not divisible by 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk has length 4")))
        .collect())
}

pub fn u64s_to_le_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn le_bytes_to_u64s(bytes: &[u8]) -> xabi::Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(xabi::Error::Export(format!(
            "u64 byte payload length {} is not divisible by 8",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunk has length 8")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xabi_layout_snapshots() {
        xabi_assert::assert_abi!(XabiV1AbiTraitLanceTrainingDataAbi);
        xabi_assert::assert_abi!(XabiV1AbiTraitLanceIndexStoreAbi);
        xabi_assert::assert_abi!(XabiV1AbiTraitLanceIndexBuildProgressAbi);
        xabi_assert::assert_abi!(XabiV1AbiTraitLanceScalarIndexAbi);
        xabi_assert::assert_abi!(XabiV1AbiTraitLanceScalarIndexPluginAbi);
    }
}
