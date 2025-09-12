// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating scalar indices with datasets
//!

use std::sync::{Arc, LazyLock};

use crate::index::DatasetIndexInternalExt;
use crate::session::index_caches::ProstAny;
use crate::{
    dataset::{index::LanceIndexStoreExt, scanner::ColumnOrdering},
    Dataset,
};
use arrow_schema::DataType;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::TryStreamExt;
use itertools::Itertools;
use lance_core::datatypes::Field;
use lance_core::{Error, Result, ROW_ADDR, ROW_ID};
use lance_datafusion::exec::LanceExecutionOptions;
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::inverted::{InvertedIndexPlugin, METADATA_FILE};
use lance_index::scalar::registry::{
    ScalarIndexPlugin, ScalarIndexPluginRegistry, TrainingCriteria, TrainingOrdering,
    VALUE_COLUMN_NAME,
};
use lance_index::scalar::CreatedIndex;
use lance_index::scalar::{
    bitmap::BITMAP_LOOKUP_NAME, inverted::INVERT_LIST_FILE, lance_format::LanceIndexStore,
    ScalarIndex, ScalarIndexParams,
};
use lance_index::{DatasetIndexExt, IndexType, ScalarIndexCriteria, VECTOR_INDEX_VERSION};
use lance_table::format::{Fragment, Index};
use log::info;
use snafu::location;
use tracing::instrument;

// Log an update every TRAINING_UPDATE_FREQ million rows processed
const TRAINING_UPDATE_FREQ: usize = 1000000;

pub(crate) struct TrainingRequest {
    pub fragment_ids: Option<Vec<u32>>,
}

impl TrainingRequest {
    pub fn with_fragment_ids(
        _dataset: Arc<Dataset>,
        _column: String,
        fragment_ids: Vec<u32>,
    ) -> Self {
        Self {
            fragment_ids: Some(fragment_ids),
        }
    }

    async fn create_empty_stream(
        dataset: &Dataset,
        column: &str,
        criteria: &TrainingCriteria,
    ) -> Result<SendableRecordBatchStream> {
        let column_field = dataset.schema().field(column).ok_or(Error::InvalidInput {
            source: format!("No column with name {}", column).into(),
            location: location!(),
        })?;

        let mut fields = Vec::with_capacity(3);
        fields.push(arrow_schema::Field::new(
            VALUE_COLUMN_NAME,
            column_field.data_type(),
            true,
        ));
        if criteria.needs_row_ids {
            fields.push(arrow_schema::Field::new(
                ROW_ID,
                arrow_schema::DataType::UInt64,
                false,
            ));
        }
        if criteria.needs_row_addrs {
            fields.push(arrow_schema::Field::new(
                ROW_ADDR,
                arrow_schema::DataType::UInt64,
                false,
            ));
        }

        // Create schema with the column and row_id field (matching scan_chunks behavior)
        let schema = Arc::new(arrow_schema::Schema::new(fields));

        // Create empty stream
        let empty_stream = futures::stream::empty();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            empty_stream,
        )))
    }
}

pub(crate) async fn scan_training_data(
    dataset: &Dataset,
    column: &str,
    criteria: &TrainingCriteria,
    fragments: Option<Vec<Fragment>>,
) -> Result<SendableRecordBatchStream> {
    let num_rows = dataset.count_all_rows().await?;

    let mut scan = dataset.scan();

    // Fragment filtering is now handled in load_training_data function
    // This function just processes the fragments passed to it

    let column_field = dataset.schema().field(column).ok_or(Error::InvalidInput {
        source: format!("No column with name {}", column).into(),
        location: location!(),
    })?;

    // Datafusion currently has bugs with spilling on string columns
    // See https://github.com/apache/datafusion/issues/10073
    //
    // One we upgrade we can remove this
    let use_spilling = !matches!(
        column_field.data_type(),
        DataType::Utf8 | DataType::LargeUtf8
    );

    // Note: we don't need to sort for TrainingOrdering::Addresses because
    // Lance will return data in the order of the row_address by default.
    if TrainingOrdering::Values == criteria.ordering {
        scan.order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            column.to_string(),
        )]))?;
    }

    if criteria.needs_row_ids {
        scan.with_row_id();
    }
    if criteria.needs_row_addrs {
        scan.with_row_address();
    }

    scan.project_with_transform(&[(VALUE_COLUMN_NAME, column)])?;

    if let Some(fragments) = fragments {
        scan.with_fragments(fragments);
    }

    let batches = scan
        .try_into_dfstream(LanceExecutionOptions {
            use_spilling,
            ..Default::default()
        })
        .await?;

    let schema = batches.schema();
    let mut rows_processed = 0;
    let mut next_update = TRAINING_UPDATE_FREQ;
    let training_uuid = uuid::Uuid::new_v4().to_string();
    info!(
        "Starting index training job with id {} on column {}",
        training_uuid, column
    );
    info!("Training index (job_id={}): 0/{}", training_uuid, num_rows);
    let batches = batches.map_ok(move |batch| {
        rows_processed += batch.num_rows();
        if rows_processed >= next_update {
            next_update += TRAINING_UPDATE_FREQ;
            info!(
                "Training index (job_id={}): {}/{}",
                training_uuid, rows_processed, num_rows
            );
        }
        batch
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, batches)))
}

pub(crate) async fn load_training_data(
    dataset: &Dataset,
    column: &str,
    criteria: &TrainingCriteria,
    fragments: Option<Vec<Fragment>>,
    train: bool,
    fragment_ids: Option<Vec<u32>>,
) -> Result<SendableRecordBatchStream> {
    // Create training request with fragment_ids if provided
    let training_request = Box::new(match fragment_ids.clone() {
        Some(fragment_ids) => TrainingRequest::with_fragment_ids(
            Arc::new(dataset.clone()),
            column.to_string(),
            fragment_ids,
        ),
        None => TrainingRequest { fragment_ids: None },
    });

    if train {
        // Use the training request to scan data with fragment filtering
        if let Some(ref fragment_ids) = training_request.fragment_ids {
            let fragment_ids = fragment_ids.clone().into_iter().dedup().collect_vec();
            let frags = dataset.get_frags_from_ordered_ids(&fragment_ids);
            let frags: Result<Vec<_>> = fragment_ids
                .iter()
                .zip(frags)
                .map(|(id, frag)| {
                    let Some(frag) = frag else {
                        return Err(Error::InvalidInput {
                            source: format!("No fragment with id {}", id).into(),
                            location: location!(),
                        });
                    };
                    Ok(frag.metadata().clone())
                })
                .collect();
            scan_training_data(dataset, column, criteria, Some(frags?)).await
        } else {
            scan_training_data(dataset, column, criteria, fragments).await
        }
    } else {
        TrainingRequest::create_empty_stream(dataset, column, criteria).await
    }
}

// TODO: Allow users to register their own plugins
static SCALAR_INDEX_PLUGIN_REGISTRY: LazyLock<Arc<ScalarIndexPluginRegistry>> =
    LazyLock::new(ScalarIndexPluginRegistry::with_default_plugins);

pub struct IndexDetails(pub Arc<prost_types::Any>);

impl IndexDetails {
    /// Returns true if the index is a vector index
    pub fn is_vector(&self) -> bool {
        self.0.type_url.ends_with("VectorIndexDetails")
    }

    /// Returns true if the index supports FTS
    pub fn supports_fts(&self) -> bool {
        // In the future this may need to change if we want FTS indices to be pluggable
        self.0.type_url.ends_with("InvertedIndexDetails")
    }

    /// Returns the plugin for the index
    pub fn get_plugin(&self) -> Result<&dyn ScalarIndexPlugin> {
        SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_details(self.0.as_ref())
    }

    /// Returns the index version
    pub fn index_version(&self) -> Result<u32> {
        if self.is_vector() {
            Ok(VECTOR_INDEX_VERSION)
        } else {
            self.get_plugin().map(|p| p.version())
        }
    }
}

/// Build a Scalar Index (returns details to store in the manifest)
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_scalar_index(
    dataset: &Dataset,
    column: &str,
    uuid: &str,
    params: &ScalarIndexParams,
    train: bool,
    fragment_ids: Option<Vec<u32>>,
) -> Result<CreatedIndex> {
    let field = dataset.schema().field(column).ok_or(Error::InvalidInput {
        source: format!("No column with name {}", column).into(),
        location: location!(),
    })?;
    let field: arrow_schema::Field = field.into();

    let index_store = LanceIndexStore::from_dataset_for_new(dataset, uuid)?;

    let plugin = SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_name(&params.index_type)?;
    let training_request =
        plugin.new_training_request(params.params.as_deref().unwrap_or("{}"), &field)?;

    let training_data = load_training_data(
        dataset,
        column,
        training_request.criteria(),
        None,
        train,
        fragment_ids,
    )
    .await?;

    plugin
        .train_index(training_data, &index_store, training_request)
        .await
}

/// Build a Scalar Index
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_inverted_index(
    dataset: &Dataset,
    column: &str,
    uuid: &str,
    params: &InvertedIndexParams,
    train: bool,
    fragment_ids: Option<Vec<u32>>,
) -> Result<CreatedIndex> {
    let data = load_training_data(
        dataset,
        column,
        &TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        None,
        train,
        fragment_ids.clone(),
    )
    .await?;
    let index_store = LanceIndexStore::from_dataset_for_new(dataset, uuid)?;
    InvertedIndexPlugin::train_inverted_index(data, &index_store, params.clone(), fragment_ids)
        .await
}

/// Fetches the scalar index plugin for a given index metadata
///
/// The fast path, on newer datasets, is just a plugin lookup by the type URL of the index details.
///
/// If the index details are missing (older dataset) then we need to look at the files present in the
/// index directory to guess the index type.
pub async fn fetch_index_details(
    dataset: &Dataset,
    column: &str,
    index: &Index,
) -> Result<Arc<prost_types::Any>> {
    let index_details = match index.index_details.as_ref() {
        Some(details) => details.clone(),
        None => infer_scalar_index_details(dataset, column, index).await?,
    };

    Ok(index_details)
}

pub async fn open_scalar_index(
    dataset: &Dataset,
    column: &str,
    index: &Index,
    metrics: &dyn MetricsCollector,
) -> Result<Arc<dyn ScalarIndex>> {
    let uuid_str = index.uuid.to_string();
    let index_store = Arc::new(LanceIndexStore::from_dataset_for_existing(dataset, index)?);

    let index_details = fetch_index_details(dataset, column, index).await?;
    let plugin = SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_details(index_details.as_ref())?;

    let frag_reuse_index = dataset.open_frag_reuse_index(metrics).await?;

    let index_cache = dataset
        .index_cache
        .for_index(&uuid_str, frag_reuse_index.as_ref().map(|f| &f.uuid));

    plugin
        .load_index(index_store, &index_details, frag_reuse_index, &index_cache)
        .await
}

pub(crate) async fn infer_scalar_index_details(
    dataset: &Dataset,
    column: &str,
    index: &Index,
) -> Result<Arc<prost_types::Any>> {
    let uuid = index.uuid.to_string();
    let type_key = crate::session::index_caches::ScalarIndexDetailsKey { uuid: &uuid };
    if let Some(index_details) = dataset.index_cache.get_with_key(&type_key).await {
        return Ok(index_details.0.clone());
    }

    let index_dir = dataset.indice_files_dir(index)?.child(uuid.clone());
    let col = dataset.schema().field(column).ok_or(Error::Internal {
        message: format!(
            "Index refers to column {} which does not exist in dataset schema",
            column
        ),
        location: location!(),
    })?;

    let bitmap_page_lookup = index_dir.child(BITMAP_LOOKUP_NAME);
    let inverted_list_lookup = index_dir.child(METADATA_FILE);
    let legacy_inverted_list_lookup = index_dir.child(INVERT_LIST_FILE);
    let index_details = if let DataType::List(_) = col.data_type() {
        prost_types::Any::from_msg(&lance_index::pb::LabelListIndexDetails::default()).unwrap()
    } else if dataset.object_store.exists(&bitmap_page_lookup).await? {
        prost_types::Any::from_msg(&lance_index::pb::BitmapIndexDetails::default()).unwrap()
    } else if dataset.object_store.exists(&inverted_list_lookup).await?
        || dataset
            .object_store
            .exists(&legacy_inverted_list_lookup)
            .await?
    {
        prost_types::Any::from_msg(&lance_index::pb::InvertedIndexDetails::default()).unwrap()
    } else {
        prost_types::Any::from_msg(&lance_index::pb::BTreeIndexDetails::default()).unwrap()
    };

    let index_details = Arc::new(index_details);
    let prost_any = Arc::new(ProstAny(index_details.clone()));

    dataset
        .index_cache
        .insert_with_key(&type_key, prost_any)
        .await;
    Ok(index_details)
}

pub fn index_matches_criteria(
    index: &Index,
    criteria: &ScalarIndexCriteria,
    field: &Field,
    has_multiple_indices: bool,
) -> Result<bool> {
    if let Some(name) = &criteria.has_name {
        if &index.name != name {
            return Ok(false);
        }
    }

    if let Some(for_column) = criteria.for_column {
        if index.fields.len() != 1 {
            return Ok(false);
        }
        if for_column != field.name {
            return Ok(false);
        }
    }

    let index_details = index.index_details.clone().map(IndexDetails);
    let Some(index_details) = index_details else {
        if has_multiple_indices {
            return Err(Error::InvalidInput {
                                source: format!(
                                    "An index {} on the field with id {} co-exists with other indices on the same column but was written with an older Lance version, and this is not supported.  Please retrain this index.",
                                    index.name,
                                    index.fields.first().unwrap_or(&0),
                                ).into(),
                                location: location!(),
                            });
        }

        // If we don't have details then allow it for backwards compatibility
        return Ok(true);
    };

    if index_details.is_vector() {
        // This method is only for finding matching scalar indexes today so reject any vector indexes
        return Ok(false);
    }

    if criteria.must_support_fts && !index_details.supports_fts() {
        return Ok(false);
    }

    // We should not use FTS / NGram indices for exact equality queries
    // (i.e. merge insert with a join on the indexed column)
    if criteria.must_support_exact_equality {
        let plugin = index_details.get_plugin()?;
        if !plugin.provides_exact_answer() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Initialize a scalar index from a source dataset
pub async fn initialize_scalar_index(
    target_dataset: &mut Dataset,
    source_dataset: &Dataset,
    source_index: &Index,
    field_names: &[&str],
) -> Result<()> {
    if field_names.is_empty() || field_names.len() > 1 {
        return Err(Error::Index {
            message: format!("Unsupported fields for scalar index: {:?}", field_names),
            location: location!(),
        });
    }

    // Scalar indices currently support only single fields, use the first one
    let column_name = field_names[0];

    let source_scalar_index = source_dataset
        .open_scalar_index(
            column_name,
            &source_index.uuid.to_string(),
            &NoOpMetricsCollector,
        )
        .await?;

    let params = source_scalar_index.derive_index_params()?;
    let index_type = source_scalar_index.index_type();

    // For Inverted index, we need to parse the params JSON and create InvertedIndexParams
    if index_type == IndexType::Inverted {
        // Extract the JSON string from ScalarIndexParams
        let params_json = params.params.as_ref().ok_or_else(|| Error::Index {
            message: "Inverted index params missing".to_string(),
            location: location!(),
        })?;

        // Parse the JSON into InvertedIndexParams
        let inverted_params: InvertedIndexParams = serde_json::from_str(params_json)?;

        target_dataset
            .create_index(
                &[column_name],
                index_type,
                Some(source_index.name.clone()),
                &inverted_params,
                false,
            )
            .await?;
    } else {
        target_dataset
            .create_index(
                &[column_name],
                index_type,
                Some(source_index.name.clone()),
                &params,
                false,
            )
            .await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    use super::*;
    use arrow::{
        array::AsArray,
        datatypes::{Int32Type, UInt64Type},
    };
    use arrow_schema::DataType;
    use futures::TryStreamExt;
    use lance_core::{datatypes::Field, utils::address::RowAddress};
    use lance_datagen::array;
    use lance_index::{
        pb::{BTreeIndexDetails, InvertedIndexDetails, NGramIndexDetails},
        IndexType,
    };
    use lance_table::format::pb::VectorIndexDetails;

    fn make_index_metadata(
        name: &str,
        field_id: i32,
        index_type: Option<IndexType>,
    ) -> crate::index::IndexMetadata {
        let index_details = index_type
            .map(|index_type| match index_type {
                IndexType::BTree => {
                    prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap()
                }
                IndexType::Inverted => {
                    prost_types::Any::from_msg(&InvertedIndexDetails::default()).unwrap()
                }
                IndexType::NGram => {
                    prost_types::Any::from_msg(&NGramIndexDetails::default()).unwrap()
                }
                IndexType::Vector => {
                    prost_types::Any::from_msg(&VectorIndexDetails::default()).unwrap()
                }
                _ => {
                    unimplemented!("unsupported index type: {}", index_type)
                }
            })
            .map(Arc::new);
        crate::index::IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![field_id],
            dataset_version: 1,
            fragment_bitmap: None,
            index_details,
            index_version: 0,
            created_at: None,
            base_id: None,
        }
    }

    #[test]
    fn test_index_matches_criteria_vector_index() {
        let index1 = make_index_metadata("vector_index", 1, Some(IndexType::Vector));

        let criteria = ScalarIndexCriteria {
            must_support_fts: false,
            must_support_exact_equality: false,
            for_column: None,
            has_name: None,
        };

        let field = Field::new_arrow("mycol", DataType::Int32, true).unwrap();
        let result = index_matches_criteria(&index1, &criteria, &field, true).unwrap();
        assert!(!result);

        let result = index_matches_criteria(&index1, &criteria, &field, false).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_index_matches_criteria_scalar_index() {
        let btree_index = make_index_metadata("btree_index", 1, Some(IndexType::BTree));
        let inverted_index = make_index_metadata("inverted_index", 1, Some(IndexType::Inverted));
        let ngram_index = make_index_metadata("ngram_index", 1, Some(IndexType::NGram));

        let criteria = ScalarIndexCriteria {
            must_support_fts: false,
            must_support_exact_equality: false,
            for_column: None,
            has_name: None,
        };

        let field = Field::new_arrow("mycol", DataType::Int32, true).unwrap();
        let result = index_matches_criteria(&btree_index, &criteria, &field, true).unwrap();
        assert!(result);

        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        // test for_column
        let mut criteria = ScalarIndexCriteria {
            must_support_fts: false,
            must_support_exact_equality: false,
            for_column: Some("mycol"),
            has_name: None,
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        criteria.for_column = Some("mycol2");
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(!result);

        // test has_name
        let mut criteria = ScalarIndexCriteria {
            must_support_fts: false,
            must_support_exact_equality: false,
            for_column: None,
            has_name: Some("btree_index"),
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, true).unwrap();
        assert!(result);
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        criteria.has_name = Some("btree_index2");
        let result = index_matches_criteria(&btree_index, &criteria, &field, true).unwrap();
        assert!(!result);
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(!result);

        // test supports_exact_equality
        let mut criteria = ScalarIndexCriteria {
            must_support_fts: false,
            must_support_exact_equality: true,
            for_column: None,
            has_name: None,
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, false).unwrap();
        assert!(result);

        criteria.must_support_fts = true;
        let result = index_matches_criteria(&inverted_index, &criteria, &field, false).unwrap();
        assert!(!result);

        criteria.must_support_fts = false;
        let result = index_matches_criteria(&ngram_index, &criteria, &field, true).unwrap();
        assert!(!result);

        // test multiple indices
        let mut criteria = ScalarIndexCriteria {
            must_support_fts: false,
            must_support_exact_equality: false,
            for_column: None,
            has_name: None,
        };
        let result = index_matches_criteria(&btree_index, &criteria, &field, true).unwrap();
        assert!(result);

        criteria.must_support_fts = true;
        let result = index_matches_criteria(&inverted_index, &criteria, &field, true).unwrap();
        assert!(result);

        criteria.must_support_fts = false;
        let result = index_matches_criteria(&ngram_index, &criteria, &field, true).unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_load_training_data_addr_sort() {
        // Create test data using lance_datagen
        let dataset = lance_datagen::gen_batch()
            .col("values", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(10))
            .await
            .unwrap();

        // Test scan_aligned_chunks with different chunk sizes
        log::info!("Testing with chunk_size=10:");
        let stream = load_training_data(
            &dataset,
            "values",
            &TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
            None,
            true,
            None,
        )
        .await
        .unwrap();

        // Row addresses should be strictly increasing and ending with fragment id=3
        let mut max_frag_id_seen = 0;
        let mut last_rowaddr = 0;
        for batch in stream.try_collect::<Vec<_>>().await.unwrap() {
            for rowaddr in batch
                .column_by_name(ROW_ADDR)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values()
            {
                assert!(last_rowaddr == 0 || *rowaddr > last_rowaddr);
                last_rowaddr = *rowaddr;
                let frag_id = RowAddress::from(*rowaddr).fragment_id();
                max_frag_id_seen = frag_id;
            }
        }
        assert_eq!(max_frag_id_seen, 3);
    }

    #[tokio::test]
    async fn test_initialize_scalar_index_btree() {
        use crate::dataset::Dataset;
        use arrow_array::types::Float32Type;
        use lance_datagen::{array, BatchCount, RowCount};
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::DatasetIndexExt;
        use tempfile::tempdir;

        let test_dir = tempdir().unwrap();
        let source_uri = test_dir.path().join("source").to_str().unwrap().to_string();
        let target_uri = test_dir.path().join("target").to_str().unwrap().to_string();

        // Create source dataset with BTree index
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("value", array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create BTree index on source with custom zone_size
        use lance_index::scalar::btree::BTreeParameters;

        let btree_params = BTreeParameters {
            zone_size: Some(50),
        };
        let params_json = serde_json::to_value(&btree_params).unwrap();
        let index_params =
            ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree)
                .with_params(params_json);

        source_dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &index_params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "id_btree")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("value", array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize BTree index on target
        super::initialize_scalar_index(&mut target_dataset, &source_dataset, source_index, &["id"])
            .await
            .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "id_btree",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![0],
            "Index should be on field 0 (id)"
        );

        // Verify the index type is correct
        let target_scalar_index = target_dataset
            .open_scalar_index(
                "id",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        assert_eq!(
            target_scalar_index.index_type(),
            IndexType::BTree,
            "Index type should be BTree"
        );

        // Verify BTree parameters are preserved
        let derived_params = target_scalar_index.derive_index_params().unwrap();
        if let Some(params_json) = derived_params.params {
            let params: BTreeParameters = serde_json::from_str(&params_json).unwrap();
            assert_eq!(params.zone_size, Some(50), "BTree zone_size should be 50");
        } else {
            panic!("BTree index should have parameters");
        }
    }

    #[tokio::test]
    async fn test_initialize_scalar_index_bitmap() {
        use crate::dataset::Dataset;
        use arrow_array::types::Float32Type;
        use lance_datagen::{array, BatchCount, RowCount};
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::DatasetIndexExt;
        use tempfile::tempdir;

        let test_dir = tempdir().unwrap();
        let source_uri = test_dir.path().join("source").to_str().unwrap().to_string();
        let target_uri = test_dir.path().join("target").to_str().unwrap().to_string();

        // Create source dataset with low cardinality column for bitmap index
        let source_reader = lance_datagen::gen_batch()
            .col(
                "category",
                array::cycle::<Int32Type>((0..10).collect::<Vec<_>>()),
            )
            .col("value", array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create Bitmap index on source
        source_dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                Some("category_bitmap".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "category_bitmap")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col(
                "category",
                array::cycle::<Int32Type>((0..10).collect::<Vec<_>>()),
            )
            .col("value", array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize Bitmap index on target
        super::initialize_scalar_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["category"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "category_bitmap",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![0],
            "Index should be on field 0 (category)"
        );
    }

    #[tokio::test]
    async fn test_initialize_scalar_index_inverted() {
        use crate::dataset::Dataset;
        use lance_datagen::{array, BatchCount, ByteCount, RowCount};
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
        use lance_index::DatasetIndexExt;
        use tempfile::tempdir;

        let test_dir = tempdir().unwrap();
        let source_uri = test_dir.path().join("source").to_str().unwrap().to_string();
        let target_uri = test_dir.path().join("target").to_str().unwrap().to_string();

        // Create source dataset with text column for inverted index
        let source_reader = lance_datagen::gen_batch()
            .col("text", array::rand_utf8(ByteCount::from(50), false))
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create Inverted (FTS) index on source with custom parameters
        let inverted_params = InvertedIndexParams::default()
            .base_tokenizer("whitespace".to_string())
            .with_position(true)
            .max_token_length(Some(128))
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .ascii_folding(false);

        source_dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &inverted_params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "text_fts")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("text", array::rand_utf8(ByteCount::from(50), false))
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize Inverted index on target
        super::initialize_scalar_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["text"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "text_fts",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![0],
            "Index should be on field 0 (text)"
        );

        // Verify the index type is correct
        let target_scalar_index = target_dataset
            .open_scalar_index(
                "text",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        assert_eq!(
            target_scalar_index.index_type(),
            IndexType::Inverted,
            "Index type should be Inverted"
        );

        // Verify parameters are preserved by parsing the JSON params
        let derived_params = target_scalar_index.derive_index_params().unwrap();
        assert!(
            derived_params.params.is_some(),
            "Inverted index should have parameters"
        );

        // Parse the JSON parameters to verify specific fields
        let params_json = derived_params.params.unwrap();
        let params: serde_json::Value = serde_json::from_str(&params_json).unwrap();

        // Verify all the custom parameters we set
        assert_eq!(
            params["base_tokenizer"].as_str().unwrap(),
            "whitespace",
            "Base tokenizer should be whitespace"
        );
        assert!(
            params["with_position"].as_bool().unwrap(),
            "with_position should be true"
        );
        assert_eq!(
            params["max_token_length"].as_u64().unwrap(),
            128,
            "max_token_length should be 128"
        );
        assert!(
            !params["lower_case"].as_bool().unwrap(),
            "lower_case should be false"
        );
        assert!(!params["stem"].as_bool().unwrap(), "stem should be false");
        assert!(
            !params["remove_stop_words"].as_bool().unwrap(),
            "remove_stop_words should be false"
        );
        assert!(
            !params["ascii_folding"].as_bool().unwrap(),
            "ascii_folding should be false"
        );
    }

    #[tokio::test]
    async fn test_initialize_scalar_index_zonemap() {
        use crate::dataset::Dataset;
        use arrow_array::types::Float32Type;
        use lance_datagen::{array, BatchCount, RowCount};
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::zonemap::ZoneMapIndexBuilderParams;
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::DatasetIndexExt;
        use tempfile::tempdir;

        let test_dir = tempdir().unwrap();
        let source_uri = test_dir.path().join("source").to_str().unwrap().to_string();
        let target_uri = test_dir.path().join("target").to_str().unwrap().to_string();

        // Create source dataset with ZoneMap index
        let source_reader = lance_datagen::gen_batch()
            .col("value", array::rand::<Float32Type>())
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create ZoneMap index on source with custom rows_per_zone
        let zonemap_params = ZoneMapIndexBuilderParams::new(200);
        let params_json = serde_json::to_value(&zonemap_params).unwrap();
        let index_params =
            ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::ZoneMap)
                .with_params(params_json);

        source_dataset
            .create_index(
                &["value"],
                IndexType::ZoneMap,
                Some("value_zonemap".to_string()),
                &index_params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "value_zonemap")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("value", array::rand::<Float32Type>())
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize ZoneMap index on target
        super::initialize_scalar_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["value"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "value_zonemap",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![0],
            "Index should be on field 0 (value)"
        );

        // Verify the index type is correct
        let target_scalar_index = target_dataset
            .open_scalar_index(
                "value",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        assert_eq!(
            target_scalar_index.index_type(),
            IndexType::ZoneMap,
            "Index type should be ZoneMap"
        );

        // Verify ZoneMap statistics show correct rows_per_zone
        let stats = target_scalar_index.statistics().unwrap();
        let max_zonemap_size = stats["max_zonemap_size"].as_u64().unwrap();
        assert_eq!(
            max_zonemap_size, 200,
            "ZoneMap max_zonemap_size should be 200"
        );
    }
}
