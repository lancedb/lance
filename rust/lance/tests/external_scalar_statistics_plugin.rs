// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A downstream, statistics-only plugin using the public Rust extension interface.

use std::{
    any::Any,
    process::Command,
    sync::{Arc, LazyLock},
};

use arrow_array::{RecordBatchIterator, record_batch};
use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance::{
    Dataset,
    dataset::WriteParams,
    index::{DatasetIndexExt, DatasetIndexInternalExt, register_scalar_index_plugin},
};
use lance_core::{
    Error, Result, cache::LanceCache, deepsize::DeepSizeOf, utils::row_addr_remap::RowAddrRemap,
};
use lance_index::{
    Index, IndexType,
    metrics::{MetricsCollector, NoOpMetricsCollector},
    progress::IndexBuildProgress,
    scalar::{
        AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, RowIdRemapper, ScalarIndex,
        ScalarIndexParams, SearchResult, TrainingCriteria, TrainingOrdering, UpdateCriteria,
        expression::ScalarQueryParser,
        registry::{BasicTrainer, ScalarIndexPlugin, TrainingRequest},
    },
};
use prost::Name;
use roaring::RoaringBitmap;
use rstest::rstest;
use tempfile::tempdir;

#[derive(Clone, PartialEq, prost::Message)]
struct ExampleIndexDetails {}

impl Name for ExampleIndexDetails {
    const NAME: &'static str = "ExampleIndexDetails";
    const PACKAGE: &'static str = "example.index";
}

#[derive(Debug)]
struct ExampleIndexPlugin;

impl Default for ExampleIndexPlugin {
    fn default() -> Self {
        // Constructors run after freezing, outside the registration lock.
        // A constructor calling registration must fail rather than deadlock.
        let error = register_scalar_index_plugin::<ExampleIndexDetails, Self>().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("before the first plugin registry")
        );
        Self
    }
}

// All tests in this binary register before any dataset or registry operation.
static REGISTRATION: LazyLock<()> = LazyLock::new(|| {
    let error = register_scalar_index_plugin::<
        lance_index::pbold::ZoneMapIndexDetails,
        ExampleIndexPlugin,
    >()
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("already registered"));

    register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap();
    let error =
        register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("already registered"));
});

struct ExampleTrainingRequest(TrainingCriteria);

impl TrainingRequest for ExampleTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.0
    }
}

#[async_trait]
impl BasicTrainer for ExampleIndexPlugin {
    fn new_training_request(
        &self,
        _params: &str,
        _field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        Ok(Box::new(ExampleTrainingRequest(TrainingCriteria::new(
            TrainingOrdering::None,
        ))))
    }

    async fn train_index(
        &self,
        mut data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let mut rows = 0u64;
        while let Some(batch) = data.try_next().await? {
            rows += batch.num_rows() as u64;
        }
        let payload = record_batch!(("rows", UInt64, [rows]))?;
        let mut writer = index_store
            .new_index_file("statistics.lance", payload.schema())
            .await?;
        writer.write_record_batch(payload).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&ExampleIndexDetails {})?,
            index_version: self.version(),
            files: vec![writer.finish().await?],
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for ExampleIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }
    fn name(&self) -> &str {
        "Example"
    }
    fn provides_exact_answer(&self) -> bool {
        false
    }
    fn version(&self) -> u32 {
        1
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        index_version: u32,
        _frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        _cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        index_details.to_msg::<ExampleIndexDetails>()?;
        assert_eq!(index_version, self.version());
        let reader = index_store.open_index_file("statistics.lance").await?;
        let batch = reader.read_record_batch(0, 1).await?;
        let rows = batch["rows"]
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap()
            .value(0);
        Ok(Arc::new(ExampleIndex { rows }))
    }
}

#[derive(Debug, DeepSizeOf)]
struct ExampleIndex {
    rows: u64,
}

#[async_trait]
impl Index for ExampleIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }
    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({ "rows": self.rows }))
    }
    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }
    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for ExampleIndex {
    async fn search(
        &self,
        _query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        Err(Error::not_supported("statistics do not answer predicates"))
    }
    fn can_remap(&self) -> bool {
        false
    }
    async fn remap(
        &self,
        _mapping: &RowAddrRemap,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported("statistics are maintained externally"))
    }
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
        _old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported("statistics are maintained externally"))
    }
    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria {
            requires_old_data: false,
            data_criteria: TrainingCriteria::new(TrainingOrdering::None),
        }
    }
    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::new("example".to_string()))
    }
}

async fn assert_external_index(dataset: &Dataset, rows: u64, coverage: RoaringBitmap) {
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let metadata = &indices[0];
    assert_eq!(
        metadata.index_details.as_ref().unwrap().type_url,
        ExampleIndexDetails::type_url()
    );
    assert_eq!(metadata.fragment_bitmap.as_ref().unwrap(), &coverage);
    assert_eq!(metadata.index_version, 1);
    assert_eq!(metadata.files.as_ref().unwrap()[0].path, "statistics.lance");
    let first = dataset
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(first.as_any().is::<ExampleIndex>());
    assert_eq!(
        first.statistics().unwrap(),
        serde_json::json!({ "rows": rows })
    );
    let second = dataset
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn subsequent_registries_include_external_plugins() {
    LazyLock::force(&REGISTRATION);
    for _ in 0..2 {
        let registry = lance_index::registry::IndexPluginRegistry::with_default_plugins();
        let details = prost_types::Any::from_msg(&ExampleIndexDetails {}).unwrap();
        assert!(registry.supports_details(&details));
        assert_eq!(
            registry.get_plugin_by_details(&details).unwrap().name(),
            "Example"
        );
        assert!(registry.get_plugin_by_name("EXAMPLE").is_ok());
        assert!(registry.get_plugin_by_name("btree").is_ok());
    }
}

#[rstest]
#[case::trained(4, true)]
#[case::definition_only(4, false)]
#[case::empty_dataset(0, true)]
#[tokio::test]
async fn create_and_reopen_external_index(#[case] rows: usize, #[case] train: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [Some(1), None, Some(3), Some(4)]))
        .unwrap()
        .slice(0, rows);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), rows / 2);
    dataset
        .create_index_builder(
            &["value"],
            IndexType::Scalar,
            &ScalarIndexParams::new("EXAMPLE".to_string()),
        )
        .name("example_index".to_string())
        .train(train)
        .await
        .unwrap();
    let coverage = if train {
        (0..rows as u32 / 2).collect()
    } else {
        RoaringBitmap::new()
    };
    let expected_rows = if train { rows as u64 } else { 0 };
    assert_external_index(&dataset, expected_rows, coverage.clone()).await;
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_external_index(&reopened, expected_rows, coverage).await;

    let error =
        register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("before the first plugin registry")
    );
}

#[rstest]
#[case::fragment_coverage(false)]
#[case::empty_coverage(true)]
#[tokio::test]
async fn commit_and_reopen_external_segment(#[case] empty_coverage: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::new("example".to_string());
    let mut segment = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
    // An externally built statistics payload can cover rows without claiming
    // fragment coverage for filtering or row-ID remapping.
    if empty_coverage {
        segment.fragment_bitmap = Some(RoaringBitmap::new());
    }
    let coverage = segment.fragment_bitmap.clone().unwrap();
    let uuid = segment.uuid;
    dataset
        .commit_existing_index_segments("example_index", "value", vec![segment])
        .await
        .unwrap();
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(reopened.load_indices().await.unwrap()[0].uuid, uuid);
    assert_external_index(&reopened, 4, coverage).await;
}

#[tokio::test]
async fn reopen_in_another_process() {
    const DATASET_ENV: &str = "LANCE_EXTERNAL_PLUGIN_TEST_DATASET";
    LazyLock::force(&REGISTRATION);
    if let Some(uri) = std::env::var_os(DATASET_ENV) {
        let dataset = Dataset::open(uri.to_str().unwrap()).await.unwrap();
        assert_external_index(&dataset, 2, RoaringBitmap::new()).await;
        return;
    }

    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [1, 2])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        None,
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::new("example".to_string());
    let mut segment = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    segment.fragment_bitmap = Some(RoaringBitmap::new());
    dataset
        .commit_existing_index_segments("example_index", "value", vec![segment])
        .await
        .unwrap();
    // A fresh executable has its own registration state and index cache.
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "reopen_in_another_process", "--nocapture"])
        .env(DATASET_ENV, dataset.uri())
        .output()
        .unwrap();
    assert!(output.status.success(), "child process failed: {output:?}");
}
