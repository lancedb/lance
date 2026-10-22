// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A downstream, statistics-only plugin using the public Rust extension interface.

use std::{
    any::Any,
    sync::{Arc, LazyLock},
};

use arrow_array::{RecordBatchIterator, record_batch};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance::{
    Dataset,
    dataset::{
        ColumnAlteration, WriteParams,
        optimize::{CompactionOptions, compact_files},
    },
    index::{DatasetIndexExt, DatasetIndexInternalExt, register_scalar_index_plugin},
};
use lance_core::{
    Error, Result, cache::LanceCache, deepsize::DeepSizeOf, utils::row_addr_remap::RowAddrRemap,
};
use lance_index::{
    Index, IndexType,
    metrics::{MetricsCollector, NoOpMetricsCollector},
    optimize::OptimizeOptions,
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
use tempfile::{TempDir, tempdir};

#[derive(Clone, PartialEq, prost::Message)]
struct AuxiliaryIndexDetails {}

impl Name for AuxiliaryIndexDetails {
    const NAME: &'static str = "AuxiliaryIndexDetails";
    const PACKAGE: &'static str = "auxiliary.index";
}

#[derive(Debug, Default)]
struct AuxiliaryIndexPlugin;

// All tests in this binary register before any dataset or registry operation.
static REGISTRATION: LazyLock<()> = LazyLock::new(|| {
    register_scalar_index_plugin::<AuxiliaryIndexDetails, AuxiliaryIndexPlugin>().unwrap();
});

struct AuxiliaryTrainingRequest(TrainingCriteria);

impl TrainingRequest for AuxiliaryTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.0
    }
}

#[async_trait]
impl BasicTrainer for AuxiliaryIndexPlugin {
    fn new_training_request(
        &self,
        _params: &str,
        _field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        Ok(Box::new(AuxiliaryTrainingRequest(TrainingCriteria::new(
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
            index_details: prost_types::Any::from_msg(&AuxiliaryIndexDetails {})?,
            index_version: self.version(),
            files: vec![writer.finish().await?],
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for AuxiliaryIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }
    fn name(&self) -> &str {
        "Auxiliary"
    }
    fn provides_exact_answer(&self) -> bool {
        false
    }
    fn supports_optimization(&self) -> bool {
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
        index_details.to_msg::<AuxiliaryIndexDetails>()?;
        assert_eq!(index_version, self.version());
        let reader = index_store.open_index_file("statistics.lance").await?;
        let batch = reader.read_record_batch(0, 1).await?;
        let rows = batch["rows"]
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap()
            .value(0);
        Ok(Arc::new(AuxiliaryIndex { rows }))
    }
}

#[derive(Debug, DeepSizeOf)]
struct AuxiliaryIndex {
    rows: u64,
}

#[async_trait]
impl Index for AuxiliaryIndex {
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
impl ScalarIndex for AuxiliaryIndex {
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
        Ok(ScalarIndexParams::new("auxiliary".to_string()))
    }
}

async fn dataset_with_auxiliary_index(stable_row_ids: bool) -> (TempDir, Dataset) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(
        ("value", Int64, [Some(1), None, Some(3), Some(4)]),
        ("other", Int64, [1, 2, 3, 4]),
        ("keep", Int64, [1, 2, 3, 4])
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    dataset
        .create_index_builder(
            &["value"],
            IndexType::Scalar,
            &ScalarIndexParams::new("auxiliary".into()),
        )
        .name("profile".into())
        .train(false)
        .await
        .unwrap();
    (directory, dataset)
}

#[rstest]
#[case::default(OptimizeOptions::default())]
#[case::append(OptimizeOptions::append())]
#[case::merge(OptimizeOptions::merge(1))]
#[case::retrain(OptimizeOptions::retrain())]
#[case::bounded(OptimizeOptions::default().max_rows_per_task(Some(2)))]
#[tokio::test]
async fn externally_maintained_index_is_skipped(#[case] options: OptimizeOptions) {
    let (_directory, mut dataset) = dataset_with_auxiliary_index(false).await;
    let before = dataset.load_indices().await.unwrap();
    let version = dataset.version().version;
    let plan = lance::index::optimize::plan_optimize_indices(&dataset, &options)
        .await
        .unwrap();
    assert!(plan.tasks.is_empty());
    assert!(plan.merges.is_empty());
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), before);

    let error = dataset
        .optimize_indices(&options.index_names(vec!["profile".into()]))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("profile"));
    assert!(error.to_string().contains("maintained externally"));
}

#[tokio::test]
async fn externally_maintained_index_does_not_block_builtin_optimization() {
    let (_directory, mut dataset) = dataset_with_auxiliary_index(false).await;
    let profile = dataset.load_indices().await.unwrap()[0].clone();
    dataset
        .create_index(
            &["other"],
            IndexType::BTree,
            Some("lookup".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let batch = record_batch!(
        ("value", Int64, [5, 6]),
        ("other", Int64, [5, 6]),
        ("keep", Int64, [5, 6])
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().any(|index| index == &profile));
    let lookup = indices.iter().find(|index| index.name == "lookup").unwrap();
    assert_eq!(lookup.fragment_bitmap.as_ref().unwrap().len(), 3);
    assert_eq!(
        dataset
            .scan()
            .filter("other >= 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        2
    );
}

#[rstest]
#[case::row_addresses(false)]
#[case::stable_row_ids(true)]
#[tokio::test]
async fn empty_coverage_survives_compaction(#[case] stable_row_ids: bool) {
    let (_directory, mut dataset) = dataset_with_auxiliary_index(stable_row_ids).await;
    let before = dataset.load_indices().await.unwrap();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
    assert_eq!(dataset.load_indices().await.unwrap(), before);
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    let index = reopened
        .open_scalar_index("value", &before[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!index.can_remap());
    assert_eq!(index.statistics().unwrap()["rows"], 0);
}

#[tokio::test]
async fn registered_plugin_type_is_visible() {
    let (_directory, dataset) = dataset_with_auxiliary_index(false).await;
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].name(), "profile");
    assert_eq!(descriptions[0].index_type(), "Auxiliary");
    let statistics: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("profile").await.unwrap()).unwrap();
    assert_eq!(statistics["index_type"], "Auxiliary");
}

#[tokio::test]
async fn schema_changes_and_drop_index_respect_plugin_fields() {
    let (_directory, mut dataset) = dataset_with_auxiliary_index(false).await;
    let before = dataset.load_indices().await.unwrap();
    let version = dataset.version().version;
    let error = dataset
        .alter_columns(&[ColumnAlteration::new("value".into()).cast_to(DataType::Int32)])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("profile"));
    assert!(error.to_string().contains("drop_index()"));
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), before);

    dataset.drop_index("profile").await.unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
    dataset
        .alter_columns(&[ColumnAlteration::new("value".into()).cast_to(DataType::Int32)])
        .await
        .unwrap();
    assert_eq!(
        dataset.schema().field("value").unwrap().data_type(),
        DataType::Int32
    );
}

#[tokio::test]
async fn dropping_plugin_field_removes_index() {
    let (_directory, mut dataset) = dataset_with_auxiliary_index(false).await;
    dataset.drop_columns(&["other"]).await.unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    dataset.drop_columns(&["value"]).await.unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
}
