// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A downstream maintainer uses real BTree files to exercise the generic hook.
//! This needs no new index type or process-global reader registration.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow_array::{RecordBatchIterator, record_batch};
use async_trait::async_trait;
use lance::{
    Dataset, Error, Result,
    dataset::WriteParams,
    index::{
        DatasetIndexExt, DatasetIndexInternalExt, IndexSegment, IntoIndexSegment,
        optimize::plan_optimize_indices,
        optimizer::{ScalarIndexOptimization, ScalarIndexOptimizeContext, ScalarIndexOptimizer},
    },
    session::Session,
};
use lance_core::deepsize::DeepSizeOf;
use lance_index::{IndexType, optimize::OptimizeOptions, pbold, scalar::ScalarIndexParams};
use rstest::rstest;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, DeepSizeOf)]
enum Mode {
    Unchanged,
    Copy,
    AddUnindexed,
    Error,
    LoseCoverage,
    ForeignReplacement,
    ReuseUuid,
    NonliveCoverage,
    OverlapRetained,
    WrongField,
    WrongType,
    FutureVersion,
    ConcurrentAppend,
}

#[derive(Debug, DeepSizeOf)]
struct CopyOptimizer {
    mode: Mode,
    calls: AtomicUsize,
}

#[async_trait]
impl ScalarIndexOptimizer for CopyOptimizer {
    async fn optimize(
        &self,
        context: ScalarIndexOptimizeContext<'_>,
        _: &OptimizeOptions,
    ) -> Result<ScalarIndexOptimization> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.mode, Mode::Unchanged) {
            return Ok(ScalarIndexOptimization::Unchanged);
        }
        if matches!(self.mode, Mode::Error) {
            return Err(Error::not_supported("test maintainer refused"));
        }
        let dataset = context.dataset;
        if matches!(self.mode, Mode::AddUnindexed) {
            let mut dataset = dataset.clone();
            let reference = context.segments[0];
            let ids = dataset
                .unindexed_fragments(&reference.name)
                .await?
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect();
            let segment = dataset
                .create_index_builder(&["value"], IndexType::BTree, &ScalarIndexParams::default())
                .name(reference.name.clone())
                .replace(true)
                .fragments(ids)
                .execute_uncommitted()
                .await?;
            return Ok(ScalarIndexOptimization::Updated {
                new_segments: vec![segment.into_index_segment()?],
                replaced_segments: Vec::new(),
            });
        }
        let store = dataset.object_store(None).await?;
        let mut outputs = Vec::new();
        let mut replaced = Vec::new();
        for input in context.segments {
            let uuid = if matches!(self.mode, Mode::ReuseUuid) {
                input.uuid
            } else {
                Uuid::new_v4()
            };
            if uuid != input.uuid {
                for file in input.files.as_ref().unwrap() {
                    store
                        .copy(
                            &dataset
                                .indices_dir()
                                .join(input.uuid.to_string())
                                .join(file.path.as_str()),
                            &dataset
                                .indices_dir()
                                .join(uuid.to_string())
                                .join(file.path.as_str()),
                        )
                        .await?;
                }
            }
            let mut bitmap = input.fragment_bitmap.clone().unwrap();
            let mut fields = input.fields.clone();
            let mut details = input.index_details.clone().unwrap();
            match self.mode {
                Mode::LoseCoverage => bitmap.clear(),
                Mode::NonliveCoverage => {
                    bitmap.insert(u32::MAX);
                }
                Mode::WrongField => fields[0] += 1,
                Mode::WrongType => {
                    details = Arc::new(prost_types::Any::from_msg(&pbold::BitmapIndexDetails {})?);
                }
                _ => {}
            }
            outputs.push(IndexSegment::new(
                uuid,
                bitmap,
                fields,
                details,
                input.index_version,
                if matches!(self.mode, Mode::FutureVersion) {
                    dataset.version_id() + 1
                } else {
                    input.dataset_version
                },
                [],
            ));
            replaced.push(input.uuid);
        }
        match self.mode {
            Mode::ForeignReplacement => replaced.push(Uuid::new_v4()),
            Mode::OverlapRetained => replaced.clear(),
            Mode::ConcurrentAppend => {
                let batch = record_batch!(("value", Int64, [5, 6]))?;
                let schema = batch.schema();
                let mut concurrent = dataset.clone();
                concurrent
                    .append(RecordBatchIterator::new([Ok(batch)], schema), None)
                    .await?;
            }
            _ => {}
        }
        Ok(ScalarIndexOptimization::Updated {
            new_segments: outputs,
            replaced_segments: replaced,
        })
    }
}

async fn fixture(mode: Mode, full_coverage: bool) -> (Dataset, Arc<CopyOptimizer>) {
    let optimizer = Arc::new(CopyOptimizer {
        mode,
        calls: AtomicUsize::new(0),
    });
    let mut session = Session::default();
    session
        .register_scalar_index_optimizer::<pbold::BTreeIndexDetails>(optimizer.clone())
        .unwrap();
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            session: Some(Arc::new(session)),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let ids = dataset.fragment_bitmap().iter().collect::<Vec<_>>();
    dataset
        .create_index_builder(&["value"], IndexType::BTree, &ScalarIndexParams::default())
        .name("custom".to_string())
        .fragments(if full_coverage { ids } else { vec![ids[0]] })
        .await
        .unwrap();
    (dataset, optimizer)
}

async fn unreadable_unindexed_fragment(dataset: &Dataset) {
    let file = &dataset.fragments()[1].files[0];
    dataset
        .object_store(None)
        .await
        .unwrap()
        .delete(&dataset.data_dir().join(file.path.as_str()))
        .await
        .unwrap();
}

#[rstest]
#[case::uncovered(false)]
#[case::fully_covered(true)]
#[tokio::test]
async fn unchanged_is_final_and_does_not_commit(#[case] full: bool) {
    let (mut dataset, optimizer) = fixture(Mode::Unchanged, full).await;
    if !full {
        unreadable_unindexed_fragment(&dataset).await;
    }
    let version = dataset.version_id();
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 1);
    assert_eq!(dataset.version_id(), version);
    assert_eq!(dataset.load_indices().await.unwrap()[0].uuid, uuid);
}

#[tokio::test]
async fn coverage_is_not_expanded_to_every_unindexed_fragment() {
    let (mut dataset, optimizer) = fixture(Mode::Copy, false).await;
    let old = dataset.load_indices().await.unwrap()[0].clone();
    let options = OptimizeOptions::default().transaction_properties(
        std::collections::HashMap::from([("source".into(), "custom-optimize".into())]),
    );
    let version = dataset.version_id();
    dataset.optimize_indices(&options).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_ne!(indices[0].uuid, old.uuid);
    assert_eq!(indices[0].fragment_bitmap, old.fragment_bitmap);
    assert_eq!(indices[0].dataset_version, old.dataset_version);
    assert_eq!(dataset.version_id(), version + 1);
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 1);
    let transaction = dataset
        .read_version_transaction(dataset.version_id())
        .await
        .unwrap()
        .transaction
        .unwrap();
    assert_eq!(
        transaction.transaction_properties.as_ref().unwrap()["source"],
        "custom-optimize"
    );
    let mut indexed = dataset.scan();
    indexed.filter("value >= 3").unwrap();
    let result = indexed.try_into_batch().await.unwrap();
    let mut scan = dataset.scan();
    scan.filter("value >= 3").unwrap().use_scalar_index(false);
    assert_eq!(result, scan.try_into_batch().await.unwrap());
    assert_eq!(result.num_rows(), 2);
}

#[tokio::test]
async fn custom_replacement_does_not_read_unindexed_data() {
    let (mut dataset, _) = fixture(Mode::Copy, false).await;
    unreadable_unindexed_fragment(&dataset).await;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(
        dataset.load_indices().await.unwrap()[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .len(),
        1
    );
}

#[rstest]
#[case::foreign_replacement(Mode::ForeignReplacement, "does not belong")]
#[case::uuid_reuse(Mode::ReuseUuid, "fresh UUIDs")]
#[case::nonlive(Mode::NonliveCoverage, "only current fragments")]
#[case::retained_overlap(Mode::OverlapRetained, "overlapping fragment coverage")]
#[case::wrong_field(Mode::WrongField, "field declaration")]
#[case::wrong_type(Mode::WrongType, "details type")]
#[case::future_version(Mode::FutureVersion, "future dataset version")]
#[tokio::test]
async fn invalid_output_never_changes_the_manifest(#[case] mode: Mode, #[case] message: &str) {
    let (mut dataset, _) = fixture(mode, false).await;
    let version = dataset.version_id();
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let error = dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(dataset.version_id(), version);
    assert_eq!(dataset.load_indices().await.unwrap()[0].uuid, uuid);
}

#[tokio::test]
async fn explicit_replacement_can_leave_fragments_unindexed() {
    let (mut dataset, _) = fixture(Mode::LoseCoverage, false).await;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert!(
        dataset.load_indices().await.unwrap()[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .is_empty()
    );
    let mut indexed = dataset.scan();
    indexed.filter("value >= 1").unwrap();
    let mut scan = dataset.scan();
    scan.filter("value >= 1").unwrap().use_scalar_index(false);
    assert_eq!(
        indexed.try_into_batch().await.unwrap(),
        scan.try_into_batch().await.unwrap()
    );
}

#[tokio::test]
async fn explicit_append_cannot_replace_a_populated_segment() {
    let (mut dataset, _) = fixture(Mode::Copy, false).await;
    let version = dataset.version_id();
    let error = dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("num_indices_to_merge"));
    assert_eq!(dataset.version_id(), version);
}

#[tokio::test]
async fn explicit_append_can_add_coverage_without_replacing_old_segments() {
    let (mut dataset, _) = fixture(Mode::AddUnindexed, false).await;
    let old = dataset.load_indices().await.unwrap()[0].clone();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let after = dataset.load_indices().await.unwrap();
    assert_eq!(after.len(), 2);
    assert!(after.iter().any(|segment| segment.uuid == old.uuid));
    let covered = after
        .iter()
        .fold(roaring::RoaringBitmap::new(), |covered, segment| {
            covered | segment.fragment_bitmap.as_ref().unwrap()
        });
    assert_eq!(covered, **dataset.fragment_bitmap());
    let mut indexed = dataset.scan();
    indexed.filter("value >= 2").unwrap();
    let mut scan = dataset.scan();
    scan.filter("value >= 2").unwrap().use_scalar_index(false);
    assert_eq!(
        indexed.try_into_batch().await.unwrap(),
        scan.try_into_batch().await.unwrap()
    );
}

#[rstest]
#[case::retrain(OptimizeOptions::retrain())]
#[case::task_bound(OptimizeOptions::default().max_rows_per_task(Some(2)))]
#[case::segment_bound(OptimizeOptions::default().max_rows_per_segment(Some(2)))]
#[tokio::test]
async fn unsupported_modes_are_rejected_before_the_callback(#[case] options: OptimizeOptions) {
    let (mut dataset, optimizer) = fixture(Mode::Copy, false).await;
    let version = dataset.version_id();
    let error = dataset.optimize_indices(&options).await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("retrain or distributed row bounds")
    );
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 0);
    assert_eq!(dataset.version_id(), version);
}

#[tokio::test]
async fn distributed_planning_does_not_fall_back_to_the_builtin_algorithm() {
    let (dataset, optimizer) = fixture(Mode::Copy, false).await;
    let error = plan_optimize_indices(&dataset, &OptimizeOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("distributed custom optimization is not supported")
    );
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn concurrent_append_is_rebased_without_expanding_coverage() {
    let (mut dataset, _) = fixture(Mode::ConcurrentAppend, false).await;
    let coverage = dataset.load_indices().await.unwrap()[0]
        .fragment_bitmap
        .clone();
    let version = dataset.version_id();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(dataset.version_id(), version + 2);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
    assert_eq!(
        dataset.load_indices().await.unwrap()[0].fragment_bitmap,
        coverage
    );
}

#[tokio::test]
async fn multiple_outputs_share_one_commit() {
    let (mut dataset, optimizer) = fixture(Mode::Copy, false).await;
    let second_id = dataset.fragments()[1].id as u32;
    let segment = dataset
        .create_index_builder(&["value"], IndexType::BTree, &ScalarIndexParams::default())
        .name("custom".to_string())
        .replace(true)
        .fragments(vec![second_id])
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("custom", "value", vec![segment])
        .await
        .unwrap();
    let before = dataset.load_indices().await.unwrap();
    assert_eq!(before.len(), 2);
    let version = dataset.version_id();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let after = dataset.load_indices().await.unwrap();
    assert_eq!(dataset.version_id(), version + 1);
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 1);
    assert_eq!(after.len(), 2);
    for output in after.iter() {
        assert!(before.iter().all(|input| input.uuid != output.uuid));
    }
    let mut indexed = dataset.scan();
    indexed.filter("value >= 2").unwrap();
    let mut scan = dataset.scan();
    scan.filter("value >= 2").unwrap().use_scalar_index(false);
    assert_eq!(
        indexed.try_into_batch().await.unwrap(),
        scan.try_into_batch().await.unwrap()
    );
}

#[tokio::test]
async fn index_names_and_builtin_optimization_still_work() {
    let (mut dataset, optimizer) = fixture(Mode::Error, false).await;
    let id = dataset.fragments()[0].id as u32;
    dataset
        .create_index_builder(&["value"], IndexType::Bitmap, &ScalarIndexParams::default())
        .name("ordinary".to_string())
        .fragments(vec![id])
        .await
        .unwrap();
    let custom_uuid = dataset.load_indices_by_name("custom").await.unwrap()[0].uuid;
    dataset
        .optimize_indices(&OptimizeOptions::default().index_names(vec!["ordinary".to_string()]))
        .await
        .unwrap();
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        dataset.load_indices_by_name("ordinary").await.unwrap()[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    let version = dataset.version_id();
    unreadable_unindexed_fragment(&dataset).await;
    let error = dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("test maintainer refused"));
    assert_eq!(dataset.version_id(), version);
    assert_eq!(
        dataset.load_indices_by_name("custom").await.unwrap()[0].uuid,
        custom_uuid
    );
}

#[tokio::test]
async fn registration_rejects_duplicates() {
    let optimizer = Arc::new(CopyOptimizer {
        mode: Mode::Unchanged,
        calls: AtomicUsize::new(0),
    });
    let mut session = Session::default();
    session
        .register_scalar_index_optimizer::<pbold::BTreeIndexDetails>(optimizer.clone())
        .unwrap();
    let error = session
        .register_scalar_index_optimizer::<pbold::BTreeIndexDetails>(optimizer)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("already registered"));
}

#[tokio::test]
async fn another_session_still_uses_builtin_optimization() {
    let (mut custom, optimizer) = fixture(Mode::Unchanged, false).await;
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut ordinary = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let first_id = ordinary.fragments()[0].id as u32;
    ordinary
        .create_index_builder(&["value"], IndexType::BTree, &ScalarIndexParams::default())
        .name("custom".to_string())
        .fragments(vec![first_id])
        .await
        .unwrap();
    custom
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    ordinary
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(optimizer.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        custom.load_indices().await.unwrap()[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        ordinary.load_indices().await.unwrap()[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .len(),
        2
    );
}
