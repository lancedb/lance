// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod maintenance;

use std::num::NonZero;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::index::DatasetIndexRemapperOptions;
use lance::dataset::optimize::{CompactionOptions, CompactionPlan, TaskData, commit_compaction};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_io::utils::tracking_store::IoOperation;
use lance_linalg::distance::MetricType;
use lance_table::format::{DataFile, Fragment};

fn fragment(id: u64, base_id: Option<u32>) -> Fragment {
    let mut fragment = Fragment::new(id);
    fragment.files.push(DataFile::new(
        format!("{id}.lance"),
        vec![0],
        vec![0],
        2,
        0,
        NonZero::new(1),
        base_id,
    ));
    fragment.physical_rows = Some(1);
    fragment
}

#[test]
fn no_stable_replay_uses_physical_fragment_order_within_policy_group() {
    let mut fragments = (0..8)
        .map(|id| fragment(id, Some(1)))
        .chain(std::iter::once(fragment(8, None)))
        .collect::<Vec<_>>();
    fragments.sort_by(maintenance::policy_fragment_order);

    assert_eq!(
        fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![8, 0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        maintenance::fragments_for_native_compaction(&fragments, true)
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        (0..9).collect::<Vec<_>>()
    );
    assert_eq!(
        maintenance::fragments_for_native_compaction(&fragments, false)
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![8, 0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn aggregate_delete_control_request_is_metadata() {
    assert_eq!(
        maintenance::aggregate_control_path_category(IoOperation::Delete, "", 0),
        Some("metadata")
    );
    assert_eq!(
        maintenance::aggregate_control_path_category(IoOperation::Delete, "data/fragment.lance", 0,),
        None
    );
}

#[tokio::test]
async fn no_stable_single_group_consolidates_vector_segments_before_compaction() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    let make_reader = || {
        gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(Dimension::from(8)))
            .into_reader_rows(RowCount::from(128), BatchCount::from(1))
    };
    let mut dataset = Dataset::write(
        make_reader(),
        uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            max_rows_per_file: 128,
            enable_stable_row_ids: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();
    dataset.append(make_reader(), None).await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    let indices_before = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_before
            .iter()
            .filter(|index| index.name == "vector_idx")
            .count(),
        2
    );
    let fragments_before = dataset.manifest().fragments.as_ref().clone();
    let source_version = dataset.version().version;
    assert!(
        maintenance::consolidate_no_stable_index_segments(&mut dataset, indices_before.as_ref(),)
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(dataset.version().version, source_version + 1);
    assert_eq!(
        dataset.manifest().fragments.as_ref(),
        fragments_before.as_slice()
    );
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

    let options = CompactionOptions {
        target_rows_per_fragment: 256,
        max_rows_per_group: 128,
        num_threads: Some(1),
        ..Default::default()
    };
    let plan = CompactionPlan {
        tasks: vec![TaskData {
            fragments: dataset.manifest().fragments.as_ref().clone(),
            v2_3_plan: None,
        }],
        read_version: dataset.version().version,
        options,
        planning_metrics: Default::default(),
    };
    let mut completed = Vec::new();
    for task in plan.compaction_tasks() {
        completed.push(task.execute(&dataset).await.unwrap());
    }
    commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        plan.options(),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 256);
}
