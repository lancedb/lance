// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::Float32Type;
use arrow_array::{Array, RecordBatch};
use lance_core::{Result, ROW_ID};
use lance_index::vector::{
    quantizer::Quantization,
    sq::{builder::SQBuildParams, storage::ScalarQuantizationStorage, ScalarQuantizer},
};
use lance_linalg::{distance::MetricType, kernels::normalize_fsl};

use crate::{index::vector::utils::maybe_sample_training_data, Dataset};

pub(super) async fn build_sq_model(
    dataset: &Dataset,
    column: &str,
    metric_type: MetricType,
    params: &SQBuildParams,
) -> Result<ScalarQuantizer> {
    log::info!("Start to train SQ code: SQ{}", params.num_bits);
    let expected_sample_size = 2usize.pow(params.num_bits as u32) * params.sample_rate;
    log::info!(
        "Loading training data for SQ. Sample size: {}",
        expected_sample_size
    );
    let start = std::time::Instant::now();
    let mut training_data =
        maybe_sample_training_data(dataset, column, expected_sample_size).await?;
    log::info!(
        "Finished loading training data in {:02} seconds",
        start.elapsed().as_secs_f32()
    );

    log::info!(
        "starting to compute partitions for SQ training, sample size: {}",
        training_data.value_length()
    );

    if metric_type == MetricType::Cosine {
        log::info!("Normalize training data for SQ training: Cosine");
        training_data = normalize_fsl(&training_data)?;
    }

    log::info!("Start train SQ: params={:#?}", params);
    let sq = params.build(&training_data, MetricType::L2)?;
    log::info!(
        "Trained SQ{}[{:?}] in: {} seconds",
        sq.num_bits(),
        sq.bounds(),
        start.elapsed().as_secs_f32()
    );
    Ok(sq)
}

pub fn build_sq_storage(
    metric_type: MetricType,
    row_ids: Arc<dyn Array>,
    vectors: Arc<dyn Array>,
    sq: ScalarQuantizer,
) -> Result<ScalarQuantizationStorage> {
    let code_column = sq.transform::<Float32Type>(vectors.as_ref())?;
    std::mem::drop(vectors);

    let pq_batch = RecordBatch::try_from_iter_with_nullable(vec![
        (ROW_ID, row_ids, true),
        (sq.column(), code_column, false),
    ])?;
    let store = ScalarQuantizationStorage::new(sq.num_bits(), metric_type, sq.bounds().to_vec(), pq_batch)?;

    Ok(store)
}
