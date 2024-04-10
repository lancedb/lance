// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Result;
use lance_index::vector::sq::{builder::SQBuildParams, ScalarQuantizer};
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
