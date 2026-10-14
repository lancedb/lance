// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset-aware PCA training.

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
};
use arrow_schema::DataType;
use lance_index::progress::IndexBuildProgress;
use lance_index::vector::pca::{PcaModel, PcaParams, accumulate_centered_scatter};

use super::utils::{get_column_from_batch, get_vector_dim, sample_valid_vector_offsets_seeded};
use crate::dataset::Dataset;
use crate::{Error, Result};

const PCA_TRAINING_BATCH_SIZE: usize = 4096;

/// Shared PCA / IVF / RQ shape comparison used by both writers and readers.
pub enum PcaComponentMismatch {
    InputDimension { actual: usize, expected: usize },
    CentroidType { actual: DataType },
    OutputDimension { actual: usize, expected: usize },
    RqDimension { actual: usize, expected: usize },
}

pub fn pca_component_mismatch(
    model: &PcaModel,
    expected_input_dimension: usize,
    ivf_dimension: usize,
    centroid_type: &DataType,
    rq_rotated_dimension: usize,
) -> Option<PcaComponentMismatch> {
    if model.input_dimension() != expected_input_dimension {
        return Some(PcaComponentMismatch::InputDimension {
            actual: model.input_dimension(),
            expected: expected_input_dimension,
        });
    }
    if centroid_type != &DataType::Float32 {
        return Some(PcaComponentMismatch::CentroidType {
            actual: centroid_type.clone(),
        });
    }
    if model.output_dimension() != ivf_dimension {
        return Some(PcaComponentMismatch::OutputDimension {
            actual: model.output_dimension(),
            expected: ivf_dimension,
        });
    }
    if rq_rotated_dimension != model.output_dimension() {
        return Some(PcaComponentMismatch::RqDimension {
            actual: rq_rotated_dimension,
            expected: model.output_dimension(),
        });
    }
    None
}

/// Trains a PCA model from bounded, repeatable dataset reads.
#[derive(Debug)]
pub(super) struct PcaTrainer {
    dataset: Dataset,
    column: String,
    fragment_ids: Option<Vec<u32>>,
    progress: Arc<dyn IndexBuildProgress>,
}

impl PcaTrainer {
    /// Creates a trainer for one vector column and optional fragment selection.
    ///
    /// Fragment-limited PCA training is currently rejected explicitly during
    /// sampling because dataset-wide logical offsets cannot represent it.
    pub(super) fn new(
        dataset: Dataset,
        column: impl Into<String>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Self {
        Self {
            dataset,
            column: column.into(),
            fragment_ids,
            progress,
        }
    }

    /// Samples valid offsets and trains using two streaming passes.
    pub(super) async fn train_dataset(&self, params: &PcaParams) -> Result<PcaModel> {
        let input_dimension = get_vector_dim(self.dataset.schema(), &self.column)?;
        params.validate(input_dimension, params.sample_size)?;
        let sample = sample_valid_vector_offsets_seeded(
            &self.dataset,
            &self.column,
            params.sample_size,
            params.sampling_seed,
            self.fragment_ids.as_deref(),
            self.progress.clone(),
        )
        .await?;
        let sample_count = sample.row_offsets.len();
        params.validate_streaming_memory_budget(
            input_dimension,
            sample_count,
            PCA_TRAINING_BATCH_SIZE.min(sample_count),
        )?;

        let projection = Arc::new(self.dataset.schema().project(&[&self.column])?);
        let center_total = sample_count
            .checked_mul(2)
            .ok_or_else(|| Error::invalid_input("PCA center progress total overflows usize"))?;
        self.progress
            .stage_start("center_pca", Some(center_total as u64), "rows")
            .await?;

        let mut sums = vec![0.0_f64; input_dimension];
        let mut corrections = vec![0.0_f64; input_dimension];
        let mut completed = 0_usize;
        for offsets in sample.row_offsets.chunks(PCA_TRAINING_BATCH_SIZE) {
            let values = self
                .read_values(offsets, projection.clone(), input_dimension)
                .await?;
            for row in values.chunks_exact(input_dimension) {
                for (column, value) in row.iter().enumerate() {
                    let adjusted = *value - corrections[column];
                    let next = sums[column] + adjusted;
                    corrections[column] = (next - sums[column]) - adjusted;
                    sums[column] = next;
                }
            }
            completed = completed.checked_add(offsets.len()).ok_or_else(|| {
                Error::invalid_input("PCA center progress completed rows overflows usize")
            })?;
            self.progress
                .stage_progress("center_pca", completed as u64)
                .await?;
        }
        let denominator = sample_count as f64;
        for value in &mut sums {
            *value /= denominator;
            if !value.is_finite() {
                return Err(Error::invalid_input(
                    "PCA streaming mean calculation produced a non-finite value".to_string(),
                ));
            }
        }
        let mean = sums;

        let statistics = if sample_count >= input_dimension {
            let matrix_len = input_dimension
                .checked_mul(input_dimension)
                .ok_or_else(|| Error::invalid_input("PCA scatter matrix shape overflows usize"))?;
            let mut scatter = vec![0.0_f64; matrix_len];
            for offsets in sample.row_offsets.chunks(PCA_TRAINING_BATCH_SIZE) {
                let values = self
                    .read_values(offsets, projection.clone(), input_dimension)
                    .await?;
                accumulate_centered_scatter(&values, &mean, &mut scatter)?;
                completed = completed.checked_add(offsets.len()).ok_or_else(|| {
                    Error::invalid_input("PCA center progress completed rows overflows usize")
                })?;
                self.progress
                    .stage_progress("center_pca", completed as u64)
                    .await?;
            }
            PcaStatistics::Scatter(scatter)
        } else {
            let centered_len = sample_count
                .checked_mul(input_dimension)
                .ok_or_else(|| Error::invalid_input("PCA centered matrix shape overflows usize"))?;
            let mut centered = Vec::with_capacity(centered_len);
            for offsets in sample.row_offsets.chunks(PCA_TRAINING_BATCH_SIZE) {
                let values = self
                    .read_values(offsets, projection.clone(), input_dimension)
                    .await?;
                for row in values.chunks_exact(input_dimension) {
                    centered.extend(row.iter().zip(&mean).map(|(value, mean)| value - mean));
                }
                completed = completed.checked_add(offsets.len()).ok_or_else(|| {
                    Error::invalid_input("PCA center progress completed rows overflows usize")
                })?;
                self.progress
                    .stage_progress("center_pca", completed as u64)
                    .await?;
            }
            PcaStatistics::Centered(centered)
        };
        self.progress.stage_complete("center_pca").await?;

        self.progress
            .stage_start("eigendecompose_pca", Some(1), "work_units")
            .await?;
        let model = match statistics {
            PcaStatistics::Scatter(scatter) => {
                PcaModel::train_from_scatter(&mean, sample_count, &scatter, params)?
            }
            PcaStatistics::Centered(centered) => {
                PcaModel::train_from_centered(&mean, sample_count, &centered, params)?
            }
        };
        self.progress
            .stage_progress("eigendecompose_pca", 1)
            .await?;
        self.progress.stage_complete("eigendecompose_pca").await?;
        Ok(model)
    }

    async fn read_values(
        &self,
        offsets: &[u64],
        projection: Arc<lance_core::datatypes::Schema>,
        expected_dimension: usize,
    ) -> Result<Vec<f64>> {
        let batch = self.dataset.take(offsets, projection).await?;
        let array = get_column_from_batch(&batch, &self.column)?;
        let vectors = array.as_fixed_size_list();
        vectors_as_f64(vectors, expected_dimension)
    }
}

enum PcaStatistics {
    Scatter(Vec<f64>),
    Centered(Vec<f64>),
}

fn vectors_as_f64(vectors: &FixedSizeListArray, expected_dimension: usize) -> Result<Vec<f64>> {
    let dimension = usize::try_from(vectors.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "PCA vector value_length={} is not a valid dimension",
            vectors.value_length()
        ))
    })?;
    if dimension != expected_dimension {
        return Err(Error::invalid_input(format!(
            "PCA vector dimension={dimension} does not match expected input_dimension={expected_dimension}"
        )));
    }
    let values_start = vectors
        .offset()
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA vector value offset overflows usize"))?;
    let value_count = vectors
        .len()
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA vector values length overflows usize"))?;
    let values_end = values_start
        .checked_add(value_count)
        .ok_or_else(|| Error::invalid_input("PCA vector value range overflows usize"))?;
    let values = vectors.values();
    if vectors.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PCA sampled vectors unexpectedly contain {} top-level nulls",
            vectors.null_count()
        )));
    }
    if values_end > values.len() {
        return Err(Error::invalid_input(format!(
            "PCA vector value range [{values_start}, {values_end}) exceeds child values length {}",
            values.len()
        )));
    }
    if (values_start..values_end).any(|index| values.is_null(index)) {
        return Err(Error::invalid_input(
            "PCA sampled vectors unexpectedly contain null child values".to_string(),
        ));
    }

    let converted = match values.data_type() {
        DataType::Float16 => values.as_primitive::<Float16Type>().values()
            [values_start..values_end]
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect(),
        DataType::Float32 => values.as_primitive::<Float32Type>().values()
            [values_start..values_end]
            .iter()
            .map(|value| f64::from(*value))
            .collect(),
        DataType::Float64 => {
            values.as_primitive::<Float64Type>().values()[values_start..values_end].to_vec()
        }
        data_type => {
            return Err(Error::invalid_input(format!(
                "PCA training requires Float16, Float32, or Float64 vector values, got {data_type}"
            )));
        }
    };
    if converted.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input(
            "PCA sampled vectors unexpectedly contain NaN or infinity".to_string(),
        ));
    }
    Ok(converted)
}

/// Builds a full IVF_RQ index with a PCA pre-transform.
///
/// Kept off the shared vector-index builder so non-PCA builds keep the original
/// `build_vector_index` signature and control flow.
pub async fn build_pca_ivf_rq_index(
    dataset: &Dataset,
    column: &str,
    uuid: uuid::Uuid,
    params: &super::VectorIndexParams,
    frag_reuse_index: Option<Arc<lance_index::frag_reuse::CompactFragReuseIndex>>,
    full_new_build: bool,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<super::builder::VectorIndexBuildSummary> {
    let (vector_type, _) = super::utils::get_vector_type(dataset.schema(), column)?;
    let input_dimension = get_vector_dim(dataset.schema(), column)?;
    super::validate_pca_build_supported(params, &vector_type, input_dimension, full_new_build)?;
    let pca_params = params.pca_params().cloned().ok_or_else(|| {
        Error::invalid_input("PCA IVF_RQ build requires PCA parameters".to_string())
    })?;
    let (_element_type, _index_type, ivf_params, shuffler, _temp_dir) =
        super::prepare_vector_segment_build(
            dataset,
            column,
            params,
            progress.clone(),
            "Build PCA IVF_RQ",
            false,
            None,
        )
        .await?;
    let super::StageParams::RQ(rq_params) = &params.stages[1] else {
        return Err(Error::index(
            "PCA IVF_RQ requires RQ as the second stage".to_string(),
        ));
    };

    super::builder::IvfIndexBuilder::<
        lance_index::vector::flat::index::FlatIndex,
        lance_index::vector::bq::builder::RabitQuantizer,
    >::new(
        dataset.clone(),
        column.to_owned(),
        dataset.indices_dir().join(uuid.to_string()),
        params.metric_type,
        shuffler,
        Some(ivf_params),
        Some(rq_params.clone()),
        (),
        frag_reuse_index,
    )?
    .with_pca_params(pca_params)
    .with_transpose(!params.skip_transpose)
    .with_progress(progress)
    .build()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::progress::noop_progress;

    use crate::dataset::InsertBuilder;

    fn training_vectors() -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                2.0, 0.0, 1.0, -2.0, 0.0, -1.0, 0.0, 3.0, 1.0, 0.0, -3.0, -1.0, 1.0, 1.0, 2.0,
                -1.0, -1.0, -2.0, 3.0, 2.0, 0.5, -3.0, -2.0, -0.5,
            ]),
            3,
        )
        .unwrap()
    }

    async fn dataset_from_vectors(vectors: FixedSizeListArray) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            vectors.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(vectors)]).unwrap();
        InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_dataset_trainer_matches_in_memory_training() {
        let dataset = dataset_from_vectors(training_vectors()).await;
        let params = PcaParams::new(2).with_sample_size(8);
        let expected = PcaModel::train(&training_vectors(), &params).unwrap();
        let actual = PcaTrainer::new(dataset, "vector", None, noop_progress())
            .train_dataset(&params)
            .await
            .unwrap();

        for (actual, expected) in actual.mean().values().iter().zip(expected.mean().values()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1.0e-6);
        }
        for (actual, expected) in actual
            .transform_matrix()
            .values()
            .iter()
            .zip(expected.transform_matrix().values())
        {
            assert_relative_eq!(*actual, *expected, epsilon = 1.0e-5);
        }
    }

    #[tokio::test]
    async fn test_dataset_trainer_enforces_streaming_memory_budget() {
        let dataset = dataset_from_vectors(training_vectors()).await;
        let params = PcaParams::new(2)
            .with_sample_size(8)
            .with_max_training_memory_bytes(1);
        let error = PcaTrainer::new(dataset, "vector", None, noop_progress())
            .train_dataset(&params)
            .await
            .expect_err("streaming training must enforce its memory budget");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("M_batch="));
    }

    #[tokio::test]
    async fn test_dataset_trainer_requires_output_dimension_plus_one_samples() {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
            3,
        )
        .unwrap();
        let dataset = dataset_from_vectors(vectors).await;
        let params = PcaParams::new(2).with_sample_size(3);
        let error = PcaTrainer::new(dataset, "vector", None, noop_progress())
            .train_dataset(&params)
            .await
            .expect_err("two samples cannot train two output dimensions");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("output_dimension+1=3"));
        assert!(error.to_string().contains("got 2"));
    }
}
