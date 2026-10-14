// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::FixedSizeListArray;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_index::vector::pca::PcaModel;
use lance_linalg::distance::DistanceType;

use super::utils::{filter_finite_training_data, get_vector_dim, maybe_sample_training_data};
use crate::Dataset;

/// A trained pre-transform as seen by IVF training and partition assignment.
pub trait TrainingVectorTransform: Send + Sync {
    fn input_dimension(&self) -> usize;
    fn output_dimension(&self) -> usize;
    /// Maps original-space vectors to Float32 vectors of `output_dimension`.
    fn transform_vectors(&self, vectors: &FixedSizeListArray) -> Result<FixedSizeListArray>;
}

impl TrainingVectorTransform for PcaModel {
    fn input_dimension(&self) -> usize {
        Self::input_dimension(self)
    }

    fn output_dimension(&self) -> usize {
        Self::output_dimension(self)
    }

    fn transform_vectors(&self, vectors: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        self.transform(vectors)
    }
}

/// Training vectors for IVF/RQ. A pre-transform wraps a raw source so downstream stages only see the reduced dimension.
#[async_trait]
pub(super) trait VectorTrainingDataSource: Send + Sync {
    async fn sample(&self, sample_size: usize) -> Result<FixedSizeListArray>;

    fn dimension(&self) -> Result<usize>;
}

/// Samples original-space vectors from the dataset.
pub(super) struct DatasetVectorSource<'a> {
    dataset: &'a Dataset,
    column: &'a str,
    fragment_ids: Option<&'a [u32]>,
}

impl<'a> DatasetVectorSource<'a> {
    pub(super) fn new(
        dataset: &'a Dataset,
        column: &'a str,
        fragment_ids: Option<&'a [u32]>,
    ) -> Self {
        Self {
            dataset,
            column,
            fragment_ids,
        }
    }

    /// Samples like [`VectorTrainingDataSource::sample`], normalizing cosine
    /// vectors before dropping non-finite rows.
    pub(super) async fn sample_for_distance(
        &self,
        sample_size: usize,
        distance_type: DistanceType,
    ) -> Result<FixedSizeListArray> {
        let training_data =
            maybe_sample_training_data(self.dataset, self.column, sample_size, self.fragment_ids)
                .await?;
        let training_data = if distance_type == DistanceType::Cosine {
            lance_linalg::kernels::normalize_fsl_owned(training_data)?
        } else {
            training_data
        };
        filter_finite_training_data(training_data)
    }
}

#[async_trait]
impl VectorTrainingDataSource for DatasetVectorSource<'_> {
    async fn sample(&self, sample_size: usize) -> Result<FixedSizeListArray> {
        let training_data =
            maybe_sample_training_data(self.dataset, self.column, sample_size, self.fragment_ids)
                .await?;
        filter_finite_training_data(training_data)
    }

    fn dimension(&self) -> Result<usize> {
        get_vector_dim(self.dataset.schema(), self.column)
    }
}

/// Applies a trained pre-transform to another source's samples.
pub(super) struct PreTransformVectorSource<'a, S> {
    source: S,
    transform: &'a dyn TrainingVectorTransform,
}

impl<'a, S: VectorTrainingDataSource> PreTransformVectorSource<'a, S> {
    pub(super) fn try_new(source: S, transform: &'a dyn TrainingVectorTransform) -> Result<Self> {
        let source_dimension = source.dimension()?;
        if transform.input_dimension() != source_dimension {
            return Err(Error::invalid_input(format!(
                "pre-transform input_dimension={} does not match source vector dimension={source_dimension}",
                transform.input_dimension()
            )));
        }
        Ok(Self { source, transform })
    }
}

#[async_trait]
impl<S: VectorTrainingDataSource> VectorTrainingDataSource for PreTransformVectorSource<'_, S> {
    async fn sample(&self, sample_size: usize) -> Result<FixedSizeListArray> {
        self.transform
            .transform_vectors(&self.source.sample(sample_size).await?)
    }

    fn dimension(&self) -> Result<usize> {
        Ok(self.transform.output_dimension())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch,
        RecordBatchIterator, cast::AsArray,
    };
    use arrow_schema::DataType;
    use async_trait::async_trait;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::pca::{PcaModel, PcaParams};

    use super::*;

    struct InMemorySource {
        data: FixedSizeListArray,
    }

    #[async_trait]
    impl VectorTrainingDataSource for InMemorySource {
        async fn sample(&self, _sample_size: usize) -> Result<FixedSizeListArray> {
            Ok(self.data.clone())
        }

        fn dimension(&self) -> Result<usize> {
            Ok(self.data.value_length() as usize)
        }
    }

    #[tokio::test]
    async fn dataset_source_reports_dimension_and_filters_non_finite_vectors() {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0,
                2.0,
                3.0,
                f32::NAN,
                4.0,
                5.0,
                6.0,
                f32::INFINITY,
                7.0,
            ]),
            3,
        )
        .unwrap();
        let batch =
            RecordBatch::try_from_iter([("vector", Arc::new(vectors) as ArrayRef)]).unwrap();
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        let source = DatasetVectorSource::new(&dataset, "vector", None);

        assert_eq!(source.dimension().unwrap(), 3);
        let sampled = source.sample(3).await.unwrap();
        assert_eq!(sampled.len(), 1);
        assert_eq!(
            sampled
                .values()
                .as_primitive::<arrow_array::types::Float32Type>(),
            &Float32Array::from(vec![1.0, 2.0, 3.0])
        );
    }

    #[tokio::test]
    async fn pre_transform_source_reports_output_dimension_and_float32_values() {
        let training_data = FixedSizeListArray::try_new_from_values(
            Float64Array::from(vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0,
            ]),
            3,
        )
        .unwrap();
        let model = Arc::new(PcaModel::train(&training_data, &PcaParams::new(2)).unwrap());
        let source = PreTransformVectorSource::try_new(
            InMemorySource {
                data: training_data,
            },
            model.as_ref(),
        )
        .unwrap();

        assert_eq!(source.dimension().unwrap(), 2);
        let transformed = source.sample(4).await.unwrap();
        assert_eq!(transformed.value_type(), DataType::Float32);
        assert_eq!(transformed.value_length(), 2);
        assert_eq!(transformed.len(), 4);
    }

    #[test]
    fn pre_transform_source_validates_input_dimension() {
        let model_data = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0,
            ]),
            3,
        )
        .unwrap();
        let model = PcaModel::train(&model_data, &PcaParams::new(2)).unwrap();
        let source_data =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0; 16]), 4).unwrap();

        let error = PreTransformVectorSource::try_new(InMemorySource { data: source_data }, &model)
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("input_dimension=3 does not match source vector dimension=4")
        );
    }
}
