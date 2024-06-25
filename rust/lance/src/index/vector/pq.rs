// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::{any::Any, collections::HashMap};

use arrow::compute::concat;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::UInt32Array;
use arrow_array::{
    cast::{as_primitive_array, AsArray},
    Array, FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::take::take;
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::ROW_ID;
use lance_core::{utils::address::RowAddress, ROW_ID_FIELD};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::storage::ProductQuantizationStorage;
use lance_index::vector::quantizer::{Quantization, QuantizationType, Quantizer};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    vector::{pq::ProductQuantizer, Query, DIST_COL},
    Index, IndexType,
};
use lance_io::{traits::Reader, utils::read_fixed_stride_array};
use lance_linalg::distance::{DistanceType, MetricType};
use log::info;
use roaring::RoaringBitmap;
use serde_json::json;
use snafu::{location, Location};
use tracing::{instrument, span, Level};

// Re-export
pub use lance_index::vector::pq::{PQBuildParams, ProductQuantizerImpl};
use lance_linalg::kernels::normalize_fsl;

use super::VectorIndex;
use crate::index::prefilter::PreFilter;
use crate::index::vector::utils::maybe_sample_training_data;
use crate::{arrow::*, Dataset};
use crate::{Error, Result};

/// Product Quantization Index.
///
#[derive(Clone)]
pub struct PQIndex {
    /// Product quantizer.
    pub pq: Arc<dyn ProductQuantizer>,

    /// PQ code
    pub code: Option<Arc<UInt8Array>>,

    /// ROW Id used to refer to the actual row in dataset.
    pub row_ids: Option<Arc<UInt64Array>>,

    /// Metric type.
    metric_type: MetricType,
}

impl DeepSizeOf for PQIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.pq.deep_size_of_children(context)
            + self
                .code
                .as_ref()
                .map(|code| code.get_array_memory_size())
                .unwrap_or(0)
            + self
                .row_ids
                .as_ref()
                .map(|row_ids| row_ids.get_array_memory_size())
                .unwrap_or(0)
    }
}

impl std::fmt::Debug for PQIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PQ(m={}, nbits={}, {})",
            self.pq.num_sub_vectors(),
            self.pq.num_bits(),
            self.metric_type
        )
    }
}

impl PQIndex {
    /// Load a PQ index (page) from the disk.
    pub(crate) fn new(pq: Arc<dyn ProductQuantizer>, metric_type: MetricType) -> Self {
        Self {
            code: None,
            row_ids: None,
            pq,
            metric_type,
        }
    }

    /// Filter the row id and PQ code arrays based on the pre-filter.
    fn filter_arrays(
        pre_filter: &dyn PreFilter,
        code: Arc<UInt8Array>,
        row_ids: Arc<UInt64Array>,
        num_sub_vectors: i32,
    ) -> Result<(Arc<UInt8Array>, Arc<UInt64Array>)> {
        let indices_to_keep = pre_filter.filter_row_ids(Box::new(row_ids.values().iter()));
        let indices_to_keep = UInt64Array::from(indices_to_keep);

        let row_ids = take(row_ids.as_ref(), &indices_to_keep, None)?;
        let row_ids = Arc::new(as_primitive_array(&row_ids).clone());

        let code = FixedSizeListArray::try_new_from_values(code.as_ref().clone(), num_sub_vectors)
            .unwrap();
        let code = take(&code, &indices_to_keep, None)?;
        let code = as_fixed_size_list_array(&code).values().clone();
        let code = Arc::new(as_primitive_array(&code).clone());

        Ok((code, row_ids))
    }
}

#[async_trait]
impl Index for PQIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Ok(self)
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(json!({
            "index_type": "PQ",
            "nbits": self.pq.num_bits(),
            "num_sub_vectors": self.pq.num_sub_vectors(),
            "dimension": self.pq.dimension(),
            "metric_type": self.metric_type.to_string(),
        }))
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        if let Some(row_ids) = &self.row_ids {
            let mut frag_ids = row_ids
                .values()
                .iter()
                .map(|&row_id| RowAddress::new_from_id(row_id).fragment_id())
                .collect::<Vec<_>>();
            frag_ids.sort();
            frag_ids.dedup();
            Ok(RoaringBitmap::from_sorted_iter(frag_ids).unwrap())
        } else {
            Err(Error::Index {
                message: "PQIndex::caclulate_included_frags: PQ is not initialized".to_string(),
                location: location!(),
            })
        }
    }
}

#[async_trait]
impl VectorIndex for PQIndex {
    /// Search top-k nearest neighbors for `key` within one PQ partition.
    ///
    #[instrument(level = "debug", skip_all, name = "PQIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<dyn PreFilter>) -> Result<RecordBatch> {
        if self.code.is_none() || self.row_ids.is_none() {
            return Err(Error::Index {
                message: "PQIndex::search: PQ is not initialized".to_string(),
                location: location!(),
            });
        }
        pre_filter.wait_for_ready().await?;

        let code = self.code.as_ref().unwrap().clone();
        let row_ids = self.row_ids.as_ref().unwrap().clone();

        let pq = self.pq.clone();
        let query = query.clone();
        let num_sub_vectors = self.pq.num_sub_vectors() as i32;
        spawn_cpu(move || {
            let (code, row_ids) = if pre_filter.is_empty() {
                Ok((code, row_ids))
            } else {
                Self::filter_arrays(pre_filter.as_ref(), code, row_ids, num_sub_vectors)
            }?;

            // Pre-compute distance table for each sub-vector.
            let distances = pq.compute_distances(query.key.as_ref(), &code)?;

            debug_assert_eq!(distances.len(), row_ids.len());

            let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
            let indices = sort_to_indices(&distances, None, Some(limit))?;
            let distances = take(&distances, &indices, None)?;
            let row_ids = take(row_ids.as_ref(), &indices, None)?;

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new(DIST_COL, DataType::Float32, true),
                ROW_ID_FIELD.clone(),
            ]));
            Ok(RecordBatch::try_new(schema, vec![distances, row_ids])?)
        })
        .await
    }

    fn find_partitions(&self, _: &Query) -> Result<UInt32Array> {
        unimplemented!("only for IVF")
    }

    async fn search_in_partition(
        &self,
        _: usize,
        _: &Query,
        _: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch> {
        unimplemented!("only for IVF")
    }

    fn is_loadable(&self) -> bool {
        true
    }

    fn use_residual(&self) -> bool {
        self.pq.use_residual()
    }

    /// Load a PQ index (page) from the disk.
    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let pq_code_length = self.pq.num_sub_vectors() * length;
        let pq_code = read_fixed_stride_array(
            reader.as_ref(),
            &DataType::UInt8,
            offset,
            pq_code_length,
            ..,
        )
        .await?;

        let row_id_offset = offset + pq_code_length /* *1 */;
        let row_ids = read_fixed_stride_array(
            reader.as_ref(),
            &DataType::UInt64,
            row_id_offset,
            length,
            ..,
        )
        .await?;

        Ok(Box::new(Self {
            code: Some(Arc::new(pq_code.as_primitive().clone())),
            row_ids: Some(Arc::new(row_ids.as_primitive().clone())),
            pq: self.pq.clone(),
            metric_type: self.metric_type,
        }))
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &u64>> {
        todo!("this method is for only IVF_HNSW_* index");
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        let code = self
            .code
            .as_ref()
            .unwrap()
            .values()
            .chunks_exact(self.pq.num_sub_vectors());
        let row_ids = self.row_ids.as_ref().unwrap().values().iter();
        let remapped = row_ids
            .zip(code)
            .filter_map(|(old_row_id, code)| {
                let new_row_id = mapping.get(old_row_id).cloned();
                // If the row id is not in the mapping then this row is not remapped and we keep as is
                let new_row_id = new_row_id.unwrap_or(Some(*old_row_id));
                new_row_id.map(|new_row_id| (new_row_id, code))
            })
            .collect::<Vec<_>>();

        self.row_ids = Some(Arc::new(UInt64Array::from_iter_values(
            remapped.iter().map(|(row_id, _)| *row_id),
        )));
        self.code = Some(Arc::new(UInt8Array::from_iter_values(
            remapped.into_iter().flat_map(|(_, code)| code).copied(),
        )));
        Ok(())
    }

    fn ivf_model(&self) -> IvfModel {
        unimplemented!("only for IVF")
    }
    fn quantizer(&self) -> Quantizer {
        unimplemented!("only for IVF")
    }

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        (SubIndexType::Flat, QuantizationType::Product)
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
    }
}

/// Train Product Quantizer model.
///
/// Parameters:
/// - `dataset`: The dataset to train the PQ model.
/// - `column`: The column name of the dataset.
/// - `dim`: The dimension of the vectors.
/// - `metric_type`: The metric type of the vectors.
/// - `params`: The parameters to train the PQ model.
/// - `ivf`: If provided, the IVF model to compute the residual for PQ training.
pub(super) async fn build_pq_model(
    dataset: &Dataset,
    column: &str,
    dim: usize,
    metric_type: MetricType,
    params: &PQBuildParams,
    ivf: Option<&IvfModel>,
) -> Result<Arc<dyn ProductQuantizer>> {
    if let Some(codebook) = &params.codebook {
        let mt = if metric_type == MetricType::Cosine {
            info!("Normalize training data for PQ training: Cosine");
            MetricType::L2
        } else {
            metric_type
        };

        return match codebook.data_type() {
            DataType::Float16 => Ok(Arc::new(ProductQuantizerImpl::<Float16Type>::new(
                params.num_sub_vectors,
                params.num_bits as u32,
                dim,
                Arc::new(codebook.as_primitive().clone()),
                mt,
            ))),
            DataType::Float32 => Ok(Arc::new(ProductQuantizerImpl::<Float32Type>::new(
                params.num_sub_vectors,
                params.num_bits as u32,
                dim,
                Arc::new(codebook.as_primitive().clone()),
                mt,
            ))),
            DataType::Float64 => Ok(Arc::new(ProductQuantizerImpl::<Float64Type>::new(
                params.num_sub_vectors,
                params.num_bits as u32,
                dim,
                Arc::new(codebook.as_primitive().clone()),
                mt,
            ))),
            _ => {
                return Err(Error::Index {
                    message: format!("Wrong codebook data type: {:?}", codebook.data_type()),
                    location: location!(),
                });
            }
        };
    }
    info!(
        "Start to train PQ code: PQ{}, bits={}",
        params.num_sub_vectors, params.num_bits
    );
    let expected_sample_size =
        lance_index::vector::pq::num_centroids(params.num_bits as u32) * params.sample_rate;
    info!(
        "Loading training data for PQ. Sample size: {}",
        expected_sample_size
    );
    let start = std::time::Instant::now();
    let mut training_data =
        maybe_sample_training_data(dataset, column, expected_sample_size).await?;
    info!(
        "Finished loading training data in {:02} seconds",
        start.elapsed().as_secs_f32()
    );

    info!(
        "starting to compute partitions for PQ training, sample size: {}",
        training_data.value_length()
    );

    if metric_type == MetricType::Cosine {
        info!("Normalize training data for PQ training: Cosine");
        training_data = normalize_fsl(&training_data)?;
    }

    let training_data = if let Some(ivf) = ivf {
        // Compute residual for PQ training.
        //
        // TODO: consolidate IVF models to `lance_index`.
        let ivf2 = lance_index::vector::ivf::new_ivf_transformer(
            ivf.centroids.clone().unwrap(),
            MetricType::L2,
            vec![],
        );
        span!(Level::INFO, "compute residual for PQ training")
            .in_scope(|| ivf2.compute_residual(&training_data))?
    } else {
        training_data
    };
    info!("Start train PQ: params={:#?}", params);
    let pq = params.build(&training_data, MetricType::L2).await?;
    info!("Trained PQ in: {} seconds", start.elapsed().as_secs_f32());
    Ok(pq)
}

pub(crate) fn build_pq_storage(
    distance_type: DistanceType,
    row_ids: Arc<dyn Array>,
    code_array: Vec<Arc<dyn Array>>,
    pq: Arc<dyn ProductQuantizer>,
) -> Result<ProductQuantizationStorage> {
    let pq_arrs = code_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
    let pq_column = concat(&pq_arrs)?;
    std::mem::drop(code_array);

    let pq_batch = RecordBatch::try_from_iter_with_nullable(vec![
        (ROW_ID, row_ids, true),
        (pq.column(), pq_column, false),
    ])?;
    let pq_store = ProductQuantizationStorage::new(
        pq.codebook_as_fsl()
            .values()
            .as_primitive::<Float32Type>()
            .clone()
            .into(),
        pq_batch.clone(),
        pq.num_bits(),
        pq.num_sub_vectors(),
        pq.dimension(),
        distance_type,
    )?;

    Ok(pq_store)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::ivf::build_ivf_model;
    use arrow_array::RecordBatchIterator;
    use arrow_schema::{Field, Schema};
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_testing::datagen::generate_random_array_with_range;
    use std::ops::Range;
    use tempfile::tempdir;

    const DIM: usize = 128;
    async fn generate_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range(1000 * DIM, range);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema = Arc::new(
            Schema::new(vec![Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            )])
            .with_metadata(metadata),
        );
        let fsl = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![fsl.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        (Dataset::write(batches, test_uri, None).await.unwrap(), fsl)
    }

    #[tokio::test]
    async fn test_build_pq_model_l2() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (dataset, _) = generate_dataset(test_uri, 100.0..120.0).await;

        let centroids = generate_random_array_with_range(4 * DIM, -1.0..1.0);
        let fsl = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf = IvfModel::new(fsl);
        let params = PQBuildParams::new(16, 8);
        let pq = build_pq_model(&dataset, "vector", DIM, MetricType::L2, &params, Some(&ivf))
            .await
            .unwrap();

        assert_eq!(pq.num_sub_vectors(), 16);
        assert_eq!(pq.num_bits(), 8);
        assert_eq!(pq.dimension(), DIM);

        let codebook = pq.codebook_as_fsl();
        assert_eq!(codebook.len(), 256);
        codebook
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!((99.0..121.0).contains(v));
            });
    }

    #[tokio::test]
    async fn test_build_pq_model_cosine() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (dataset, vectors) = generate_dataset(test_uri, 100.0..120.0).await;

        let ivf_params = IvfBuildParams::new(4);
        let ivf = build_ivf_model(&dataset, "vector", DIM, MetricType::Cosine, &ivf_params)
            .await
            .unwrap();
        let params = PQBuildParams::new(16, 8);
        let pq = build_pq_model(
            &dataset,
            "vector",
            DIM,
            MetricType::Cosine,
            &params,
            Some(&ivf),
        )
        .await
        .unwrap();

        assert_eq!(pq.num_sub_vectors(), 16);
        assert_eq!(pq.num_bits(), 8);
        assert_eq!(pq.dimension(), DIM);

        let codebook = pq.codebook_as_fsl();
        assert_eq!(codebook.len(), 256);
        codebook
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!((-1.0..1.0).contains(v));
            });

        let vectors = normalize_fsl(&vectors).unwrap();
        let row = vectors.slice(0, 1);

        let ivf2 = lance_index::vector::ivf::new_ivf_transformer(
            ivf.centroids.clone().unwrap(),
            MetricType::L2,
            vec![],
        );

        let residual_query = ivf2.compute_residual(&row).unwrap();
        let pq_code = pq.transform(&residual_query).unwrap();
        let distances = pq
            .compute_distances(
                &residual_query.value(0),
                pq_code.as_fixed_size_list().values().as_primitive(),
            )
            .unwrap();
        assert!(
            distances.values().iter().all(|&d| d <= 0.001),
            "distances: {:?}",
            distances
        );
    }
}
