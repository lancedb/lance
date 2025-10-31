// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;
use std::{collections::HashMap, pin::Pin};

use arrow::array::{AsArray as _, PrimitiveBuilder, UInt32Builder, UInt64Builder};
use arrow::compute::sort_to_indices;
use arrow::datatypes::{self};
use arrow::datatypes::{Float16Type, Float64Type, UInt64Type, UInt8Type};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, PrimitiveArray,
    RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Fields};
use futures::{
    prelude::stream::{StreamExt, TryStreamExt},
    Stream,
};
use futures::{stream, FutureExt};
use itertools::Itertools;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::datatypes::Schema;
use lance_core::utils::tempfile::TempStdDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::ROW_ID;
use lance_core::{Error, Result, ROW_ID_FIELD};
use lance_file::v2::writer::FileWriter;
use lance_index::frag_reuse::FragReuseIndex;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::vector::bq::storage::{unpack_codes, RABIT_CODE_COLUMN};
use lance_index::vector::kmeans::KMeansParams;
use lance_index::vector::pq::storage::transpose;
use lance_index::vector::quantizer::{
    QuantizationMetadata, QuantizationType, QuantizerBuildParams,
};
use lance_index::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use lance_index::vector::storage::STORAGE_METADATA_KEY;
use lance_index::vector::transform::Flatten;
use lance_index::vector::utils::is_finite;
use lance_index::vector::v3::shuffler::{EmptyReader, IvfShufflerReader};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::{ivf::storage::IvfModel, PART_ID_FIELD};
use lance_index::vector::{VectorIndex, LOSS_METADATA_KEY, PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_index::{
    pb,
    vector::{
        ivf::{storage::IVF_METADATA_KEY, IvfBuildParams},
        quantizer::Quantization,
        storage::{StorageBuilder, VectorStore},
        transform::Transformer,
        v3::{
            shuffler::{ShuffleReader, Shuffler},
            subindex::IvfSubIndex,
        },
        DISTANCE_TYPE_KEY,
    },
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME,
};
use lance_index::{
    IndexMetadata, IndexType, INDEX_METADATA_SCHEMA_KEY, MAX_PARTITION_SIZE_FACTOR,
    MIN_PARTITION_SIZE_PERCENT,
};
use lance_io::local::to_local_path;
use lance_io::stream::RecordBatchStream;
use lance_io::{object_store::ObjectStore, stream::RecordBatchStreamAdapter};
use lance_linalg::distance::{DistanceType, Dot, Normalize, L2};
use lance_linalg::kernels::normalize_fsl;
use log::info;
use object_store::path::Path;
use prost::Message;
use snafu::location;
use tracing::{instrument, span, Level};

use crate::dataset::ProjectionRequest;
use crate::index::vector::ivf::v2::PartitionEntry;
use crate::index::vector::utils::{infer_vector_dim, infer_vector_element_type};
use crate::Dataset;

use super::v2::IVFIndex;
use super::{
    ivf::load_precomputed_partitions_if_available,
    utils::{self, get_vector_type},
};

// the number of partitions to evaluate for reassigning
const REASSIGN_RANGE: usize = 64;

// Builder for IVF index
// The builder will train the IVF model and quantizer, shuffle the dataset, and build the sub index
// for each partition.
// To build the index for the whole dataset, call `build` method.
// To build the index for given IVF, quantizer, data stream,
// call `with_ivf`, `with_quantizer`, `shuffle_data`, and `build` in order.
pub struct IvfIndexBuilder<S: IvfSubIndex, Q: Quantization> {
    store: ObjectStore,
    column: String,
    index_dir: Path,
    distance_type: DistanceType,
    // build params, only needed for building new IVF, quantizer
    dataset: Option<Dataset>,
    shuffler: Option<Arc<dyn Shuffler>>,
    ivf_params: Option<IvfBuildParams>,
    quantizer_params: Option<Q::BuildParams>,
    sub_index_params: Option<S::BuildParams>,
    _temp_dir: TempStdDir, // store this for keeping the temp dir alive and clean up after build
    temp_dir: Path,

    // fields will be set during build
    ivf: Option<IvfModel>,
    quantizer: Option<Q>,
    shuffle_reader: Option<Arc<dyn ShuffleReader>>,

    // fields for merging indices / remapping
    existing_indices: Vec<Arc<dyn VectorIndex>>,

    frag_reuse_index: Option<Arc<FragReuseIndex>>,

    // optimize options for only incremental build
    optimize_options: Option<OptimizeOptions>,
    // number of indices merged
    merged_num: usize,
}

type BuildStream<S, Q> =
    Pin<Box<dyn Stream<Item = Result<Option<(<Q as Quantization>::Storage, S, f64)>>> + Send>>;

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> IvfIndexBuilder<S, Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        ivf_params: Option<IvfBuildParams>,
        quantizer_params: Option<Q::BuildParams>,
        sub_index_params: S::BuildParams,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let temp_dir = TempStdDir::default();
        let temp_dir_path = Path::from_filesystem_path(&temp_dir)?;
        Ok(Self {
            store: dataset.object_store().clone(),
            column,
            index_dir,
            distance_type,
            dataset: Some(dataset),
            shuffler: Some(shuffler.into()),
            ivf_params,
            quantizer_params,
            sub_index_params: Some(sub_index_params),
            _temp_dir: temp_dir,
            temp_dir: temp_dir_path,
            // fields will be set during build
            ivf: None,
            quantizer: None,
            shuffle_reader: None,
            existing_indices: Vec::new(),
            frag_reuse_index,
            optimize_options: None,
            merged_num: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_incremental(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        sub_index_params: S::BuildParams,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        optimize_options: OptimizeOptions,
    ) -> Result<Self> {
        let mut builder = Self::new(
            dataset,
            column,
            index_dir,
            distance_type,
            shuffler,
            None,
            None,
            sub_index_params,
            frag_reuse_index,
        )?;
        builder.optimize_options = Some(optimize_options);
        Ok(builder)
    }

    pub fn new_remapper(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        index: Arc<dyn VectorIndex>,
    ) -> Result<Self> {
        let ivf_index =
            index
                .as_any()
                .downcast_ref::<IVFIndex<S, Q>>()
                .ok_or(Error::invalid_input(
                    "existing index is not IVF index",
                    location!(),
                ))?;

        let temp_dir = TempStdDir::default();
        let temp_dir_path = Path::from_filesystem_path(&temp_dir)?;
        Ok(Self {
            store: dataset.object_store().clone(),
            column,
            index_dir,
            distance_type: ivf_index.metric_type(),
            dataset: Some(dataset),
            shuffler: None,
            ivf_params: None,
            quantizer_params: None,
            sub_index_params: None,
            _temp_dir: temp_dir,
            temp_dir: temp_dir_path,
            ivf: Some(ivf_index.ivf_model().clone()),
            quantizer: Some(ivf_index.quantizer().try_into()?),
            shuffle_reader: None,
            existing_indices: vec![index],
            frag_reuse_index: None,
            optimize_options: None,
            merged_num: 0,
        })
    }

    // build the index with the all data in the dataset,
    // return the number of indices merged
    pub async fn build(&mut self) -> Result<usize> {
        // step 1. train IVF & quantizer
        self.with_ivf(self.load_or_build_ivf().await?);

        self.with_quantizer(self.load_or_build_quantizer().await?);

        // step 2. shuffle the dataset
        if self.shuffle_reader.is_none() {
            self.shuffle_dataset().await?;
        }

        // step 3. build partitions
        let build_idx_stream = self.build_partitions().boxed().await?;

        // step 4. merge all partitions
        self.merge_partitions(build_idx_stream).await?;

        Ok(self.merged_num)
    }

    pub async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        debug_assert_eq!(self.existing_indices.len(), 1);
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input(
                "IVF model not set before remapping",
                location!(),
            ));
        };

        let Some(quantizer) = self.quantizer.as_ref() else {
            return Err(Error::invalid_input(
                "quantizer not set before remapping",
                location!(),
            ));
        };

        let existing_index = self.existing_indices[0].clone();

        let joined_part_idx = Self::should_join(ivf, &self.existing_indices, mapping).await?;
        let assign_batches = match joined_part_idx {
            Some(part_idx) => {
                log::info!("join partition {}", part_idx);
                let results = self.join_partition(part_idx, ivf).await?;
                let Some(ivf) = self.ivf.as_mut() else {
                    return Err(Error::invalid_input(
                        "IVF model not set before joining partition",
                        location!(),
                    ));
                };
                ivf.centroids = Some(results.new_centroids);
                results.assign_batches
            }
            None => {
                vec![None; ivf.num_partitions()]
            }
        };

        let mapping = Arc::new(mapping.clone());
        let column = self.column.clone();
        let distance_type = self.distance_type;
        let quantizer = quantizer.clone();
        let build_iter =
            assign_batches
                .into_iter()
                .enumerate()
                .map(move |(part_id, assign_batch)| {
                    let original_part_id = match joined_part_idx {
                        Some(joined_part_idx) if part_id >= joined_part_idx => part_id + 1,
                        _ => part_id,
                    };
                    let existing_index = existing_index.clone();
                    let mapping = mapping.clone();
                    let column = column.clone();
                    let distance_type = distance_type;
                    let quantizer = quantizer.clone();
                    async move {
                        let ivf_index = existing_index
                            .as_any()
                            .downcast_ref::<IVFIndex<S, Q>>()
                            .ok_or(Error::invalid_input(
                                "existing index is not IVF index",
                                location!(),
                            ))?;
                        let part = ivf_index
                            .load_partition(original_part_id, false, &NoOpMetricsCollector)
                            .await?;
                        let part = part.as_any().downcast_ref::<PartitionEntry<S, Q>>().ok_or(
                            Error::Internal {
                                message: "failed to downcast partition entry".to_string(),
                                location: location!(),
                            },
                        )?;

                        let storage = if let Some((assign_batch, _)) = assign_batch {
                            let (mut batches, _) = Self::take_partition_batches(
                                original_part_id,
                                &[existing_index],
                                None,
                            )
                            .await?;
                            if assign_batch.num_rows() > 0 {
                                let assign_batch = assign_batch.drop_column(PART_ID_COLUMN)?;
                                batches.push(assign_batch);
                            }
                            let storage =
                                StorageBuilder::new(column, distance_type, quantizer, None)?
                                    .build(batches)?;
                            storage.remap(&mapping)?
                        } else {
                            part.storage.remap(&mapping)?
                        };
                        let index = part.index.remap(&mapping, &storage)?;
                        Result::Ok(Some((storage, index, 0.0)))
                    }
                });

        self.merge_partitions(
            stream::iter(build_iter)
                .buffer_unordered(get_num_compute_intensive_cpus())
                .boxed(),
        )
        .await?;
        Ok(())
    }

    pub fn with_ivf(&mut self, ivf: IvfModel) -> &mut Self {
        self.ivf = Some(ivf);
        self
    }

    pub fn with_quantizer(&mut self, quantizer: Q) -> &mut Self {
        self.quantizer = Some(quantizer);
        self
    }

    pub fn with_existing_indices(&mut self, indices: Vec<Arc<dyn VectorIndex>>) -> &mut Self {
        self.existing_indices = indices;
        self
    }

    #[instrument(name = "load_or_build_ivf", level = "debug", skip_all)]
    async fn load_or_build_ivf(&self) -> Result<IvfModel> {
        match &self.ivf {
            Some(ivf) => Ok(ivf.clone()),
            None => {
                let Some(dataset) = self.dataset.as_ref() else {
                    return Err(Error::invalid_input(
                        "dataset not set before loading or building IVF",
                        location!(),
                    ));
                };
                let dim = utils::get_vector_dim(dataset.schema(), &self.column)?;
                let ivf_params = self.ivf_params.as_ref().ok_or(Error::invalid_input(
                    "IVF build params not set",
                    location!(),
                ))?;
                super::build_ivf_model(dataset, &self.column, dim, self.distance_type, ivf_params)
                    .await
            }
        }
    }

    #[instrument(name = "load_or_build_quantizer", level = "debug", skip_all)]
    async fn load_or_build_quantizer(&self) -> Result<Q> {
        if self.quantizer.is_some() {
            return Ok(self.quantizer.clone().unwrap());
        }

        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before loading or building quantizer",
                location!(),
            ));
        };
        let sample_size_hint = match &self.quantizer_params {
            Some(params) => params.sample_size(),
            None => 256 * 256, // here it must be retrain, let's just set sample size to the default value
        };

        let start = std::time::Instant::now();
        info!(
            "loading training data for quantizer. sample size: {}",
            sample_size_hint
        );
        let training_data =
            utils::maybe_sample_training_data(dataset, &self.column, sample_size_hint).await?;
        info!(
            "Finished loading training data in {:02} seconds",
            start.elapsed().as_secs_f32()
        );

        // If metric type is cosine, normalize the training data, and after this point,
        // treat the metric type as L2.
        let training_data = if self.distance_type == DistanceType::Cosine {
            lance_linalg::kernels::normalize_fsl(&training_data)?
        } else {
            training_data
        };

        // we filtered out nulls when sampling, but we still need to filter out NaNs and INFs here
        let training_data = arrow::compute::filter(&training_data, &is_finite(&training_data))?;
        let training_data = training_data.as_fixed_size_list();

        let training_data = match (self.ivf.as_ref(), Q::use_residual(self.distance_type)) {
            (Some(ivf), true) => {
                let ivf_transformer = lance_index::vector::ivf::new_ivf_transformer(
                    ivf.centroids.clone().unwrap(),
                    DistanceType::L2,
                    vec![],
                );
                span!(Level::INFO, "compute residual for PQ training")
                    .in_scope(|| ivf_transformer.compute_residual(training_data))?
            }
            _ => training_data.clone(),
        };

        info!("Start to train quantizer");
        let start = std::time::Instant::now();
        let quantizer = match &self.quantizer {
            Some(q) => q.clone(),
            None => {
                let quantizer_params = self.quantizer_params.as_ref().ok_or(
                    Error::invalid_input("quantizer build params not set", location!()),
                )?;
                Q::build(&training_data, DistanceType::L2, quantizer_params)?
            }
        };
        info!(
            "Trained quantizer in {:02} seconds",
            start.elapsed().as_secs_f32()
        );
        Ok(quantizer)
    }

    fn rename_row_id(
        stream: impl RecordBatchStream + Unpin + 'static,
        row_id_idx: usize,
    ) -> impl RecordBatchStream + Unpin + 'static {
        let new_schema = Arc::new(arrow_schema::Schema::new(
            stream
                .schema()
                .fields
                .iter()
                .enumerate()
                .map(|(field_idx, field)| {
                    if field_idx == row_id_idx {
                        arrow_schema::Field::new(
                            ROW_ID,
                            field.data_type().clone(),
                            field.is_nullable(),
                        )
                    } else {
                        field.as_ref().clone()
                    }
                })
                .collect::<Fields>(),
        ));
        RecordBatchStreamAdapter::new(
            new_schema.clone(),
            stream.map_ok(move |batch| {
                RecordBatch::try_new(new_schema.clone(), batch.columns().to_vec()).unwrap()
            }),
        )
    }

    async fn shuffle_dataset(&mut self) -> Result<()> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before shuffling",
                location!(),
            ));
        };

        let stream = match self
            .ivf_params
            .as_ref()
            .and_then(|p| p.precomputed_shuffle_buffers.as_ref())
        {
            Some((uri, _)) => {
                let uri = to_local_path(uri);
                // the uri points to data directory,
                // so need to trim the "data" suffix for reading the dataset
                let uri = uri.trim_end_matches("data");
                log::info!("shuffle with precomputed shuffle buffers from {}", uri);
                let ds = Dataset::open(uri).await?;
                ds.scan().try_into_stream().await?
            }
            _ => {
                log::info!("shuffle column {} over dataset", self.column);
                let mut builder = dataset.scan();
                builder
                    .batch_readahead(get_num_compute_intensive_cpus())
                    .project(&[self.column.as_str()])?
                    .with_row_id();

                let (vector_type, _) = get_vector_type(dataset.schema(), &self.column)?;
                let is_multivector = matches!(vector_type, datatypes::DataType::List(_));
                if is_multivector {
                    builder.batch_size(64);
                }
                builder.try_into_stream().await?
            }
        };

        if let Some((row_id_idx, _)) = stream.schema().column_with_name("row_id") {
            // When using precomputed shuffle buffers we can't use the column name _rowid
            // since it is reserved.  So we tolerate `row_id` as well here (and rename it
            // to _rowid to match the non-precomputed path)
            self.shuffle_data(Some(Self::rename_row_id(stream, row_id_idx)))
                .await?;
        } else {
            self.shuffle_data(Some(stream)).await?;
        }
        Ok(())
    }

    // shuffle the unindexed data and existing indices
    // data must be with schema | ROW_ID | vector_column |
    // the shuffled data will be with schema | ROW_ID | PART_ID | code_column |
    pub async fn shuffle_data(
        &mut self,
        data: Option<impl RecordBatchStream + Unpin + 'static>,
    ) -> Result<&mut Self> {
        let Some(data) = data else {
            // If we don't specify the shuffle reader, it's going to re-read the
            // dataset and duplicate the data.
            self.shuffle_reader = Some(Arc::new(EmptyReader));

            return Ok(self);
        };

        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input(
                "IVF not set before shuffle data",
                location!(),
            ));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before shuffle data",
                location!(),
            ));
        };
        let Some(shuffler) = self.shuffler.as_ref() else {
            return Err(Error::invalid_input(
                "shuffler not set before shuffle data",
                location!(),
            ));
        };

        let code_column = quantizer.column();

        let transformer = Arc::new(
            lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
                ivf.centroids.clone().unwrap(),
                self.distance_type,
                &self.column,
                quantizer.into(),
                None,
            )?,
        );

        let precomputed_partitions = if let Some(params) = self.ivf_params.as_ref() {
            load_precomputed_partitions_if_available(params)
                .await?
                .unwrap_or_default()
        } else {
            HashMap::new()
        };
        let partition_map = Arc::new(precomputed_partitions);
        let mut transformed_stream = Box::pin(
            data.map(move |batch| {
                let partition_map = partition_map.clone();
                let ivf_transformer = transformer.clone();
                tokio::spawn(async move {
                    let mut batch = batch?;
                    if !partition_map.is_empty() {
                        let row_ids = &batch[ROW_ID];
                        let part_ids = UInt32Array::from_iter(
                            row_ids
                                .as_primitive::<UInt64Type>()
                                .values()
                                .iter()
                                .map(|row_id| partition_map.get(row_id).copied()),
                        );
                        let part_ids = UInt32Array::from(part_ids);
                        batch = batch
                            .try_with_column(PART_ID_FIELD.clone(), Arc::new(part_ids.clone()))
                            .expect("failed to add part id column");

                        if part_ids.null_count() > 0 {
                            log::info!(
                                "Filter out rows without valid partition IDs: null_count={}",
                                part_ids.null_count()
                            );
                            let indices = UInt32Array::from_iter(
                                part_ids
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(idx, v)| v.map(|_| idx as u32)),
                            );
                            assert_eq!(indices.len(), batch.num_rows() - part_ids.null_count());
                            batch = batch.take(&indices)?;
                        }
                    }
                    match batch.schema().column_with_name(code_column) {
                        Some(_) => {
                            // this batch is already transformed (in case of GPU training)
                            Ok(batch)
                        }
                        None => ivf_transformer.transform(&batch),
                    }
                })
            })
            .buffered(get_num_compute_intensive_cpus())
            .map(|x| x.unwrap())
            .peekable(),
        );

        let batch = transformed_stream.as_mut().peek().await;
        let schema = match batch {
            Some(Ok(b)) => b.schema(),
            Some(Err(e)) => panic!("do this better: error reading first batch: {:?}", e),
            None => {
                log::info!("no data to shuffle");
                self.shuffle_reader = Some(Arc::new(IvfShufflerReader::new(
                    Arc::new(self.store.clone()),
                    self.temp_dir.clone(),
                    vec![0; ivf.num_partitions()],
                    0.0,
                )));
                return Ok(self);
            }
        };

        self.shuffle_reader = Some(
            shuffler
                .shuffle(Box::new(RecordBatchStreamAdapter::new(
                    schema,
                    transformed_stream,
                )))
                .await?
                .into(),
        );

        Ok(self)
    }

    #[instrument(name = "build_partitions", level = "debug", skip_all)]
    async fn build_partitions(&mut self) -> Result<BuildStream<S, Q>> {
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input(
                "IVF not set before building partitions",
                location!(),
            ));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before building partition",
                location!(),
            ));
        };
        let Some(sub_index_params) = self.sub_index_params.clone() else {
            return Err(Error::invalid_input(
                "sub index params not set before building partition",
                location!(),
            ));
        };
        let Some(reader) = self.shuffle_reader.as_ref() else {
            return Err(Error::invalid_input(
                "shuffle reader not set before building partitions",
                location!(),
            ));
        };

        // if no partitions to split, we just create a new delta index,
        // otherwise, we need to merge all existing indices and split large partitions.
        let reader = reader.clone();
        let (assign_batches, merge_indices) =
            match Self::should_split(ivf, reader.as_ref(), &self.existing_indices)? {
                Some(partition) => {
                    // Perform split and record the fact for downstream build/merge
                    log::info!(
                        "split partition {}, will merge all {} delta indices",
                        partition,
                        self.existing_indices.len()
                    );
                    let split_results = self.split_partition(partition, ivf).await?;
                    let Some(ivf) = self.ivf.as_mut() else {
                        return Err(Error::invalid_input(
                            "IVF not set before building partitions",
                            location!(),
                        ));
                    };
                    ivf.centroids = Some(split_results.new_centroids);
                    (
                        split_results.assign_batches,
                        Arc::new(self.existing_indices.clone()),
                    )
                }
                None => {
                    let is_retrain = self
                        .optimize_options
                        .as_ref()
                        .map(|opt| opt.retrain)
                        .unwrap_or(false);
                    let num_to_merge = match is_retrain {
                        true => self.existing_indices.len(), // retrain, merge all indices
                        false => self
                            .optimize_options
                            .as_ref()
                            .and_then(|opt| opt.num_indices_to_merge)
                            .unwrap_or(0),
                    };

                    let indices_to_merge = self.existing_indices
                        [self.existing_indices.len().saturating_sub(num_to_merge)..]
                        .to_vec();

                    (vec![None; ivf.num_partitions()], Arc::new(indices_to_merge))
                }
            };
        self.merged_num = merge_indices.len();
        log::info!(
            "merge {}/{} delta indices",
            self.merged_num,
            self.existing_indices.len()
        );

        let distance_type = self.distance_type;
        let column = self.column.clone();
        let frag_reuse_index = self.frag_reuse_index.clone();
        let build_iter =
            assign_batches
                .into_iter()
                .enumerate()
                .map(move |(partition, assign_batch)| {
                    let reader = reader.clone();
                    let indices = merge_indices.clone();
                    let distance_type = distance_type;
                    let quantizer = quantizer.clone();
                    let sub_index_params = sub_index_params.clone();
                    let column = column.clone();
                    let frag_reuse_index = frag_reuse_index.clone();
                    async move {
                        let (mut batches, loss) = Self::take_partition_batches(
                            partition,
                            indices.as_ref(),
                            Some(reader.as_ref()),
                        )
                        .await?;

                        if let Some((assign_batch, deleted_row_ids)) = assign_batch {
                            if !deleted_row_ids.is_empty() {
                                let deleted_row_ids = HashSet::<u64>::from_iter(
                                    deleted_row_ids.values().iter().copied(),
                                );
                                for batch in batches.iter_mut() {
                                    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                                    let mask =
                                        BooleanArray::from_iter(row_ids.iter().map(|row_id| {
                                            row_id.map(|row_id| !deleted_row_ids.contains(&row_id))
                                        }));
                                    *batch = arrow::compute::filter_record_batch(batch, &mask)?;
                                }
                            }

                            if assign_batch.num_rows() > 0 {
                                // Drop PART_ID column from assign_batch to match schema of existing batches
                                let assign_batch = assign_batch.drop_column(PART_ID_COLUMN)?;
                                batches.push(assign_batch);
                            }
                        }

                        let num_rows = batches.iter().map(|b| b.num_rows()).sum::<usize>();
                        if num_rows == 0 {
                            return Ok(None);
                        }

                        let (storage, sub_index) = Self::build_index(
                            distance_type,
                            quantizer,
                            sub_index_params,
                            batches,
                            column,
                            frag_reuse_index,
                        )?;
                        Ok(Some((storage, sub_index, loss)))
                    }
                });
        Ok(stream::iter(build_iter)
            .buffered(get_num_compute_intensive_cpus())
            .boxed())
    }

    #[instrument(name = "build_index", level = "debug", skip_all)]
    #[allow(clippy::too_many_arguments)]
    fn build_index(
        distance_type: DistanceType,
        quantizer: Q,
        sub_index_params: S::BuildParams,
        batches: Vec<RecordBatch>,
        column: String,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<(Q::Storage, S)> {
        let storage = StorageBuilder::new(column, distance_type, quantizer, frag_reuse_index)?
            .build(batches)?;
        let sub_index = S::index_vectors(&storage, sub_index_params)?;

        Ok((storage, sub_index))
    }

    #[instrument(name = "take_partition_batches", level = "debug", skip_all)]
    async fn take_partition_batches(
        part_id: usize,
        existing_indices: &[Arc<dyn VectorIndex>],
        reader: Option<&dyn ShuffleReader>,
    ) -> Result<(Vec<RecordBatch>, f64)> {
        let mut batches = Vec::new();
        for existing_index in existing_indices.iter() {
            let existing_index = existing_index
                .as_any()
                .downcast_ref::<IVFIndex<S, Q>>()
                .ok_or(Error::invalid_input(
                    "existing index is not IVF index",
                    location!(),
                ))?;

            // Skip if this partition doesn't exist in the existing index
            // This can happen after a split creates a new partition
            if part_id >= existing_index.ivf_model().num_partitions() {
                continue;
            }

            let part_storage = existing_index.load_partition_storage(part_id).await?;
            let mut part_batches = part_storage.to_batches()?.collect::<Vec<_>>();
            // for PQ, the PQ codes are transposed, so we need to transpose them back
            match Q::quantization_type() {
                QuantizationType::Product => {
                    for batch in part_batches.iter_mut() {
                        if batch.num_rows() == 0 {
                            continue;
                        }

                        let codes = batch[PQ_CODE_COLUMN]
                            .as_fixed_size_list()
                            .values()
                            .as_primitive::<datatypes::UInt8Type>();
                        let codes_num_bytes = codes.len() / batch.num_rows();
                        let original_codes = transpose(codes, codes_num_bytes, batch.num_rows());
                        let original_codes = FixedSizeListArray::try_new_from_values(
                            original_codes,
                            codes_num_bytes as i32,
                        )?;
                        *batch = batch
                            .replace_column_by_name(PQ_CODE_COLUMN, Arc::new(original_codes))?
                            .drop_column(PART_ID_COLUMN)?;
                    }
                }
                QuantizationType::Rabit => {
                    for batch in part_batches.iter_mut() {
                        if batch.num_rows() == 0 {
                            continue;
                        }

                        let codes = batch[RABIT_CODE_COLUMN].as_fixed_size_list();
                        let original_codes = unpack_codes(codes);
                        *batch = batch
                            .replace_column_by_name(RABIT_CODE_COLUMN, Arc::new(original_codes))?
                            .drop_column(PART_ID_COLUMN)?;
                    }
                }
                _ => {}
            }
            batches.extend(part_batches);
        }

        let mut loss = 0.0;
        // Skip if this partition doesn't exist in the reader
        // This can happen after a split creates a new partition
        if let Some(reader) = reader {
            if reader.partition_size(part_id)? > 0 {
                let mut partition_data = reader.read_partition(part_id).await?.ok_or(Error::io(
                    format!("partition {} is empty", part_id).as_str(),
                    location!(),
                ))?;
                while let Some(batch) = partition_data.try_next().await? {
                    loss += batch
                        .metadata()
                        .get(LOSS_METADATA_KEY)
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                        .unwrap_or(0.0);
                    let batch = batch.drop_column(PART_ID_COLUMN)?;
                    batches.push(batch);
                }
            }
        }

        Ok((batches, loss))
    }

    #[instrument(name = "merge_partitions", level = "debug", skip_all)]
    async fn merge_partitions(&mut self, mut build_stream: BuildStream<S, Q>) -> Result<()> {
        let Some(ivf) = self.ivf.as_ref() else {
            return Err(Error::invalid_input(
                "IVF not set before merge partitions",
                location!(),
            ));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before merge partitions",
                location!(),
            ));
        };

        // prepare the final writers
        let storage_path = self.index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        let index_path = self.index_dir.child(INDEX_FILE_NAME);

        let mut fields = vec![ROW_ID_FIELD.clone(), quantizer.field()];
        fields.extend(quantizer.extra_fields());
        let storage_schema: Schema = (&arrow_schema::Schema::new(fields)).try_into()?;
        let mut storage_writer = FileWriter::try_new(
            self.store.create(&storage_path).await?,
            storage_schema.clone(),
            Default::default(),
        )?;
        let mut index_writer = FileWriter::try_new(
            self.store.create(&index_path).await?,
            S::schema().as_ref().try_into()?,
            Default::default(),
        )?;

        // maintain the IVF partitions
        let mut storage_ivf = IvfModel::empty();
        let mut index_ivf = IvfModel::new(ivf.centroids.clone().unwrap(), ivf.loss);
        let mut partition_index_metadata = Vec::with_capacity(ivf.num_partitions());

        let mut part_id = 0;
        let mut total_loss = 0.0;
        log::info!("merging {} partitions", ivf.num_partitions());
        while let Some(part) = build_stream.try_next().await? {
            part_id += 1;
            let Some((storage, index, loss)) = part else {
                log::warn!("partition {} is empty, skipping", part_id);

                storage_ivf.add_partition(0);
                index_ivf.add_partition(0);
                partition_index_metadata.push(String::new());

                continue;
            };
            total_loss += loss;

            if storage.len() == 0 {
                storage_ivf.add_partition(0);
            } else {
                let batches = storage.to_batches()?.collect::<Vec<_>>();
                let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
                storage_writer.write_batch(&batch).await?;
                storage_ivf.add_partition(batch.num_rows() as u32);
            }

            let index_batch = index.to_batch()?;
            if index_batch.num_rows() == 0 {
                index_ivf.add_partition(0);
                partition_index_metadata.push(String::new());
            } else {
                index_writer.write_batch(&index_batch).await?;
                index_ivf.add_partition(index_batch.num_rows() as u32);
                partition_index_metadata.push(
                    index_batch
                        .schema()
                        .metadata
                        .get(S::metadata_key())
                        .cloned()
                        .unwrap_or_default(),
                );
            }
        }

        match self.shuffle_reader.as_ref() {
            Some(reader) => {
                // it's building index, the loss is already calculated in the shuffle reader
                if let Some(loss) = reader.total_loss() {
                    total_loss += loss;
                }
                index_ivf.loss = Some(total_loss);
            }
            None => {
                // it's remapping, we don't need to change the loss
            }
        }

        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, self.distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        // For now, each partition's metadata is just the quantizer,
        // it's all the same for now, so we just take the first one
        let mut metadata = quantizer.metadata(Some(QuantizationMetadata {
            codebook_position: Some(0),
            codebook: None,
            transposed: true,
        }));
        if let Some(extra_metadata) = metadata.extra_metadata()? {
            let idx = storage_writer.add_global_buffer(extra_metadata).await?;
            metadata.set_buffer_index(idx);
        }
        let metadata = serde_json::to_string(&metadata)?;
        let storage_partition_metadata = vec![metadata];
        storage_writer.add_schema_metadata(
            STORAGE_METADATA_KEY,
            serde_json::to_string(&storage_partition_metadata)?,
        );

        let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
        let index_metadata = IndexMetadata {
            index_type: index_type_string(S::name().try_into()?, Q::quantization_type()),
            distance_type: self.distance_type.to_string(),
        };
        index_writer.add_schema_metadata(
            INDEX_METADATA_SCHEMA_KEY,
            serde_json::to_string(&index_metadata)?,
        );
        let ivf_buffer_pos = index_writer
            .add_global_buffer(index_ivf_pb.encode_to_vec().into())
            .await?;
        index_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        index_writer.add_schema_metadata(
            S::metadata_key(),
            serde_json::to_string(&partition_index_metadata)?,
        );

        storage_writer.finish().await?;
        index_writer.finish().await?;

        log::info!("merging {} partitions done", ivf.num_partitions());

        Ok(())
    }

    // take raw vectors from the dataset
    //
    // returns batches of schema | row_id | vector |
    async fn take_vectors(
        dataset: &Dataset,
        column: &str,
        store: &ObjectStore,
        row_ids: &[u64],
    ) -> Result<Vec<RecordBatch>> {
        let projection = Arc::new(dataset.schema().project(&[column])?);
        // arrow uses i32 for index, so we chunk the row ids to avoid large batch causing overflow
        let mut batches = Vec::new();
        let row_ids = dataset.filter_deleted_ids(row_ids).await?;
        for chunk in row_ids.chunks(store.block_size()) {
            let batch = dataset
                .take_rows(chunk, ProjectionRequest::Schema(projection.clone()))
                .await?;
            if batch.num_rows() != chunk.len() {
                return Err(Error::invalid_input(
                    format!(
                        "batch.num_rows() != chunk.len() ({} != {})",
                        batch.num_rows(),
                        chunk.len()
                    ),
                    location!(),
                ));
            }
            let batch = batch.try_with_column(
                ROW_ID_FIELD.clone(),
                Arc::new(UInt64Array::from(chunk.to_vec())),
            )?;
            batches.push(batch);
        }
        Ok(batches)
    }

    // helper to load row ids and vectors for a partition
    async fn load_partition_raw_vectors(
        &self,
        part_idx: usize,
    ) -> Result<Option<(UInt64Array, FixedSizeListArray)>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before split partition",
                location!(),
            ));
        };

        let mut row_ids = self.partition_row_ids(part_idx).await?;
        if !row_ids.is_sorted() {
            row_ids.sort();
        }
        // dedup is needed if it's multivector
        row_ids.dedup();

        let batches = Self::take_vectors(dataset, &self.column, &self.store, &row_ids).await?;
        if batches.is_empty() {
            return Ok(None);
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
        // for multivector, we need to flatten the vectors
        let batch = Flatten::new(&self.column).transform(&batch)?;
        // need to retrieve the row ids from the batch because some rows may have been deleted
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone();
        let vectors = batch[&self.column].as_fixed_size_list().clone();
        Ok(Some((row_ids, vectors)))
    }

    // return the partition ids that should be split.
    // split at most one partition each building,
    // return the largest partition if multiple partitions are larger than the threshold
    fn should_split(
        ivf: &IvfModel,
        reader: &dyn ShuffleReader,
        existing_indices: &[Arc<dyn VectorIndex>],
    ) -> Result<Option<usize>> {
        let index_type = IndexType::try_from(
            index_type_string(S::name().try_into()?, Q::quantization_type()).as_str(),
        )?;

        let mut split_partition = None;
        let mut max_partition_size = 0;
        for partition in 0..ivf.num_partitions() {
            let mut num_rows = reader.partition_size(partition)?;
            for index in existing_indices.iter() {
                num_rows += index.ivf_model().partition_size(partition);
            }
            if num_rows > max_partition_size
                && num_rows > MAX_PARTITION_SIZE_FACTOR * index_type.target_partition_size()
            {
                max_partition_size = num_rows;
                split_partition = Some(partition);
            }
        }
        Ok(split_partition)
    }

    // split this partition,
    // 1. take raw vectors by row ids in this partition
    // 2. run KMeans with k=2 to get 2 new centroids
    // 3. reassign the vectors to the 2 new partitions
    async fn split_partition(&self, part_idx: usize, ivf: &IvfModel) -> Result<AssignResult> {
        // take the raw vectors from dataset
        let Some((row_ids, vectors)) = self.load_partition_raw_vectors(part_idx).await? else {
            return Ok(AssignResult {
                assign_batches: vec![None; ivf.num_partitions()],
                new_centroids: ivf.centroids_array().unwrap().clone(),
            });
        };

        let element_type = infer_vector_element_type(vectors.data_type())?;
        match element_type {
            DataType::Float16 => {
                self.split_partition_impl::<Float16Type>(part_idx, ivf, &row_ids, &vectors)
                    .await
            }
            DataType::Float32 => {
                self.split_partition_impl::<Float32Type>(part_idx, ivf, &row_ids, &vectors)
                    .await
            }
            DataType::Float64 => {
                self.split_partition_impl::<Float64Type>(part_idx, ivf, &row_ids, &vectors)
                    .await
            }
            DataType::UInt8 => {
                self.split_partition_impl::<UInt8Type>(part_idx, ivf, &row_ids, &vectors)
                    .await
            }
            dt => Err(Error::invalid_input(
                format!(
                    "vectors must be float16, float32, float64 or uint8, but got {:?}",
                    dt
                ),
                location!(),
            )),
        }
    }

    async fn split_partition_impl<T: ArrowPrimitiveType>(
        &self,
        part_idx: usize,
        ivf: &IvfModel,
        row_ids: &UInt64Array,
        vectors: &FixedSizeListArray,
    ) -> Result<AssignResult>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let centroids = ivf.centroids_array().unwrap();
        let mut new_centroids: Vec<ArrayRef> = Vec::with_capacity(ivf.num_partitions() + 1);
        new_centroids.extend(centroids.iter().map(|vec| vec.unwrap()));

        let dimension = infer_vector_dim(vectors.data_type())?;
        // train kmeans to get 2 new centroids
        let (normalized_dist_type, normalized_vectors) = match self.distance_type {
            DistanceType::Cosine => {
                let vectors = normalize_fsl(vectors)?;
                (DistanceType::L2, vectors)
            }
            _ => (self.distance_type, vectors.clone()),
        };
        let params = KMeansParams::new(None, 50, 1, normalized_dist_type);
        let kmeans = lance_index::vector::kmeans::train_kmeans::<T>(
            normalized_vectors.values().as_primitive::<T>(),
            params,
            dimension,
            2,
            256,
        )?;
        // the original centroid
        let c0 = ivf.centroid(part_idx).ok_or(Error::invalid_input(
            "original centroid not found",
            location!(),
        ))?;
        // the 2 new centroids
        let c1 = kmeans.centroids.slice(0, dimension);
        let c2 = kmeans.centroids.slice(dimension, dimension);
        // replace the original centroid with the first new one
        new_centroids[part_idx] = c1.clone();
        // append the second new one
        new_centroids.push(c2.clone());
        let centroid1_part_idx = part_idx;
        let centroid2_part_idx = new_centroids.len() - 1;

        let new_centroids = new_centroids
            .iter()
            .map(|vec| vec.as_ref())
            .collect::<Vec<_>>();
        let new_centroids = arrow::compute::concat(&new_centroids)?;

        // get top REASSIGN_RANGE centroids from c0
        let (reassign_part_ids, reassign_part_centroids) =
            self.select_reassign_candidates(ivf, &c0)?;

        // compute the distance between the vectors and the 3 centroids (original one and the 2 new ones)
        let d0 = self.distance_type.arrow_batch_func()(&c0, vectors)?;
        let d1 = self.distance_type.arrow_batch_func()(&c1, vectors)?;
        let d2 = self.distance_type.arrow_batch_func()(&c2, vectors)?;
        let d0 = d0.values();
        let d1 = d1.values();
        let d2 = d2.values();

        let mut assign_ops = vec![Vec::new(); ivf.num_partitions() + 1];
        // assign the vectors in the original partition
        self.assign_vectors::<T>(
            part_idx,
            centroid1_part_idx,
            centroid2_part_idx,
            row_ids,
            vectors,
            d0,
            d1,
            d2,
            &reassign_part_ids,
            &reassign_part_centroids,
            true,
            &mut assign_ops,
        )?;
        // assign the vectors in the reassigned partitions
        for (i, idx) in reassign_part_ids.values().iter().enumerate() {
            let part_idx = *idx as usize;
            let Some((row_ids, vectors)) = self.load_partition_raw_vectors(part_idx).await? else {
                // all vectors in this partition have been deleted
                continue;
            };

            let d0 =
                self.distance_type.arrow_batch_func()(&reassign_part_centroids.value(i), &vectors)?;
            let d1 = self.distance_type.arrow_batch_func()(&c1, &vectors)?;
            let d2 = self.distance_type.arrow_batch_func()(&c2, &vectors)?;
            let d0 = d0.values();
            let d1 = d1.values();
            let d2 = d2.values();

            self.assign_vectors::<T>(
                part_idx,
                centroid1_part_idx,
                centroid2_part_idx,
                &row_ids,
                &vectors,
                d0,
                d1,
                d2,
                &reassign_part_ids,
                &reassign_part_centroids,
                false,
                &mut assign_ops,
            )?;
        }

        let new_centroids =
            FixedSizeListArray::try_new_from_values(new_centroids, dimension as i32)?;
        let assign_batches = self.build_assign_batch::<T>(&new_centroids, &assign_ops)?;

        Ok(AssignResult {
            assign_batches,
            new_centroids,
        })
    }

    // should join the partition if the number of rows in the partition is less than MIN_PARTITION_SIZE_PERCENT * target_partition_size / 100
    async fn should_join(
        ivf: &IvfModel,
        existing_indices: &[Arc<dyn VectorIndex>],
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<Option<usize>> {
        if ivf.num_partitions() <= 1 {
            // we have to keep at least one partition
            return Ok(None);
        }

        let index_type = IndexType::try_from(
            index_type_string(S::name().try_into()?, Q::quantization_type()).as_str(),
        )?;
        let mut join_partition = None;
        let mut min_partition_size = usize::MAX;
        for part_id in 0..ivf.num_partitions() {
            let mut num_rows = 0;
            for index in existing_indices.iter() {
                let ivf_index =
                    index
                        .as_any()
                        .downcast_ref::<IVFIndex<S, Q>>()
                        .ok_or(Error::invalid_input(
                            "existing index is not IVF index",
                            location!(),
                        ))?;
                let part = ivf_index
                    .load_partition(part_id, true, &NoOpMetricsCollector)
                    .await?;
                let part = part.as_any().downcast_ref::<PartitionEntry<S, Q>>().ok_or(
                    Error::Internal {
                        message: "failed to downcast partition entry".to_string(),
                        location: location!(),
                    },
                )?;

                let valid_num_rows = part
                    .storage
                    .row_ids()
                    .filter(|row_id| !matches!(mapping.get(row_id), Some(None)))
                    .count();
                num_rows += valid_num_rows;
            }
            if num_rows < min_partition_size
                && num_rows < MIN_PARTITION_SIZE_PERCENT * index_type.target_partition_size() / 100
            {
                min_partition_size = num_rows;
                join_partition = Some(part_id);
            }
        }
        Ok(join_partition)
    }

    // join the given partition:
    // 1. delete the original parttion
    // 2. reasign all vectors of the original partitions
    async fn join_partition(&self, part_idx: usize, ivf: &IvfModel) -> Result<AssignResult> {
        let centroids = ivf.centroids_array().unwrap();
        let mut new_centroids: Vec<ArrayRef> = Vec::with_capacity(ivf.num_partitions() - 1);
        new_centroids.extend(centroids.iter().enumerate().filter_map(|(i, vec)| {
            if i == part_idx {
                None
            } else {
                Some(vec.unwrap())
            }
        }));
        let new_centroids = new_centroids
            .iter()
            .map(|vec| vec.as_ref())
            .collect::<Vec<_>>();
        let new_centroids = arrow::compute::concat(&new_centroids)?;
        let new_centroids =
            FixedSizeListArray::try_new_from_values(new_centroids, centroids.value_length())?;

        // take the raw vectors from dataset
        let Some((row_ids, vectors)) = self.load_partition_raw_vectors(part_idx).await? else {
            return Ok(AssignResult {
                assign_batches: vec![None; ivf.num_partitions() - 1],
                new_centroids,
            });
        };

        match vectors.value_type() {
            DataType::Float16 => {
                self.join_partition_impl::<Float16Type>(
                    part_idx,
                    ivf,
                    &row_ids,
                    &vectors,
                    new_centroids,
                )
                .await
            }
            DataType::Float32 => {
                self.join_partition_impl::<Float32Type>(
                    part_idx,
                    ivf,
                    &row_ids,
                    &vectors,
                    new_centroids,
                )
                .await
            }
            DataType::Float64 => {
                self.join_partition_impl::<Float64Type>(
                    part_idx,
                    ivf,
                    &row_ids,
                    &vectors,
                    new_centroids,
                )
                .await
            }
            DataType::UInt8 => {
                self.join_partition_impl::<UInt8Type>(
                    part_idx,
                    ivf,
                    &row_ids,
                    &vectors,
                    new_centroids,
                )
                .await
            }
            dt => Err(Error::invalid_input(
                format!(
                    "vectors must be float16, float32, float64 or uint8, but got {:?}",
                    dt
                ),
                location!(),
            )),
        }
    }

    async fn join_partition_impl<T: ArrowPrimitiveType>(
        &self,
        part_idx: usize,
        ivf: &IvfModel,
        row_ids: &UInt64Array,
        vectors: &FixedSizeListArray,
        new_centroids: FixedSizeListArray,
    ) -> Result<AssignResult>
    where
        T::Native: Dot + L2 + Normalize,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        assert_eq!(row_ids.len(), vectors.len());

        // the original centroid
        let c0 = ivf.centroid(part_idx).ok_or(Error::invalid_input(
            "original centroid not found",
            location!(),
        ))?;

        // get top REASSIGN_RANGE centroids from c0
        let (reassign_part_ids, reassign_part_centroids) =
            self.select_reassign_candidates(ivf, &c0)?;

        let new_part_id = |idx: usize| -> usize {
            if idx < part_idx {
                idx
            } else {
                // part_idx has been deleted, so any part id after it should be decremented by 1
                idx - 1
            }
        };
        let mut assign_ops = vec![Vec::new(); ivf.num_partitions() - 1];
        // reassign the vectors in the original partition
        for (i, &row_id) in row_ids.values().iter().enumerate() {
            let ReassignPartition::ReassignCandidate(idx) = self.reassign_vectors(
                vectors.value(i).as_primitive::<T>(),
                None,
                &reassign_part_ids,
                &reassign_part_centroids,
            )?
            else {
                log::warn!("this is a bug, the vector is not reassigned");
                continue;
            };

            assign_ops[new_part_id(idx as usize)].push(AssignOp::Add((row_id, vectors.value(i))));
        }
        let assign_batches = self.build_assign_batch::<T>(&new_centroids, &assign_ops)?;

        Ok(AssignResult {
            assign_batches,
            new_centroids,
        })
    }

    // Build the assign batch form assign ops for each partition
    // returns the assign batch and the deleted row ids
    fn build_assign_batch<T: ArrowPrimitiveType>(
        &self,
        centroids: &FixedSizeListArray,
        assign_ops: &[Vec<AssignOp>],
    ) -> Result<Vec<Option<(RecordBatch, UInt64Array)>>> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before building assign batch",
                location!(),
            ));
        };
        let Some(quantizer) = self.quantizer.clone() else {
            return Err(Error::invalid_input(
                "quantizer not set before building assign batch",
                location!(),
            ));
        };

        let Some(vector_field) =
            dataset
                .schema()
                .field(&self.column)
                .map(|f| match f.data_type() {
                    DataType::List(inner) | DataType::LargeList(inner) => {
                        Field::new(self.column.as_str(), inner.data_type().clone(), true)
                    }
                    _ => f.into(),
                })
        else {
            return Err(Error::invalid_input(
                "vector field not found in dataset schema",
                location!(),
            ));
        };

        let transformer = Arc::new(
            lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
                centroids.clone(),
                self.distance_type,
                &self.column,
                quantizer.into(),
                None,
            )?,
        );

        let num_rows = assign_ops
            .iter()
            .map(|ops| {
                ops.iter()
                    .map(|op| match op {
                        AssignOp::Add(_) => 1,
                        AssignOp::Remove(_) => 0,
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();

        // build the input batch with schema | row_id | vector | part_id |
        let mut row_ids_builder = UInt64Builder::with_capacity(num_rows);
        let mut vector_builder =
            PrimitiveBuilder::<T>::with_capacity(num_rows * centroids.value_length() as usize);
        let mut part_ids_builder = UInt32Builder::with_capacity(num_rows);
        let mut deleted_row_ids = UInt64Builder::with_capacity(num_rows);

        let mut ops_count = Vec::with_capacity(assign_ops.len());
        for (part_idx, ops) in assign_ops.iter().enumerate() {
            let mut add_count = 0;
            let mut remove_count = 0;
            for op in ops {
                match op {
                    AssignOp::Add((row_id, vector)) => {
                        row_ids_builder.append_value(*row_id);
                        vector_builder.append_array(vector.as_primitive::<T>());
                        part_ids_builder.append_value(part_idx as u32);
                        add_count += 1;
                    }
                    AssignOp::Remove(row_id) => {
                        deleted_row_ids.append_value(*row_id);
                        remove_count += 1;
                    }
                }
            }
            ops_count.push((add_count, remove_count));
        }

        let row_ids = row_ids_builder.finish();
        let vector = FixedSizeListArray::try_new_from_values(
            vector_builder.finish(),
            centroids.value_length(),
        )?;
        let part_ids = part_ids_builder.finish();
        let deleted_row_ids = deleted_row_ids.finish();
        let schema = arrow_schema::Schema::new(vec![
            ROW_ID_FIELD.clone(),
            vector_field,
            PART_ID_FIELD.clone(),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(row_ids), Arc::new(vector), Arc::new(part_ids)],
        )?;
        let batch = transformer.transform(&batch)?;

        // slice the batch according to the ops count
        let mut results = Vec::with_capacity(assign_ops.len());
        let mut add_offset = 0;
        let mut remove_offset = 0;
        for (add_count, remove_count) in ops_count.into_iter() {
            if add_count == 0 && remove_count == 0 {
                results.push(None);
                continue;
            }
            let batch = batch.slice(add_offset, add_count);
            let deleted_row_ids = deleted_row_ids.slice(remove_offset, remove_count);
            results.push(Some((batch, deleted_row_ids)));
            add_offset += add_count;
            remove_offset += remove_count;
        }
        Ok(results)
    }

    async fn partition_row_ids(&self, part_idx: usize) -> Result<Vec<u64>> {
        // existing part: read from the existing indices
        let mut row_ids = Vec::new();
        for index in self.existing_indices.iter() {
            let mut reader = index
                .partition_reader(part_idx, false, &NoOpMetricsCollector)
                .await?;
            while let Some(batch) = reader.try_next().await? {
                row_ids.extend(batch[ROW_ID].as_primitive::<UInt64Type>().values());
            }
        }

        // incremental part: read from the shuffler reader
        if let Some(reader) = self.shuffle_reader.as_ref() {
            // TODO: don't read vectors here, just read row ids
            if let Some(mut reader) = reader.read_partition(part_idx).await? {
                while let Some(batch) = reader.try_next().await? {
                    row_ids.extend(batch[ROW_ID].as_primitive::<UInt64Type>().values());
                }
            }
        }
        Ok(row_ids)
    }

    // returns the closest REASSIGN_RANGE partitions (indices and centroids) from c0
    fn select_reassign_candidates(
        &self,
        ivf: &IvfModel,
        c0: &ArrayRef,
    ) -> Result<(UInt32Array, FixedSizeListArray)> {
        let reassign_range = std::cmp::min(REASSIGN_RANGE + 1, ivf.num_partitions());
        let centroids = ivf.centroids_array().unwrap();
        let centroid_dists = self.distance_type.arrow_batch_func()(&c0, centroids)?;
        let reassign_range_candidates =
            sort_to_indices(centroid_dists.as_ref(), None, Some(reassign_range))?;
        // exclude the original centroid itself
        let reassign_candidate_ids = &reassign_range_candidates.slice(1, reassign_range - 1);
        let reassign_candidate_centroids =
            arrow::compute::take(centroids, reassign_candidate_ids, None)?;
        Ok((
            reassign_candidate_ids.clone(),
            reassign_candidate_centroids.as_fixed_size_list().clone(),
        ))
    }

    // assign the vectors of original partition
    #[allow(clippy::too_many_arguments)]
    fn assign_vectors<T: ArrowPrimitiveType>(
        &self,
        part_idx: usize,
        centroid1_part_idx: usize,
        centroid2_part_idx: usize,
        row_ids: &UInt64Array,
        vectors: &FixedSizeListArray,
        d0: &[f32],
        d1: &[f32],
        d2: &[f32],
        reassign_part_ids: &UInt32Array,
        reassign_part_centroids: &FixedSizeListArray,
        // the assign ops for each partition
        // the length must be `old_num_partitions + 1`
        deleted_original_partition: bool,
        assign_ops: &mut [Vec<AssignOp>],
    ) -> Result<()> {
        for (i, &row_id) in row_ids.values().iter().enumerate() {
            if d0[i] <= d1[i] && d0[i] <= d2[i] {
                if !deleted_original_partition {
                    // the original partition is not deleted, we just keep the vector in the original partition
                    continue;
                }
                match self.reassign_vectors(
                    vectors.value(i).as_primitive::<T>(),
                    Some((d1[i], d2[i])),
                    reassign_part_ids,
                    reassign_part_centroids,
                )? {
                    ReassignPartition::NewCentroid1 => {
                        // replace the original partition with the first new one
                        assign_ops[centroid1_part_idx]
                            .push(AssignOp::Add((row_id, vectors.value(i))));
                    }
                    ReassignPartition::NewCentroid2 => {
                        // append the new second one
                        assign_ops[centroid2_part_idx]
                            .push(AssignOp::Add((row_id, vectors.value(i))));
                    }
                    ReassignPartition::ReassignCandidate(idx) => {
                        // replace the original partition with the reassigned one
                        assign_ops[idx as usize].push(AssignOp::Add((row_id, vectors.value(i))));
                    }
                }
            } else {
                if !deleted_original_partition {
                    // the original partition is not deleted, we need to remove the vector from the original partition
                    assign_ops[part_idx].push(AssignOp::Remove(row_id));
                }
                if d1[i] <= d2[i] {
                    // centroid 1 is the closest one
                    assign_ops[centroid1_part_idx].push(AssignOp::Add((row_id, vectors.value(i))));
                } else {
                    // centroid 2 is the closest one
                    assign_ops[centroid2_part_idx].push(AssignOp::Add((row_id, vectors.value(i))));
                }
            }
        }
        Ok(())
    }

    // assign a vector to the closest partition among:
    // 1. the 2 new centroids
    // 2. the closest REASSIGN_RANGE partitions from the original centroid
    fn reassign_vectors<T: ArrowPrimitiveType>(
        &self,
        vector: &PrimitiveArray<T>,
        // the dists to the 2 new centroids
        split_centroids_dists: Option<(f32, f32)>,
        reassign_candidate_ids: &UInt32Array,
        reassign_candidate_centroids: &FixedSizeListArray,
    ) -> Result<ReassignPartition> {
        let dists = self.distance_type.arrow_batch_func()(vector, reassign_candidate_centroids)?;
        let min_dist_idx = dists
            .values()
            .iter()
            .position_min_by(|a, b| a.total_cmp(b))
            .unwrap();
        let min_dist = dists.value(min_dist_idx);
        match split_centroids_dists {
            Some((d1, d2)) => {
                if min_dist <= d1 && min_dist <= d2 {
                    Ok(ReassignPartition::ReassignCandidate(
                        reassign_candidate_ids.value(min_dist_idx),
                    ))
                } else if d1 <= d2 {
                    Ok(ReassignPartition::NewCentroid1)
                } else {
                    Ok(ReassignPartition::NewCentroid2)
                }
            }
            None => Ok(ReassignPartition::ReassignCandidate(
                reassign_candidate_ids.value(min_dist_idx),
            )),
        }
    }
}

struct AssignResult {
    // the batches of new vectors that are assigned to the partition,
    // and the deleted row ids
    assign_batches: Vec<Option<(RecordBatch, UInt64Array)>>,
    new_centroids: FixedSizeListArray,
}

#[derive(Debug, Clone)]
enum AssignOp {
    // (row_id, vector)
    // TODO: add the distance to the centroid to avoid recomputing it for RQ
    Add((u64, ArrayRef)),
    // row_id
    Remove(u64),
}

#[derive(Debug, Copy, Clone)]
enum ReassignPartition {
    NewCentroid1,
    NewCentroid2,
    ReassignCandidate(u32),
}

pub(crate) fn index_type_string(sub_index: SubIndexType, quantizer: QuantizationType) -> String {
    match (sub_index, quantizer) {
        // ignore FLAT sub index,
        // IVF_FLAT_FLAT => IVF_FLAT
        // IVF_FLAT_PQ => IVF_PQ
        (SubIndexType::Flat, quantization_type) => format!("IVF_{}", quantization_type),
        (sub_index_type, quantization_type) => {
            if sub_index_type.to_string() == quantization_type.to_string() {
                // ignore redundant quantization type
                // e.g. IVF_PQ_PQ should be IVF_PQ
                format!("IVF_{}", sub_index_type)
            } else {
                format!("IVF_{}_{}", sub_index_type, quantization_type)
            }
        }
    }
}
