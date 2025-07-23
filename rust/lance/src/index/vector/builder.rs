// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::{collections::HashMap, pin::Pin};

use arrow::datatypes;
use arrow::{array::AsArray, datatypes::UInt64Type};
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::Fields;
use futures::stream;
use futures::{
    prelude::stream::{StreamExt, TryStreamExt},
    Stream,
};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::datatypes::Schema;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::ROW_ID;
use lance_core::{Error, Result, ROW_ID_FIELD};
use lance_file::v2::writer::FileWriter;
use lance_index::frag_reuse::FragReuseIndex;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::vector::pq::storage::transpose;
use lance_index::vector::quantizer::{
    QuantizationMetadata, QuantizationType, QuantizerBuildParams,
};
use lance_index::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use lance_index::vector::storage::STORAGE_METADATA_KEY;
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
use lance_index::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};
use lance_io::local::to_local_path;
use lance_io::stream::RecordBatchStream;
use lance_io::{object_store::ObjectStore, stream::RecordBatchStreamAdapter};
use lance_linalg::distance::DistanceType;
use log::info;
use object_store::path::Path;
use prost::Message;
use snafu::location;
use tempfile::{tempdir, TempDir};
use tracing::{instrument, span, Level};

use crate::dataset::ProjectionRequest;
use crate::index::vector::ivf::v2::PartitionEntry;
use crate::Dataset;

use super::v2::IVFIndex;
use super::{
    ivf::load_precomputed_partitions_if_available,
    utils::{self, get_vector_type},
};

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
    retrain: bool,
    // build params, only needed for building new IVF, quantizer
    dataset: Option<Dataset>,
    shuffler: Option<Arc<dyn Shuffler>>,
    ivf_params: Option<IvfBuildParams>,
    quantizer_params: Option<Q::BuildParams>,
    sub_index_params: Option<S::BuildParams>,
    _temp_dir: TempDir, // store this for keeping the temp dir alive and clean up after build
    temp_dir: Path,

    // fields will be set during build
    ivf: Option<IvfModel>,
    quantizer: Option<Q>,
    shuffle_reader: Option<Arc<dyn ShuffleReader>>,

    // fields for merging indices / remapping
    existing_indices: Vec<Arc<dyn VectorIndex>>,

    frag_reuse_index: Option<Arc<FragReuseIndex>>,
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
        let temp_dir = tempdir()?;
        let temp_dir_path = Path::from_filesystem_path(temp_dir.path())?;
        Ok(Self {
            store: dataset.object_store().clone(),
            column,
            index_dir,
            distance_type,
            retrain: false,
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
        })
    }

    pub fn new_incremental(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        sub_index_params: S::BuildParams,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        Self::new(
            dataset,
            column,
            index_dir,
            distance_type,
            shuffler,
            None,
            None,
            sub_index_params,
            frag_reuse_index,
        )
    }

    pub fn new_remapper(
        store: ObjectStore,
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

        let temp_dir = tempdir()?;
        let temp_dir_path = Path::from_filesystem_path(temp_dir.path())?;
        Ok(Self {
            store,
            column,
            index_dir,
            distance_type: ivf_index.metric_type(),
            retrain: false,
            dataset: None,
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
        })
    }

    // build the index with the all data in the dataset,
    pub async fn build(&mut self) -> Result<()> {
        if self.retrain {
            self.shuffle_reader = None;
            self.existing_indices = Vec::new();
        }

        // step 1. train IVF & quantizer
        self.with_ivf(self.load_or_build_ivf().await?);

        self.with_quantizer(self.load_or_build_quantizer().await?);

        // step 2. shuffle the dataset
        if self.shuffle_reader.is_none() {
            self.shuffle_dataset().await?;
        }

        // step 3. build partitions
        let build_idx_stream = self.build_partitions().await?;

        // step 4. merge all partitions
        self.merge_partitions(build_idx_stream).await?;

        Ok(())
    }

    pub async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        debug_assert_eq!(self.existing_indices.len(), 1);
        let Some(ivf_model) = self.ivf.as_ref() else {
            return Err(Error::invalid_input(
                "IVF model not set before remapping",
                location!(),
            ));
        };
        let existing_index = self.existing_indices[0].clone();
        let mapping = Arc::new(mapping.clone());
        let mapped_stream = stream::iter(0..ivf_model.num_partitions())
            .map(move |part_id| {
                let existing_index = existing_index.clone();
                let mapping = mapping.clone();
                async move {
                    let ivf_index = existing_index
                        .as_any()
                        .downcast_ref::<IVFIndex<S, Q>>()
                        .ok_or(Error::invalid_input(
                            "existing index is not IVF index",
                            location!(),
                        ))?;
                    let part = ivf_index
                        .load_partition(part_id, false, &NoOpMetricsCollector)
                        .await?;
                    let part = part.as_any().downcast_ref::<PartitionEntry<S, Q>>().ok_or(
                        Error::Internal {
                            message: "failed to downcast partition entry".to_string(),
                            location: location!(),
                        },
                    )?;
                    Result::Ok(Some((
                        part.storage.remap(&mapping)?,
                        part.index.remap(&mapping)?,
                        0.0,
                    )))
                }
            })
            .buffered(get_num_compute_intensive_cpus())
            .boxed();

        self.merge_partitions(mapped_stream).await?;
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

    pub fn retrain(&mut self, retrain: bool) -> &mut Self {
        self.retrain = retrain;
        self
    }

    #[instrument(name = "load_or_build_ivf", level = "debug", skip_all)]
    async fn load_or_build_ivf(&self) -> Result<IvfModel> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Err(Error::invalid_input(
                "dataset not set before loading or building IVF",
                location!(),
            ));
        };

        let dim = utils::get_vector_dim(dataset.schema(), &self.column)?;
        match &self.ivf {
            Some(ivf) => {
                if self.retrain {
                    // retrain the IVF model with the existing indices
                    let mut ivf_params = IvfBuildParams::new(ivf.num_partitions());
                    ivf_params.retrain = true;

                    super::build_ivf_model(
                        dataset,
                        &self.column,
                        dim,
                        self.distance_type,
                        &ivf_params,
                    )
                    .await
                } else {
                    Ok(ivf.clone())
                }
            }
            None => {
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
        if self.quantizer.is_some() && !self.retrain {
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
            Some(q) => {
                let mut q = q.clone();
                if self.retrain {
                    q.retrain(&training_data)?;
                }
                q
            }
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
        if data.is_none() {
            // If we don't specify the shuffle reader, it's going to re-read the
            // dataset and duplicate the data.
            self.shuffle_reader = Some(Arc::new(EmptyReader));

            return Ok(self);
        }
        let data = data.unwrap();

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
                    ivf_transformer.transform(&batch)
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
        let Some(ivf) = self.ivf.as_mut() else {
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

        let reader = reader.clone();
        let existing_indices = Arc::new(self.existing_indices.clone());
        let distance_type = self.distance_type;
        let column = self.column.clone();
        let frag_reuse_index = self.frag_reuse_index.clone();
        let build_iter = (0..ivf.num_partitions()).map(move |partition| {
            let reader = reader.clone();
            let existing_indices = existing_indices.clone();
            let distance_type = distance_type;
            let quantizer = quantizer.clone();
            let sub_index_params = sub_index_params.clone();
            let column = column.clone();
            let frag_reuse_index = frag_reuse_index.clone();
            async move {
                let (batches, loss) = Self::take_partition_batches(
                    partition,
                    existing_indices.as_ref(),
                    reader.as_ref(),
                )
                .await?;

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
        reader: &dyn ShuffleReader,
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

            let part_storage = existing_index.load_partition_storage(part_id).await?;
            let mut part_batches = part_storage.to_batches()?.collect::<Vec<_>>();
            // for PQ, the PQ codes are transposed, so we need to transpose them back
            if matches!(Q::quantization_type(), QuantizationType::Product) {
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
            batches.extend(part_batches);
        }

        let mut loss = 0.0;
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

        let storage_schema: Schema =
            (&arrow_schema::Schema::new(vec![ROW_ID_FIELD.clone(), quantizer.field()]))
                .try_into()?;
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

    // take vectors from the dataset
    // used for reading vectors from existing indices
    #[allow(dead_code)]
    async fn take_vectors(
        dataset: &Arc<Dataset>,
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
            let batch = batch.try_with_column(
                ROW_ID_FIELD.clone(),
                Arc::new(UInt64Array::from(chunk.to_vec())),
            )?;
            batches.push(batch);
        }
        Ok(batches)
    }
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
