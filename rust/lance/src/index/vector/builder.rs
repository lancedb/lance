// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{RecordBatch, UInt64Array};
use futures::prelude::stream::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::RecordBatchExt;
use lance_core::cache::FileMetadataCache;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, Result, ROW_ID_FIELD};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::FileReaderOptions;
use lance_file::v2::{reader::FileReader, writer::FileWriter};
use lance_index::vector::flat::storage::FlatStorage;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::quantizer::{QuantizationType, QuantizerBuildParams};
use lance_index::vector::storage::STORAGE_METADATA_KEY;
use lance_index::vector::v3::shuffler::IvfShufflerReader;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::VectorIndex;
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
use lance_io::scheduler::SchedulerConfig;
use lance_io::stream::RecordBatchStream;
use lance_io::{
    object_store::ObjectStore, scheduler::ScanScheduler, stream::RecordBatchStreamAdapter,
    ReadBatchParams,
};
use lance_linalg::distance::DistanceType;
use log::info;
use object_store::path::Path;
use prost::Message;
use snafu::{location, Location};
use tempfile::{tempdir, TempDir};
use tracing::{span, Level};

use crate::dataset::ProjectionRequest;
use crate::Dataset;

use super::utils;
use super::v2::IVFIndex;

// Builder for IVF index
// The builder will train the IVF model and quantizer, shuffle the dataset, and build the sub index
// for each partition.
// To build the index for the whole dataset, call `build` method.
// To build the index for given IVF, quantizer, data stream,
// call `with_ivf`, `with_quantizer`, `shuffle_data`, and `build` in order.
pub struct IvfIndexBuilder<S: IvfSubIndex, Q: Quantization + Clone> {
    dataset: Dataset,
    column: String,
    index_dir: Path,
    distance_type: DistanceType,
    shuffler: Arc<dyn Shuffler>,
    // build params, only needed for building new IVF, quantizer
    ivf_params: Option<IvfBuildParams>,
    quantizer_params: Option<Q::BuildParams>,
    sub_index_params: S::BuildParams,
    _temp_dir: TempDir, // store this for keeping the temp dir alive and clean up after build
    temp_dir: Path,

    // fields will be set during build
    ivf: Option<IvfModel>,
    quantizer: Option<Q>,
    shuffle_reader: Option<Box<dyn ShuffleReader>>,
    partition_sizes: Vec<(usize, usize)>,

    // fields for merging indices
    existing_indices: Vec<Arc<dyn VectorIndex>>,
}

impl<S: IvfSubIndex + 'static, Q: Quantization + Clone + 'static> IvfIndexBuilder<S, Q> {
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
    ) -> Result<Self> {
        let temp_dir = tempdir()?;
        let temp_dir_path = Path::from_filesystem_path(temp_dir.path())?;
        Ok(Self {
            dataset,
            column,
            index_dir,
            distance_type,
            shuffler: shuffler.into(),
            ivf_params,
            quantizer_params,
            sub_index_params,
            _temp_dir: temp_dir,
            temp_dir: temp_dir_path,
            // fields will be set during build
            ivf: None,
            quantizer: None,
            shuffle_reader: None,
            partition_sizes: Vec::new(),
            existing_indices: Vec::new(),
        })
    }

    pub fn new_incremental(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        sub_index_params: S::BuildParams,
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
        )
    }

    // build the index with the all data in the dataset,
    pub async fn build(&mut self) -> Result<()> {
        // step 1. train IVF & quantizer
        if self.ivf.is_none() {
            self.with_ivf(self.load_or_build_ivf().await?);
        }
        if self.quantizer.is_none() {
            self.with_quantizer(self.load_or_build_quantizer().await?);
        }

        // step 2. shuffle the dataset
        if self.shuffle_reader.is_none() {
            self.shuffle_dataset().await?;
        }

        // step 3. build partitions
        self.build_partitions().await?;

        // step 4. merge all partitions
        self.merge_partitions().await?;

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

    async fn load_or_build_ivf(&self) -> Result<IvfModel> {
        let ivf_params = self.ivf_params.as_ref().ok_or(Error::invalid_input(
            "IVF build params not set",
            location!(),
        ))?;
        let dim = utils::get_vector_dim(&self.dataset, &self.column)?;
        super::build_ivf_model(
            &self.dataset,
            &self.column,
            dim,
            self.distance_type,
            ivf_params,
        )
        .await

        // TODO: load ivf model
    }

    async fn load_or_build_quantizer(&self) -> Result<Q> {
        let quantizer_params = self.quantizer_params.as_ref().ok_or(Error::invalid_input(
            "quantizer build params not set",
            location!(),
        ))?;
        let sample_size_hint = quantizer_params.sample_size();

        let start = std::time::Instant::now();
        info!(
            "loading training data for quantizer. sample size: {}",
            sample_size_hint
        );
        let training_data =
            utils::maybe_sample_training_data(&self.dataset, &self.column, sample_size_hint)
                .await?;
        info!(
            "Finished loading training data in {:02} seconds",
            start.elapsed().as_secs_f32()
        );

        // If metric type is cosine, normalize the training data, and after this point,
        // treat the metric type as L2.
        let (training_data, dt) = if self.distance_type == DistanceType::Cosine {
            let training_data = lance_linalg::kernels::normalize_fsl(&training_data)?;
            (training_data, DistanceType::L2)
        } else {
            (training_data, self.distance_type)
        };

        let training_data = match (self.ivf.as_ref(), Q::use_residual(self.distance_type)) {
            (Some(ivf), true) => {
                let ivf_transformer = lance_index::vector::ivf::new_ivf_transformer(
                    ivf.centroids.clone().unwrap(),
                    dt,
                    vec![],
                );
                span!(Level::INFO, "compute residual for PQ training")
                    .in_scope(|| ivf_transformer.compute_residual(&training_data))?
            }
            _ => training_data,
        };

        info!("Start to train quantizer");
        let start = std::time::Instant::now();
        let quantizer = Q::build(&training_data, dt, quantizer_params)?;
        info!(
            "Trained quantizer in {:02} seconds",
            start.elapsed().as_secs_f32()
        );
        Ok(quantizer)
    }

    async fn shuffle_dataset(&mut self) -> Result<()> {
        let stream = self
            .dataset
            .scan()
            .batch_readahead(get_num_compute_intensive_cpus())
            .project(&[self.column.as_str()])?
            .with_row_id()
            .try_into_stream()
            .await?;
        self.shuffle_data(Some(stream)).await?;
        Ok(())
    }

    // shuffle the unindexed data and exsiting indices
    // data must be with schema | ROW_ID | vector_column |
    // the shuffled data will be with schema | ROW_ID | PART_ID | code_column |
    pub async fn shuffle_data(
        &mut self,
        data: Option<impl RecordBatchStream + Unpin + 'static>,
    ) -> Result<&mut Self> {
        if data.is_none() {
            return Ok(self);
        }
        let data = data.unwrap();

        let ivf = self.ivf.as_ref().ok_or(Error::invalid_input(
            "IVF not set before shuffle data",
            location!(),
        ))?;
        let quantizer = self.quantizer.clone().ok_or(Error::invalid_input(
            "quantizer not set before shuffle data",
            location!(),
        ))?;

        let transformer = Arc::new(
            lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
                ivf.centroids.clone().unwrap(),
                self.distance_type,
                &self.column,
                quantizer.into(),
                Some(0..ivf.num_partitions() as u32),
            )?,
        );
        let mut transformed_stream = Box::pin(
            data.map(move |batch| {
                let ivf_transformer = transformer.clone();
                tokio::spawn(async move { ivf_transformer.transform(&batch?) })
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
                self.shuffle_reader = Some(Box::new(IvfShufflerReader::new(
                    self.dataset.object_store.clone(),
                    self.temp_dir.clone(),
                    vec![0; ivf.num_partitions()],
                )));
                return Ok(self);
            }
        };

        self.shuffle_reader = Some(
            self.shuffler
                .shuffle(Box::new(RecordBatchStreamAdapter::new(
                    schema,
                    transformed_stream,
                )))
                .await?,
        );

        Ok(self)
    }

    async fn build_partitions(&mut self) -> Result<&mut Self> {
        let ivf = self.ivf.as_ref().ok_or(Error::invalid_input(
            "IVF not set before building partitions",
            location!(),
        ))?;

        let reader = self.shuffle_reader.as_ref().ok_or(Error::invalid_input(
            "shuffle reader not set before building partitions",
            location!(),
        ))?;

        let partition_build_order = (0..ivf.num_partitions())
            .map(|partition_id| reader.partition_size(partition_id))
            .collect::<Result<Vec<_>>>()?
            // sort by partition size in descending order
            .into_iter()
            .enumerate()
            .sorted_unstable_by(|(_, a), (_, b)| b.cmp(a))
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();

        let mut partition_sizes = vec![(0, 0); ivf.num_partitions()];
        for (i, &partition) in partition_build_order.iter().enumerate() {
            log::info!(
                "building partition {}, progress {}/{}",
                partition,
                i + 1,
                ivf.num_partitions(),
            );
            let mut batches = Vec::new();
            for existing_index in self.existing_indices.iter() {
                let existing_index = existing_index
                    .as_any()
                    .downcast_ref::<IVFIndex<S, Q>>()
                    .ok_or(Error::invalid_input(
                        "existing index is not IVF index",
                        location!(),
                    ))?;

                let part_storage = existing_index.load_partition_storage(partition).await?;
                batches.extend(
                    self.take_vectors(part_storage.row_ids().cloned().collect_vec().as_ref())
                        .await?,
                );
            }

            match reader.partition_size(partition)? {
                0 => continue,
                _ => {
                    let partition_data =
                        reader.read_partition(partition).await?.ok_or(Error::io(
                            format!("partition {} is empty", partition).as_str(),
                            location!(),
                        ))?;
                    batches.extend(partition_data.try_collect::<Vec<_>>().await?);
                }
            }

            let num_rows = batches.iter().map(|b| b.num_rows()).sum::<usize>();
            if num_rows == 0 {
                continue;
            }
            let mut batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
            if self.distance_type == DistanceType::Cosine {
                let vectors = batch
                    .column_by_name(&self.column)
                    .ok_or(Error::invalid_input(
                        format!("column {} not found", self.column).as_str(),
                        location!(),
                    ))?
                    .as_fixed_size_list();
                let vectors = lance_linalg::kernels::normalize_fsl(vectors)?;
                batch = batch.replace_column_by_name(&self.column, Arc::new(vectors))?;
            }

            let sizes = self.build_partition(partition, &batch).await?;
            partition_sizes[partition] = sizes;
            log::info!(
                "partition {} built, progress {}/{}",
                partition,
                i + 1,
                ivf.num_partitions()
            );
        }
        self.partition_sizes = partition_sizes;
        Ok(self)
    }

    async fn build_partition(&self, part_id: usize, batch: &RecordBatch) -> Result<(usize, usize)> {
        let quantizer = self.quantizer.clone().ok_or(Error::invalid_input(
            "quantizer not set before building partition",
            location!(),
        ))?;

        // build quantized vector storage
        let object_store = ObjectStore::local();
        let storage_len = {
            let storage = StorageBuilder::new(self.column.clone(), self.distance_type, quantizer)
                .build(batch)?;
            let path = self.temp_dir.child(format!("storage_part{}", part_id));
            let writer = object_store.create(&path).await?;
            let mut writer = FileWriter::try_new(
                writer,
                storage.schema().as_ref().try_into()?,
                Default::default(),
            )?;
            for batch in storage.to_batches()? {
                writer.write_batch(&batch).await?;
            }
            writer.finish().await? as usize
        };

        // build the sub index, with in-memory storage
        let index_len = {
            let vectors = batch[&self.column].as_fixed_size_list();
            let flat_storage = FlatStorage::new(vectors.clone(), self.distance_type);
            let sub_index = S::index_vectors(&flat_storage, self.sub_index_params.clone())?;
            let path = self.temp_dir.child(format!("index_part{}", part_id));
            let writer = object_store.create(&path).await?;
            let index_batch = sub_index.to_batch()?;
            let mut writer = FileWriter::try_new(
                writer,
                index_batch.schema_ref().as_ref().try_into()?,
                Default::default(),
            )?;
            writer.write_batch(&index_batch).await?;
            writer.finish().await? as usize
        };

        Ok((storage_len, index_len))
    }

    async fn merge_partitions(&mut self) -> Result<()> {
        let ivf = self.ivf.as_ref().ok_or(Error::invalid_input(
            "IVF not set before merge partitions",
            location!(),
        ))?;
        let quantizer = self.quantizer.clone().ok_or(Error::invalid_input(
            "quantizer not set before merge partitions",
            location!(),
        ))?;
        let partition_sizes = std::mem::take(&mut self.partition_sizes);
        if partition_sizes.is_empty() {
            return Err(Error::invalid_input("no partition to merge", location!()));
        }

        // prepare the final writers
        let storage_path = self.index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        let index_path = self.index_dir.child(INDEX_FILE_NAME);
        let mut storage_writer = None;
        let mut index_writer = FileWriter::try_new(
            self.dataset.object_store().create(&index_path).await?,
            S::schema().as_ref().try_into()?,
            Default::default(),
        )?;

        // maintain the IVF partitions
        let mut storage_ivf = IvfModel::empty();
        let mut index_ivf = IvfModel::new(ivf.centroids.clone().unwrap());
        let mut partition_index_metadata = Vec::with_capacity(partition_sizes.len());
        let obj_store = Arc::new(ObjectStore::local());
        let scheduler_config = SchedulerConfig::max_bandwidth(&obj_store);
        let scheduler = ScanScheduler::new(obj_store, scheduler_config);
        for (part_id, (storage_size, index_size)) in partition_sizes.into_iter().enumerate() {
            log::info!("merging partition {}/{}", part_id, ivf.num_partitions());
            if storage_size == 0 {
                storage_ivf.add_partition(0);
            } else {
                let storage_part_path = self.temp_dir.child(format!("storage_part{}", part_id));
                let reader = FileReader::try_open(
                    scheduler.open_file(&storage_part_path).await?,
                    None,
                    Arc::<DecoderPlugins>::default(),
                    &FileMetadataCache::no_cache(),
                    FileReaderOptions::default(),
                )
                .await?;
                let batches = reader
                    .read_stream(
                        ReadBatchParams::RangeFull,
                        u32::MAX,
                        1,
                        FilterExpression::no_filter(),
                    )?
                    .try_collect::<Vec<_>>()
                    .await?;
                let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
                if storage_writer.is_none() {
                    storage_writer = Some(FileWriter::try_new(
                        self.dataset.object_store().create(&storage_path).await?,
                        batch.schema_ref().as_ref().try_into()?,
                        Default::default(),
                    )?);
                }
                storage_writer.as_mut().unwrap().write_batch(&batch).await?;
                storage_ivf.add_partition(batch.num_rows() as u32);
            }

            if index_size == 0 {
                index_ivf.add_partition(0);
                partition_index_metadata.push(String::new());
            } else {
                let index_part_path = self.temp_dir.child(format!("index_part{}", part_id));
                let reader = FileReader::try_open(
                    scheduler.open_file(&index_part_path).await?,
                    None,
                    Arc::<DecoderPlugins>::default(),
                    &FileMetadataCache::no_cache(),
                    FileReaderOptions::default(),
                )
                .await?;
                let batches = reader
                    .read_stream(
                        ReadBatchParams::RangeFull,
                        u32::MAX,
                        1,
                        FilterExpression::no_filter(),
                    )?
                    .try_collect::<Vec<_>>()
                    .await?;
                let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;
                index_writer.write_batch(&batch).await?;
                index_ivf.add_partition(batch.num_rows() as u32);
                partition_index_metadata.push(
                    reader
                        .schema()
                        .metadata
                        .get(S::metadata_key())
                        .cloned()
                        .unwrap_or_default(),
                );
            }
            log::info!("merged partition {}/{}", part_id, ivf.num_partitions());
        }

        let mut storage_writer = storage_writer.unwrap();
        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, self.distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        // For now, each partition's metadata is just the quantizer,
        // it's all the same for now, so we just take the first one
        let storage_partition_metadata = vec![quantizer.metadata(None)?.to_string()];
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

        Ok(())
    }

    // take vectors from the dataset
    // used for reading vectors from existing indices
    async fn take_vectors(&self, row_ids: &[u64]) -> Result<Vec<RecordBatch>> {
        let column = self.column.clone();
        let object_store = self.dataset.object_store().clone();
        let projection = Arc::new(self.dataset.schema().project(&[column.as_str()])?);
        // arrow uses i32 for index, so we chunk the row ids to avoid large batch causing overflow
        let mut batches = Vec::new();
        for chunk in row_ids.chunks(object_store.block_size()) {
            let batch = self
                .dataset
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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range, sync::Arc};

    use arrow::datatypes::Float32Type;
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::hnsw::HNSW;

    use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::sq::ScalarQuantizer;
    use lance_index::vector::{
        flat::index::{FlatIndex, FlatQuantizer},
        ivf::IvfBuildParams,
        v3::shuffler::IvfShuffler,
    };
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_range;
    use object_store::path::Path;
    use tempfile::tempdir;

    use crate::Dataset;

    const DIM: usize = 32;

    async fn generate_test_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range::<Float32Type>(1000 * DIM, range);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema: Arc<_> = Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )])
        .with_metadata(metadata)
        .into();
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, array)
    }

    #[tokio::test]
    async fn test_build_ivf_flat() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::default();
        let index_dir: Path = tempdir().unwrap().path().to_str().unwrap().into();
        let shuffler = IvfShuffler::new(index_dir.child("shuffled"), ivf_params.num_partitions);

        super::IvfIndexBuilder::<FlatIndex, FlatQuantizer>::new(
            dataset,
            "vector".to_owned(),
            index_dir,
            DistanceType::L2,
            Box::new(shuffler),
            Some(ivf_params),
            Some(()),
            (),
        )
        .unwrap()
        .build()
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_build_ivf_pq() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::default();
        let pq_params = PQBuildParams::default();
        let index_dir: Path = tempdir().unwrap().path().to_str().unwrap().into();
        let shuffler = IvfShuffler::new(index_dir.child("shuffled"), ivf_params.num_partitions);

        super::IvfIndexBuilder::<FlatIndex, ProductQuantizer>::new(
            dataset,
            "vector".to_owned(),
            index_dir,
            DistanceType::L2,
            Box::new(shuffler),
            Some(ivf_params),
            Some(pq_params),
            (),
        )
        .unwrap()
        .build()
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_build_ivf_hnsw_sq() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let sq_params = SQBuildParams::default();
        let index_dir: Path = tempdir().unwrap().path().to_str().unwrap().into();
        let shuffler = IvfShuffler::new(index_dir.child("shuffled"), ivf_params.num_partitions);

        super::IvfIndexBuilder::<HNSW, ScalarQuantizer>::new(
            dataset,
            "vector".to_owned(),
            index_dir,
            DistanceType::L2,
            Box::new(shuffler),
            Some(ivf_params),
            Some(sq_params),
            hnsw_params,
        )
        .unwrap()
        .build()
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_build_ivf_hnsw_pq() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let pq_params = PQBuildParams::default();
        let index_dir: Path = tempdir().unwrap().path().to_str().unwrap().into();
        let shuffler = IvfShuffler::new(index_dir.child("shuffled"), ivf_params.num_partitions);

        super::IvfIndexBuilder::<HNSW, ProductQuantizer>::new(
            dataset,
            "vector".to_owned(),
            index_dir,
            DistanceType::L2,
            Box::new(shuffler),
            Some(ivf_params),
            Some(pq_params),
            hnsw_params,
        )
        .unwrap()
        .build()
        .await
        .unwrap();
    }
}
