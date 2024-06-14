// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, RecordBatch};
use futures::prelude::stream::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderMiddlewareChain, FilterExpression};
use lance_file::v2::{reader::FileReader, writer::FileWriter};
use lance_index::vector::quantizer::QuantizerBuildParams;
use lance_index::vector::storage::STORAGE_METADATA_KEY;
use lance_index::{
    pb,
    vector::{
        ivf::{
            storage::{IvfData, IVF_METADATA_KEY},
            IvfBuildParams,
        },
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
use lance_io::{
    object_store::ObjectStore, scheduler::ScanScheduler, stream::RecordBatchStreamAdapter,
    ReadBatchParams,
};
use lance_linalg::distance::DistanceType;
use log::info;
use object_store::path::Path;
use prost::Message;
use snafu::{location, Location};
use tempfile::TempDir;

use crate::Dataset;

use super::{utils, Ivf};

pub struct IvfIndexBuilder<S: IvfSubIndex, Q: Quantization + Clone> {
    dataset: Dataset,
    column: String,
    index_dir: Path,
    distance_type: DistanceType,
    shuffler: Arc<dyn Shuffler>,
    ivf_params: IvfBuildParams,
    sub_index_params: S::BuildParams,
    quantizer_params: Q::BuildParams,
    temp_dir: Path,
}

impl<S: IvfSubIndex, Q: Quantization + Clone> IvfIndexBuilder<S, Q> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Dataset,
        column: String,
        index_dir: Path,
        distance_type: DistanceType,
        shuffler: Box<dyn Shuffler>,
        ivf_params: IvfBuildParams,
        sub_index_params: S::BuildParams,
        quantizer_params: Q::BuildParams,
    ) -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let temp_dir = Path::from(temp_dir.path().to_str().unwrap());
        Ok(Self {
            dataset,
            column,
            index_dir,
            distance_type,
            shuffler: shuffler.into(),
            ivf_params,
            sub_index_params,
            quantizer_params,
            temp_dir,
        })
    }

    pub async fn build(&self) -> Result<()> {
        // step 1. train IVF & quantizer
        let ivf = self.load_or_build_ivf().await?;
        let quantizer = self.load_or_build_quantizer().await?;

        // step 2. shuffle data
        let reader = self
            .shuffle_data(ivf.centroids.clone(), quantizer.clone())
            .await?;
        let partition_build_order = (0..self.ivf_params.num_partitions)
            .map(|partition_id| reader.partiton_size(partition_id))
            .collect::<Result<Vec<_>>>()?
            // sort by partition size in descending order
            .into_iter()
            .enumerate()
            .map(|(idx, x)| (x, idx))
            .sorted()
            .rev()
            .map(|(_, idx)| idx)
            .collect::<Vec<_>>();

        // step 3. build sub index
        let mut partition_sizes = vec![(0, 0); self.ivf_params.num_partitions];
        for &partition in &partition_build_order {
            match reader.partiton_size(partition)? {
                0 => continue,
                _ => {
                    let partition_data =
                        reader.read_partition(partition).await?.ok_or(Error::io(
                            format!("partition {} is empty", partition).as_str(),
                            location!(),
                        ))?;
                    let batches = partition_data.try_collect::<Vec<_>>().await?;
                    let batch =
                        arrow::compute::concat_batches(&batches[0].schema(), batches.iter())?;

                    let sizes = self
                        .build_partition(partition, &batch, quantizer.clone())
                        .await?;
                    partition_sizes[partition] = sizes;
                }
            }
        }

        // step 4. merge all partitions
        self.merge_partitions(ivf.centroids, quantizer, partition_sizes)
            .await?;

        Ok(())
    }

    async fn load_or_build_ivf(&self) -> Result<Ivf> {
        let dim = utils::get_vector_dim(&self.dataset, &self.column)?;
        super::build_ivf_model(
            &self.dataset,
            &self.column,
            dim,
            self.distance_type,
            &self.ivf_params,
        )
        .await

        // TODO: load ivf model
    }

    async fn load_or_build_quantizer(&self) -> Result<Q> {
        let sample_size_hint = self.quantizer_params.sample_size();

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

        info!("Start to train quantizer");
        let start = std::time::Instant::now();
        let quantizer = Q::build(&training_data, dt, &self.quantizer_params)?;
        info!(
            "Trained quantizer in {:02} seconds",
            start.elapsed().as_secs_f32()
        );
        Ok(quantizer)
    }

    async fn shuffle_data(
        &self,
        centroids: FixedSizeListArray,
        quantizer: Q,
    ) -> Result<Box<dyn ShuffleReader>> {
        let transformer = Arc::new(lance_index::vector::ivf::new_ivf_with_quantizer(
            centroids,
            self.distance_type,
            &self.column,
            quantizer.into(),
            Some(0..self.ivf_params.num_partitions as u32),
        )?);

        let stream = self
            .dataset
            .scan()
            .batch_readahead(num_cpus::get() * 2)
            .project(&[self.column.as_str()])?
            .with_row_id()
            .try_into_stream()
            .await?;

        let mut transformed_stream = Box::pin(
            stream
                .map(move |batch| {
                    let ivf_transformer = transformer.clone();
                    tokio::spawn(async move { ivf_transformer.transform(&batch?) })
                })
                .buffered(num_cpus::get())
                .map(|x| x.unwrap())
                .peekable(),
        );

        let batch = transformed_stream.as_mut().peek().await;
        let schema = match batch {
            Some(Ok(b)) => b.schema(),
            Some(Err(e)) => panic!("do this better: error reading first batch: {:?}", e),
            None => panic!("no data"),
        };

        self.shuffler
            .shuffle(Box::new(RecordBatchStreamAdapter::new(
                schema,
                transformed_stream,
            )))
            .await
    }

    async fn build_partition(
        &self,
        part_id: usize,
        batch: &RecordBatch,
        quantizer: Q,
    ) -> Result<(usize, usize)> {
        let object_store = ObjectStore::local();

        // build quantized vector storage
        let storage =
            StorageBuilder::new(self.column.clone(), self.distance_type, quantizer).build(batch)?;
        let path = self.temp_dir.child(format!("storage_part{}", part_id));
        let writer = object_store.create(&path).await?;
        let mut writer = FileWriter::try_new(
            writer,
            path.to_string(),
            storage.schema().as_ref().try_into()?,
            Default::default(),
        )?;
        for batch in storage.to_batches()? {
            writer.write_batch(&batch).await?;
        }
        let storage_len = writer.finish().await? as usize;

        // build the sub index, with in-memory storage
        let sub_index = S::index_vectors(&storage, self.sub_index_params.clone())?;
        let path = self.temp_dir.child(format!("index_part{}", part_id));
        let writer = object_store.create(&path).await?;
        let index_batch = sub_index.to_batch()?;
        let mut writer = FileWriter::try_new(
            writer,
            path.to_string(),
            index_batch.schema_ref().as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&index_batch).await?;
        let index_len = writer.finish().await? as usize;

        Ok((storage_len, index_len))
    }

    async fn merge_partitions(
        &self,
        centroids: FixedSizeListArray,
        quantizer: Q,
        partition_sizes: Vec<(usize, usize)>,
    ) -> Result<()> {
        // prepare the final writers
        let storage_path = self.index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        let index_path = self.index_dir.child(INDEX_FILE_NAME);
        let mut storage_writer = None;
        let mut index_writer = FileWriter::try_new(
            self.dataset.object_store().create(&index_path).await?,
            index_path.to_string(),
            S::schema().as_ref().try_into()?,
            Default::default(),
        )?;

        // maintain the IVF partitions
        let mut storage_ivf = IvfData::empty();
        let mut index_ivf = IvfData::with_centroids(Arc::new(centroids));
        let mut partition_storage_metadata = Vec::with_capacity(partition_sizes.len());
        let mut partition_index_metadata = Vec::with_capacity(partition_sizes.len());
        let scheduler = ScanScheduler::new(Arc::new(ObjectStore::local()), 64);
        for (part_id, (storage_size, index_size)) in partition_sizes.into_iter().enumerate() {
            if storage_size == 0 {
                storage_ivf.add_partition(0);
                partition_storage_metadata.push(quantizer.metadata(None)?.to_string());
            } else {
                let storage_part_path = self.temp_dir.child(format!("storage_part{}", part_id));
                let reader = FileReader::try_open(
                    scheduler.open_file(&storage_part_path).await?,
                    None,
                    DecoderMiddlewareChain::default(),
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
                        storage_path.to_string(),
                        batch.schema_ref().as_ref().try_into()?,
                        Default::default(),
                    )?);
                }
                storage_writer.as_mut().unwrap().write_batch(&batch).await?;
                storage_ivf.add_partition(batch.num_rows() as u32);
                partition_storage_metadata.push(
                    reader
                        .schema()
                        .metadata
                        .get(STORAGE_METADATA_KEY)
                        .cloned()
                        .unwrap_or_default(),
                );
            }

            if index_size == 0 {
                index_ivf.add_partition(0);
                partition_index_metadata.push(String::new());
            } else {
                let index_part_path = self.temp_dir.child(format!("index_part{}", part_id));
                let reader = FileReader::try_open(
                    scheduler.open_file(&index_part_path).await?,
                    None,
                    DecoderMiddlewareChain::default(),
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
        }

        let mut storage_writer = storage_writer.unwrap();
        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, self.distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        storage_writer.add_schema_metadata(
            STORAGE_METADATA_KEY,
            serde_json::to_string(&partition_storage_metadata)?,
        );

        let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
        let index_metadata = IndexMetadata {
            index_type: S::name().to_string(),
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
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, ops::Range, sync::Arc};

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::hnsw::HNSW;

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
        let vectors = generate_random_array_with_range(1000 * DIM, range);
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
        let index_dir = tempdir().unwrap();
        let index_dir = Path::from(index_dir.path().to_str().unwrap());
        let shuffler = IvfShuffler::new(
            dataset.object_store().clone(),
            index_dir.child("shuffled"),
            ivf_params.num_partitions,
        );

        let builder = super::IvfIndexBuilder::<FlatIndex, FlatQuantizer>::new(
            dataset,
            "vector".to_owned(),
            index_dir,
            DistanceType::L2,
            Box::new(shuffler),
            ivf_params,
            (),
            (),
        )
        .unwrap();

        builder.build().await.unwrap();
    }

    #[tokio::test]
    async fn test_build_ivf_hnsw_sq() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let sq_params = SQBuildParams::default();
        let index_dir = tempdir().unwrap();
        let index_dir = Path::from(index_dir.path().to_str().unwrap());
        let shuffler = IvfShuffler::new(
            dataset.object_store().clone(),
            index_dir.child("shuffled"),
            ivf_params.num_partitions,
        );

        let builder = super::IvfIndexBuilder::<HNSW, ScalarQuantizer>::new(
            dataset,
            "vector".to_owned(),
            index_dir,
            DistanceType::L2,
            Box::new(shuffler),
            ivf_params,
            hnsw_params,
            sq_params,
        )
        .unwrap();
        builder.build().await.unwrap();
    }
}
