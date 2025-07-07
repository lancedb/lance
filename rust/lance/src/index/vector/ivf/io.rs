// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BinaryHeap;
use std::sync::{Arc, LazyLock};
use std::time::Instant;
use std::{cmp::Reverse, pin::Pin};

use super::IVFIndex;
use crate::dataset::ROW_ID;
use crate::index::vector::pq::{build_pq_storage, PQIndex};
use arrow::compute::concat;
use arrow_array::UInt64Array;
use arrow_array::{
    cast::AsArray, types::UInt64Type, Array, FixedSizeListArray, RecordBatch, UInt32Array,
};
use futures::stream::Peekable;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_arrow::*;
use lance_core::datatypes::Schema;
use lance_core::traits::DatasetTakeRows;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::Error;
use lance_file::reader::FileReader;
use lance_file::writer::FileWriter;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::IndexWriter;
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::hnsw::{builder::HnswBuildParams, HnswMetadata};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::storage::transpose;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::{Quantization, Quantizer};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_io::encodings::plain::PlainEncoder;
use lance_io::object_store::ObjectStore;
use lance_io::traits::Writer;
use lance_io::ReadBatchParams;
use lance_linalg::distance::{DistanceType, MetricType};
use lance_linalg::kernels::normalize_fsl;
use lance_table::format::SelfDescribingFileReader;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::location;
use tempfile::TempDir;
use tokio::sync::Semaphore;

use crate::Result;

// TODO: make it configurable, limit by the number of CPU cores & memory
static HNSW_PARTITIONS_BUILD_PARALLEL: LazyLock<usize> =
    LazyLock::new(get_num_compute_intensive_cpus);

/// Merge streams with the same partition id and collect PQ codes and row IDs.
async fn merge_streams(
    streams_heap: &mut BinaryHeap<(Reverse<u32>, usize)>,
    new_streams: &mut [Pin<Box<Peekable<impl Stream<Item = Result<RecordBatch>>>>>],
    part_id: u32,
    code_column: Option<&str>,
    code_array: &mut Vec<Arc<dyn Array>>,
    row_id_array: &mut Vec<Arc<dyn Array>>,
) -> Result<()> {
    while let Some((Reverse(stream_part_id), stream_idx)) = streams_heap.pop() {
        if stream_part_id != part_id {
            streams_heap.push((Reverse(stream_part_id), stream_idx));
            break;
        }

        let mut stream = new_streams[stream_idx].as_mut();
        let batch = match stream.next().await {
            Some(Ok(batch)) => batch,
            Some(Err(e)) => {
                return Err(Error::io(
                    format!("failed to read batch: {}", e),
                    location!(),
                ));
            }
            None => {
                return Err(Error::io(
                    "failed to read batch: unexpected end of stream".to_string(),
                    location!(),
                ));
            }
        };

        let row_ids: Arc<dyn Array> = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .expect("row id column not found")
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        row_id_array.push(row_ids);

        if let Some(column) = code_column {
            let codes = Arc::new(
                batch
                    .column_by_name(column)
                    .ok_or_else(|| Error::Index {
                        message: format!("code column {} not found", column),
                        location: location!(),
                    })?
                    .as_fixed_size_list()
                    .clone(),
            );
            code_array.push(codes);
        }

        match stream.peek().await {
            Some(Ok(batch)) => {
                let part_ids: &UInt32Array = batch
                    .column_by_name(PART_ID_COLUMN)
                    .expect("part id column not found")
                    .as_primitive();
                if !part_ids.is_empty() {
                    streams_heap.push((Reverse(part_ids.value(0)), stream_idx));
                }
            }
            Some(Err(e)) => {
                return Err(Error::io(
                    format!("IVF Shuffler::failed to read batch: {}", e),
                    location!(),
                ));
            }
            None => {}
        }
    }
    Ok(())
}

/// Write each partition of IVF_PQ index to the index file.
///
/// Parameters
/// ----------
/// `writer`: Index file writer.
/// `ivf`: IVF index to be written.
/// `streams`: RecordBatch stream of PQ codes and row ids, sorted by PQ code.
/// `existing_partitions`: Existing IVF indices to be merged. Can be zero or more.
///
/// These existing partitions must have the same centroids and PQ codebook.
///
/// TODO: migrate this function to `lance-index` crate.
pub(super) async fn write_pq_partitions(
    writer: &mut dyn Writer,
    ivf: &mut IvfModel,
    streams: Option<Vec<impl Stream<Item = Result<RecordBatch>>>>,
    existing_indices: Option<&[&IVFIndex]>,
) -> Result<()> {
    // build the initial heap
    // TODO: extract heap sort to a separate function.
    let mut streams_heap = BinaryHeap::new();
    let mut new_streams = vec![];

    if let Some(streams) = streams {
        for stream in streams {
            let mut stream = Box::pin(stream.peekable());

            match stream.as_mut().peek().await {
                Some(Ok(batch)) => {
                    let part_ids: &UInt32Array = batch
                        .column_by_name(PART_ID_COLUMN)
                        .expect("part id column not found")
                        .as_primitive();
                    let part_id = part_ids.values()[0];
                    streams_heap.push((Reverse(part_id), new_streams.len()));
                    new_streams.push(stream);
                }
                Some(Err(e)) => {
                    return Err(Error::io(
                        format!("failed to read batch: {}", e),
                        location!(),
                    ));
                }
                None => {
                    return Err(Error::io(
                        "failed to read batch: end of stream".to_string(),
                        location!(),
                    ));
                }
            }
        }
    }

    for part_id in 0..ivf.num_partitions() as u32 {
        let start = Instant::now();
        let mut pq_array: Vec<Arc<dyn Array>> = vec![];
        let mut row_id_array: Vec<Arc<dyn Array>> = vec![];

        if let Some(&previous_indices) = existing_indices.as_ref() {
            for &idx in previous_indices.iter() {
                let sub_index = idx
                    .load_partition(part_id as usize, true, &NoOpMetricsCollector)
                    .await?;
                let pq_index =
                    sub_index
                        .as_any()
                        .downcast_ref::<PQIndex>()
                        .ok_or(Error::Index {
                            message: "Invalid sub index".to_string(),
                            location: location!(),
                        })?;
                if let Some(pq_code) = pq_index.code.as_ref() {
                    let original_pq_codes = transpose(
                        pq_code,
                        pq_index.pq.num_sub_vectors,
                        pq_code.len() / pq_index.pq.code_dim(),
                    );
                    let fsl = Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            original_pq_codes,
                            pq_index.pq.code_dim() as i32,
                        )
                        .unwrap(),
                    );
                    pq_array.push(fsl);
                    row_id_array.push(pq_index.row_ids.as_ref().unwrap().clone());
                }
            }
        }

        // Merge all streams with the same partition id.
        merge_streams(
            &mut streams_heap,
            &mut new_streams,
            part_id,
            Some(PQ_CODE_COLUMN),
            &mut pq_array,
            &mut row_id_array,
        )
        .await?;

        let total_records = row_id_array.iter().map(|a| a.len()).sum::<usize>();
        ivf.add_partition_with_offset(writer.tell().await?, total_records as u32);
        if total_records > 0 {
            let pq_refs = pq_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(writer, &pq_refs).await?;

            let row_ids_refs = row_id_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(writer, row_ids_refs.as_slice()).await?;
        }
        log::info!(
            "Wrote partition {} in {} ms",
            part_id,
            start.elapsed().as_millis()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn write_hnsw_quantization_index_partitions(
    dataset: Arc<dyn DatasetTakeRows>,
    column: &str,
    distance_type: DistanceType,
    hnsw_params: &HnswBuildParams,
    writer: &mut FileWriter<ManifestDescribing>,
    mut auxiliary_writer: Option<&mut FileWriter<ManifestDescribing>>,
    ivf: &mut IvfModel,
    quantizer: Quantizer,
    streams: Option<Vec<impl Stream<Item = Result<RecordBatch>>>>,
    existing_indices: Option<&[&IVFIndex]>,
) -> Result<(Vec<HnswMetadata>, IvfModel)> {
    let hnsw_params = Arc::new(hnsw_params.clone());

    let mut streams_heap = BinaryHeap::new();
    let mut new_streams = vec![];
    if let Some(streams) = streams {
        for stream in streams {
            let mut stream = Box::pin(stream.peekable());

            match stream.as_mut().peek().await {
                Some(Ok(batch)) => {
                    let part_ids: &UInt32Array = batch
                        .column_by_name(PART_ID_COLUMN)
                        .expect("part id column not found")
                        .as_primitive();
                    let part_id = part_ids.values()[0];
                    streams_heap.push((Reverse(part_id), new_streams.len()));
                    new_streams.push(stream);
                }
                Some(Err(e)) => {
                    return Err(Error::io(
                        format!("failed to read batch: {}", e),
                        location!(),
                    ));
                }
                None => {
                    return Err(Error::io(
                        "failed to read batch: end of stream".to_string(),
                        location!(),
                    ));
                }
            }
        }
    }

    let object_store = ObjectStore::local();
    let mut part_files = Vec::with_capacity(ivf.num_partitions());
    let mut aux_part_files = Vec::with_capacity(ivf.num_partitions());
    let tmp_part_dir = Path::from_filesystem_path(TempDir::new()?)?;
    let mut tasks = Vec::with_capacity(ivf.num_partitions());
    let sem = Arc::new(Semaphore::new(*HNSW_PARTITIONS_BUILD_PARALLEL));
    for part_id in 0..ivf.num_partitions() {
        part_files.push(tmp_part_dir.child(format!("hnsw_part_{}", part_id)));
        aux_part_files.push(tmp_part_dir.child(format!("hnsw_part_aux_{}", part_id)));

        let mut code_array: Vec<Arc<dyn Array>> = vec![];
        let mut row_id_array: Vec<Arc<dyn Array>> = vec![];

        // We don't transform vectors to SQ codes while shuffling,
        // so we won't merge SQ codes from the stream.

        if let Some(&previous_indices) = existing_indices.as_ref() {
            for &idx in previous_indices.iter() {
                let sub_index = idx
                    .load_partition(part_id, true, &NoOpMetricsCollector)
                    .await?;
                let row_ids = Arc::new(UInt64Array::from_iter_values(sub_index.row_ids().cloned()));
                row_id_array.push(row_ids);
            }
        }

        let code_column = match &quantizer {
            Quantizer::Flat(_) => None,
            Quantizer::FlatBin(_) => None,
            Quantizer::Product(pq) => Some(pq.column()),
            Quantizer::Scalar(_) => None,
        };
        merge_streams(
            &mut streams_heap,
            &mut new_streams,
            part_id as u32,
            code_column,
            &mut code_array,
            &mut row_id_array,
        )
        .await?;

        if row_id_array.is_empty() {
            tasks.push(tokio::spawn(async { Ok(0) }));
            continue;
        }

        let (part_file, aux_part_file) = (&part_files[part_id], &aux_part_files[part_id]);
        let part_writer = FileWriter::<ManifestDescribing>::try_new(
            &object_store,
            part_file,
            Schema::try_from(writer.schema())?,
            &Default::default(),
        )
        .await?;

        let aux_part_writer = match auxiliary_writer.as_ref() {
            Some(writer) => Some(
                FileWriter::<ManifestDescribing>::try_new(
                    &object_store,
                    aux_part_file,
                    Schema::try_from(writer.schema())?,
                    &Default::default(),
                )
                .await?,
            ),
            None => None,
        };

        let dataset = dataset.clone();
        let column = column.to_owned();
        let hnsw_params = hnsw_params.clone();
        let quantizer = quantizer.clone();
        let sem = sem.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore error");

            log::debug!("Building HNSW partition {}", part_id);
            let result = build_hnsw_quantization_partition(
                dataset,
                &column,
                distance_type,
                hnsw_params,
                part_writer,
                aux_part_writer,
                quantizer,
                row_id_array,
                code_array,
            )
            .await;
            log::debug!("Finished building HNSW partition {}", part_id);
            result
        }));
    }

    let mut aux_ivf = IvfModel::empty();
    let mut hnsw_metadata = Vec::with_capacity(ivf.num_partitions());
    for (part_id, task) in tasks.into_iter().enumerate() {
        let offset = writer.len();
        let num_rows = task.await??;

        if num_rows == 0 {
            ivf.add_partition(0);
            aux_ivf.add_partition(0);
            hnsw_metadata.push(HnswMetadata::default());
            continue;
        }

        let (part_file, aux_part_file) = (&part_files[part_id], &aux_part_files[part_id]);
        let part_reader =
            FileReader::try_new_self_described(&object_store, part_file, None).await?;

        let batches = futures::stream::iter(0..part_reader.num_batches())
            .map(|batch_id| {
                part_reader.read_batch(
                    batch_id as i32,
                    ReadBatchParams::RangeFull,
                    part_reader.schema(),
                )
            })
            .buffered(object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;
        writer.write(&batches).await?;

        ivf.add_partition((writer.len() - offset) as u32);
        hnsw_metadata.push(serde_json::from_str(
            part_reader.schema().metadata[HNSW::metadata_key()].as_str(),
        )?);
        std::mem::drop(part_reader);
        object_store.delete(part_file).await?;

        if let Some(aux_writer) = auxiliary_writer.as_mut() {
            let aux_part_reader =
                FileReader::try_new_self_described(&object_store, aux_part_file, None).await?;

            let batches = futures::stream::iter(0..aux_part_reader.num_batches())
                .map(|batch_id| {
                    aux_part_reader.read_batch(
                        batch_id as i32,
                        ReadBatchParams::RangeFull,
                        aux_part_reader.schema(),
                    )
                })
                .buffered(object_store.io_parallelism())
                .try_collect::<Vec<_>>()
                .await?;
            std::mem::drop(aux_part_reader);
            object_store.delete(aux_part_file).await?;

            aux_writer.write(&batches).await?;
            aux_ivf.add_partition(num_rows as u32);
        }
    }

    Ok((hnsw_metadata, aux_ivf))
}

#[allow(clippy::too_many_arguments)]
async fn build_hnsw_quantization_partition(
    dataset: Arc<dyn DatasetTakeRows>,
    column: &str,
    metric_type: MetricType,
    hnsw_params: Arc<HnswBuildParams>,
    writer: FileWriter<ManifestDescribing>,
    aux_writer: Option<FileWriter<ManifestDescribing>>,
    quantizer: Quantizer,
    row_ids_array: Vec<Arc<dyn Array>>,
    code_array: Vec<Arc<dyn Array>>,
) -> Result<usize> {
    let row_ids_arrs = row_ids_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
    let row_ids = concat(&row_ids_arrs)?;
    std::mem::drop(row_ids_array);
    let num_rows = row_ids.len();

    let projection = Arc::new(dataset.schema().project(&[column])?);
    let mut vectors = dataset
        .take_rows(row_ids.as_primitive::<UInt64Type>().values(), &projection)
        .await?
        .column_by_name(column.as_ref())
        .expect("row id column not found")
        .clone();

    let mut metric_type = metric_type;
    if metric_type == MetricType::Cosine {
        // Normalize vectors for cosine similarity
        vectors =
            Arc::new(spawn_cpu(move || Ok(normalize_fsl(vectors.as_fixed_size_list())?)).await?);
        metric_type = MetricType::L2;
    }

    let build_hnsw =
        build_and_write_hnsw(vectors.clone(), (*hnsw_params).clone(), metric_type, writer);

    let build_store = match quantizer {
        Quantizer::Flat(_) => {
            return Err(Error::Index {
                message: "Flat quantizer is not supported for IVF_HNSW".to_string(),
                location: location!(),
            });
        }
        Quantizer::Product(pq) => tokio::spawn(build_and_write_pq_storage(
            metric_type,
            row_ids,
            code_array,
            pq,
            aux_writer.unwrap(),
        )),

        _ => unreachable!("IVF_HNSW_SQ has been moved to v2 index builder"),
    };

    let index_rows = futures::join!(build_hnsw, build_store).0?;
    assert!(
        index_rows >= num_rows,
        "index rows {} must be greater than or equal to num rows {}",
        index_rows,
        num_rows
    );
    Ok(num_rows)
}

async fn build_and_write_hnsw(
    vectors: Arc<dyn Array>,
    params: HnswBuildParams,
    distance_type: DistanceType,
    mut writer: FileWriter<ManifestDescribing>,
) -> Result<usize> {
    let batch = params.build(vectors, distance_type).await?.to_batch()?;
    let metadata = batch.schema_ref().metadata().clone();
    writer.write_record_batch(batch).await?;
    writer.finish_with_metadata(&metadata).await
}

async fn build_and_write_pq_storage(
    metric_type: MetricType,
    row_ids: Arc<dyn Array>,
    code_array: Vec<Arc<dyn Array>>,
    pq: ProductQuantizer,
    mut writer: FileWriter<ManifestDescribing>,
) -> Result<()> {
    let storage = spawn_cpu(move || {
        let storage = build_pq_storage(metric_type, row_ids, code_array, pq)?;
        Ok(storage)
    })
    .await?;

    writer.write_record_batch(storage.batch().clone()).await?;
    writer.finish().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::index::vector::ivf::v2;
    use crate::index::{vector::VectorIndexParams, DatasetIndexExt, DatasetIndexInternalExt};
    use crate::Dataset;
    use arrow_array::RecordBatchIterator;
    use arrow_schema::{Field, Schema};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::IndexType;
    use lance_testing::datagen::generate_random_array;

    #[tokio::test]
    async fn test_merge_multiple_indices() {
        const DIM: usize = 32;
        const TOTAL: usize = 1024;
        let vector_values = generate_random_array(TOTAL * DIM);
        let fsl =
            Arc::new(FixedSizeListArray::try_new_from_values(vector_values, DIM as i32).unwrap());

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            fsl.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![fsl.clone()]).unwrap();
        let batches =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), schema.clone());

        let tmp_uri = tempfile::tempdir().unwrap();

        let mut ds = Dataset::write(
            batches,
            tmp_uri.path().to_str().unwrap(),
            Default::default(),
        )
        .await
        .unwrap();

        let idx_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 50);
        ds.create_index(&["vector"], IndexType::Vector, None, &idx_params, true)
            .await
            .unwrap();
        let indices = ds.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(ds.get_fragments().len(), 1);

        let batches =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), schema.clone());
        ds.append(batches, None).await.unwrap();
        let indices = ds.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(ds.get_fragments().len(), 2);

        let idx = ds
            .open_vector_index(
                "vector",
                &indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let _ivf_idx = idx
            .as_any()
            .downcast_ref::<v2::IvfPq>()
            .expect("Invalid index type");

        //let indices = /ds.
    }
}
