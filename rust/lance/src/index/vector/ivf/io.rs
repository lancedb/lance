// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;
use std::{cmp::Reverse, pin::Pin};

use arrow::compute::concat;
use arrow::datatypes::Float32Type;
use arrow_array::{
    cast::AsArray, types::UInt64Type, Array, FixedSizeListArray, RecordBatch, UInt32Array,
};
use futures::stream::Peekable;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_arrow::*;
use lance_core::datatypes::Schema;
use lance_core::Error;
use lance_file::reader::FileReader;
use lance_file::writer::FileWriter;
use lance_index::scalar::IndexWriter;
use lance_index::vector::hnsw::builder::HNSW_METADATA_KEY;
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfData;
use lance_index::vector::pq::storage::ProductQuantizationStorage;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::sq::storage::ScalarQuantizationStorage;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::SQ_CODE_COLUMN;
use lance_index::vector::{
    graph::memory::InMemoryVectorStorage,
    hnsw::{builder::HnswBuildParams, HNSWBuilder, HnswMetadata},
    PART_ID_COLUMN, PQ_CODE_COLUMN,
};
use lance_io::encodings::plain::PlainEncoder;
use lance_io::object_store::ObjectStore;
use lance_io::traits::Writer;
use lance_io::ReadBatchParams;
use lance_linalg::{distance::MetricType, MatrixView};
use lance_table::format::SelfDescribingFileReader;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::{location, Location};
use tempfile::TempDir;
use tokio::sync::Semaphore;

use super::{IVFIndex, Ivf};
use crate::index::vector::pq::PQIndex;
use crate::{dataset::ROW_ID, Dataset};
use crate::{utils, Result};

// TODO: make it configurable, limit by the number of CPU cores & memory
const HNSW_PARTITIONS_BUILD_PARRALLEL: usize = 16;

/// Merge streams with the same partition id and collect PQ codes and row IDs.
async fn merge_streams(
    streams_heap: &mut BinaryHeap<(Reverse<u32>, usize)>,
    new_streams: &mut [Pin<Box<Peekable<impl Stream<Item = Result<RecordBatch>>>>>],
    part_id: u32,
    code_column: &str,
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
                return Err(Error::IO {
                    message: format!("failed to read batch: {}", e),
                    location: location!(),
                });
            }
            None => {
                return Err(Error::IO {
                    message: "failed to read batch: unexpected end of stream".to_string(),
                    location: location!(),
                });
            }
        };

        let codes = Arc::new(
            batch
                .column_by_name(code_column)
                .expect("pq code column not found")
                .as_fixed_size_list()
                .clone(),
        );
        let row_ids: Arc<dyn Array> = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .expect("row id column not found")
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        code_array.push(codes);
        row_id_array.push(row_ids);

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
                return Err(Error::IO {
                    message: format!("IVF Shuffler::failed to read batch: {}", e),
                    location: location!(),
                });
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
    ivf: &mut Ivf,
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
                    return Err(Error::IO {
                        message: format!("failed to read batch: {}", e),
                        location: location!(),
                    });
                }
                None => {
                    return Err(Error::IO {
                        message: "failed to read batch: end of stream".to_string(),
                        location: location!(),
                    });
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
                let sub_index = idx.load_partition(part_id as usize, true).await?;
                let pq_index =
                    sub_index
                        .as_any()
                        .downcast_ref::<PQIndex>()
                        .ok_or(Error::Index {
                            message: "Invalid sub index".to_string(),
                            location: location!(),
                        })?;
                if let Some(pq_code) = pq_index.code.as_ref() {
                    let fsl = Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            pq_code.as_ref().clone(),
                            pq_index.pq.num_sub_vectors() as i32,
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
            PQ_CODE_COLUMN,
            &mut pq_array,
            &mut row_id_array,
        )
        .await?;

        let total_records = row_id_array.iter().map(|a| a.len()).sum::<usize>();
        ivf.add_partition(writer.tell().await?, total_records as u32);
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
pub(super) async fn write_hnsw_pq_index_partitions(
    dataset: &Dataset,
    column: &str,
    metric_type: MetricType,
    hnsw_params: &HnswBuildParams,
    writer: &mut FileWriter<ManifestDescribing>,
    mut auxiliary_writer: Option<&mut FileWriter<ManifestDescribing>>,
    ivf: &mut Ivf,
    pq: Arc<dyn ProductQuantizer>,
    streams: Option<Vec<impl Stream<Item = Result<RecordBatch>>>>,
    _existing_indices: Option<&[&IVFIndex]>,
) -> Result<(Vec<HnswMetadata>, IvfData)> {
    let dataset = Arc::new(dataset.clone());
    let column = Arc::new(column.to_owned());
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
                    return Err(Error::IO {
                        message: format!("failed to read batch: {}", e),
                        location: location!(),
                    });
                }
                None => {
                    return Err(Error::IO {
                        message: "failed to read batch: end of stream".to_string(),
                        location: location!(),
                    });
                }
            }
        }
    }

    let object_store = ObjectStore::local();
    let mut part_files = Vec::with_capacity(ivf.num_partitions());
    let mut aux_part_files = Vec::with_capacity(ivf.num_partitions());
    let tmp_part_dir = Path::from_filesystem_path(TempDir::new()?)?;
    let mut tasks = Vec::with_capacity(ivf.num_partitions());
    let sem = Arc::new(Semaphore::new(HNSW_PARTITIONS_BUILD_PARRALLEL));
    for part_id in 0..ivf.num_partitions() {
        part_files.push(tmp_part_dir.child(format!("hnsw_part_{}", part_id)));
        aux_part_files.push(tmp_part_dir.child(format!("hnsw_part_aux_{}", part_id)));

        let mut pq_array: Vec<Arc<dyn Array>> = vec![];
        let mut row_id_array: Vec<Arc<dyn Array>> = vec![];
        merge_streams(
            &mut streams_heap,
            &mut new_streams,
            part_id as u32,
            PQ_CODE_COLUMN,
            &mut pq_array,
            &mut row_id_array,
        )
        .await?;

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
        let column = column.clone();
        let hnsw_params = hnsw_params.clone();
        let pq = pq.clone();
        let sem = sem.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore error");

            log::debug!("Building HNSW partition {}", part_id);
            let result = build_hnsw_partition(
                dataset,
                column,
                metric_type,
                hnsw_params,
                part_writer,
                aux_part_writer,
                pq,
                row_id_array,
                pq_array,
            )
            .await;
            log::debug!("Finished building HNSW partition {}", part_id);
            result
        }));
    }

    let mut aux_ivf = IvfData::empty();
    let mut hnsw_metadata = Vec::with_capacity(ivf.num_partitions());
    for (part_id, task) in tasks.into_iter().enumerate() {
        let length = task.await??;

        let (part_file, aux_part_file) = (&part_files[part_id], &aux_part_files[part_id]);
        let part_reader =
            FileReader::try_new_self_described(&object_store, part_file, None).await?;

        let offset = writer.tell().await?;
        let batches = futures::stream::iter(0..part_reader.num_batches())
            .map(|batch_id| {
                part_reader.read_batch(
                    batch_id as i32,
                    ReadBatchParams::RangeFull,
                    part_reader.schema(),
                    None,
                )
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        writer.write(&batches).await?;
        ivf.add_partition(offset, length as u32);
        hnsw_metadata.push(serde_json::from_str(
            part_reader.schema().metadata[HNSW_METADATA_KEY].as_str(),
        )?);

        if let Some(aux_writer) = auxiliary_writer.as_mut() {
            let aux_part_reader =
                FileReader::try_new_self_described(&object_store, aux_part_file, None).await?;

            let batches = futures::stream::iter(0..aux_part_reader.num_batches())
                .map(|batch_id| {
                    aux_part_reader.read_batch(
                        batch_id as i32,
                        ReadBatchParams::RangeFull,
                        aux_part_reader.schema(),
                        None,
                    )
                })
                .buffered(num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;

            let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            aux_writer.write(&batches).await?;
            aux_ivf.add_partition(num_rows as u32);
        }
    }

    Ok((hnsw_metadata, aux_ivf))
}

#[allow(clippy::too_many_arguments)]
async fn build_hnsw_partition(
    dataset: Arc<Dataset>,
    column: Arc<String>,
    metric_type: MetricType,
    hnsw_params: Arc<HnswBuildParams>,
    mut writer: FileWriter<ManifestDescribing>,
    mut aux_writer: Option<FileWriter<ManifestDescribing>>,
    pq: Arc<dyn ProductQuantizer>,
    row_id_array: Vec<Arc<dyn Array>>,
    pq_array: Vec<Arc<dyn Array>>,
) -> Result<usize> {
    let projection = Arc::new(dataset.schema().project(&[column.as_ref()])?);
    let mut vector_batches = Vec::with_capacity(row_id_array.len());
    for row_ids in row_id_array.iter() {
        let array = dataset
            .take_rows(row_ids.as_primitive::<UInt64Type>().values(), &projection)
            .await?
            .column_by_name(column.as_ref())
            .expect("row id column not found")
            .clone();
        vector_batches.push(array);
    }

    let pq = pq.clone();
    let build_with_pq = aux_writer.is_some();

    let (hnsw, pq_storage) = utils::tokio::spawn_cpu(move || {
        build_hnsw_index(
            metric_type,
            (*hnsw_params).clone(),
            row_id_array,
            pq_array,
            vector_batches,
            build_with_pq,
            pq,
        )
    })
    .await?;

    writer.add_metadata(
        HNSW_METADATA_KEY,
        serde_json::to_string(&hnsw.metadata())?.as_str(),
    );
    let length = hnsw.write_levels(&mut writer).await?;
    writer.finish().await?;

    if let Some(pq_storage) = pq_storage {
        let aux_writer = aux_writer.as_mut().unwrap();
        pq_storage.write_partition(aux_writer).await?;
        aux_writer.finish().await?;
    }

    Ok(length)
}

fn build_hnsw_index(
    metric_type: MetricType,
    hnsw_params: HnswBuildParams,
    row_ids_array: Vec<Arc<dyn Array>>,
    pq_array: Vec<Arc<dyn Array>>,
    vector_array: Vec<Arc<dyn Array>>,
    build_with_pq: bool,
    pq: Arc<dyn ProductQuantizer>,
) -> Result<(HNSW, Option<ProductQuantizationStorage>)> {
    let vector_arrs = vector_array
        .iter()
        .map(|arr| arr.as_ref())
        .collect::<Vec<_>>();
    let fsl = arrow_select::concat::concat(&vector_arrs)?;
    std::mem::drop(vector_array);

    let mat = Arc::new(MatrixView::<Float32Type>::try_from(
        fsl.as_fixed_size_list(),
    )?);
    let vec_store = Arc::new(InMemoryVectorStorage::new(mat.clone(), metric_type));
    let mut hnsw_builder = HNSWBuilder::with_params(hnsw_params, vec_store);
    let hnsw = hnsw_builder.build()?;

    let pq_storage = if build_with_pq {
        let pq_arrs = pq_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let pq_column = concat(&pq_arrs)?;
        std::mem::drop(pq_array);

        let row_ids_arrs = row_ids_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let row_ids_column = concat(&row_ids_arrs)?;
        std::mem::drop(row_ids_array);

        let pq_batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids_column, true),
            (PQ_CODE_COLUMN, pq_column, false),
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
            metric_type,
        )?;

        Some(pq_store)
    } else {
        None
    };

    Ok((hnsw, pq_storage))
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn write_hnsw_sq_index_partitions(
    dataset: &Dataset,
    column: &str,
    metric_type: MetricType,
    hnsw_params: &HnswBuildParams,
    writer: &mut FileWriter<ManifestDescribing>,
    mut auxiliary_writer: Option<&mut FileWriter<ManifestDescribing>>,
    ivf: &mut Ivf,
    sq: ScalarQuantizer,
    streams: Option<Vec<impl Stream<Item = Result<RecordBatch>>>>,
    _existing_indices: Option<&[&IVFIndex]>,
) -> Result<(Vec<HnswMetadata>, IvfData)> {
    let dataset = Arc::new(dataset.clone());
    let column = Arc::new(column.to_owned());
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
                    return Err(Error::IO {
                        message: format!("failed to read batch: {}", e),
                        location: location!(),
                    });
                }
                None => {
                    return Err(Error::IO {
                        message: "failed to read batch: end of stream".to_string(),
                        location: location!(),
                    });
                }
            }
        }
    }

    let object_store = ObjectStore::local();
    let mut part_files = Vec::with_capacity(ivf.num_partitions());
    let mut aux_part_files = Vec::with_capacity(ivf.num_partitions());
    let tmp_part_dir = Path::from_filesystem_path(TempDir::new()?)?;
    let mut tasks = Vec::with_capacity(ivf.num_partitions());
    let sem = Arc::new(Semaphore::new(HNSW_PARTITIONS_BUILD_PARRALLEL));
    for part_id in 0..ivf.num_partitions() {
        part_files.push(tmp_part_dir.child(format!("hnsw_part_{}", part_id)));
        aux_part_files.push(tmp_part_dir.child(format!("hnsw_part_aux_{}", part_id)));

        let mut code_array: Vec<Arc<dyn Array>> = vec![];
        let mut row_id_array: Vec<Arc<dyn Array>> = vec![];
        merge_streams(
            &mut streams_heap,
            &mut new_streams,
            part_id as u32,
            SQ_CODE_COLUMN,
            &mut code_array,
            &mut row_id_array,
        )
        .await?;

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
        let column = column.clone();
        let hnsw_params = hnsw_params.clone();
        let sq = sq.clone();
        let sem = sem.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore error");

            log::debug!("Building HNSW partition {}", part_id);
            let result = build_hnsw_sq_partition(
                dataset,
                column,
                metric_type,
                hnsw_params,
                part_writer,
                aux_part_writer,
                sq,
                row_id_array,
                code_array,
            )
            .await;
            log::debug!("Finished building HNSW partition {}", part_id);
            result
        }));
    }

    let mut aux_ivf = IvfData::empty();
    let mut hnsw_metadata = Vec::with_capacity(ivf.num_partitions());
    for (part_id, task) in tasks.into_iter().enumerate() {
        let length = task.await??;

        let (part_file, aux_part_file) = (&part_files[part_id], &aux_part_files[part_id]);
        let part_reader =
            FileReader::try_new_self_described(&object_store, part_file, None).await?;

        let offset = writer.tell().await?;
        let batches = futures::stream::iter(0..part_reader.num_batches())
            .map(|batch_id| {
                part_reader.read_batch(
                    batch_id as i32,
                    ReadBatchParams::RangeFull,
                    part_reader.schema(),
                    None,
                )
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        writer.write(&batches).await?;
        ivf.add_partition(offset, length as u32);
        hnsw_metadata.push(serde_json::from_str(
            part_reader.schema().metadata[HNSW_METADATA_KEY].as_str(),
        )?);

        if let Some(aux_writer) = auxiliary_writer.as_mut() {
            let aux_part_reader =
                FileReader::try_new_self_described(&object_store, aux_part_file, None).await?;

            let batches = futures::stream::iter(0..aux_part_reader.num_batches())
                .map(|batch_id| {
                    aux_part_reader.read_batch(
                        batch_id as i32,
                        ReadBatchParams::RangeFull,
                        aux_part_reader.schema(),
                        None,
                    )
                })
                .buffered(num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;

            let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            aux_writer.write(&batches).await?;
            aux_ivf.add_partition(num_rows as u32);
        }
    }

    Ok((hnsw_metadata, aux_ivf))
}

#[allow(clippy::too_many_arguments)]
async fn build_hnsw_sq_partition(
    dataset: Arc<Dataset>,
    column: Arc<String>,
    metric_type: MetricType,
    hnsw_params: Arc<HnswBuildParams>,
    mut writer: FileWriter<ManifestDescribing>,
    mut aux_writer: Option<FileWriter<ManifestDescribing>>,
    sq: ScalarQuantizer,
    row_id_array: Vec<Arc<dyn Array>>,
    code_array: Vec<Arc<dyn Array>>,
) -> Result<usize> {
    let projection = Arc::new(dataset.schema().project(&[column.as_ref()])?);
    let mut vector_batches = Vec::with_capacity(row_id_array.len());
    for row_ids in row_id_array.iter() {
        let array = dataset
            .take_rows(row_ids.as_primitive::<UInt64Type>().values(), &projection)
            .await?
            .column_by_name(column.as_ref())
            .expect("row id column not found")
            .clone();
        vector_batches.push(array);
    }

    let build_with_aux = aux_writer.is_some();
    let (hnsw, sq_storage) = utils::tokio::spawn_cpu(move || {
        build_hnsw_sq_index(
            metric_type,
            (*hnsw_params).clone(),
            row_id_array,
            code_array,
            vector_batches,
            build_with_aux,
            sq,
        )
    })
    .await?;

    writer.add_metadata(
        HNSW_METADATA_KEY,
        serde_json::to_string(&hnsw.metadata())?.as_str(),
    );
    let length = hnsw.write_levels(&mut writer).await?;
    writer.finish().await?;

    if let Some(code_storage) = sq_storage {
        let aux_writer = aux_writer.as_mut().unwrap();
        aux_writer
            .write_record_batch(code_storage.batch().clone())
            .await?;
        aux_writer.finish().await?;
    }

    Ok(length)
}

fn build_hnsw_sq_index(
    metric_type: MetricType,
    hnsw_params: HnswBuildParams,
    row_ids_array: Vec<Arc<dyn Array>>,
    code_array: Vec<Arc<dyn Array>>,
    vector_array: Vec<Arc<dyn Array>>,
    build_with_aux: bool,
    sq: ScalarQuantizer,
) -> Result<(HNSW, Option<ScalarQuantizationStorage>)> {
    let vector_arrs = vector_array
        .iter()
        .map(|arr| arr.as_ref())
        .collect::<Vec<_>>();
    let fsl = arrow_select::concat::concat(&vector_arrs)?;
    std::mem::drop(vector_array);

    let mat = Arc::new(MatrixView::<Float32Type>::try_from(
        fsl.as_fixed_size_list(),
    )?);
    let vec_store = Arc::new(InMemoryVectorStorage::new(mat.clone(), metric_type));
    let mut hnsw_builder = HNSWBuilder::with_params(hnsw_params, vec_store);
    let hnsw = hnsw_builder.build()?;

    let pq_storage = if build_with_aux {
        let code_arrs = code_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let code_column = concat(&code_arrs)?;
        std::mem::drop(code_array);

        let row_ids_arrs = row_ids_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let row_ids_column = concat(&row_ids_arrs)?;
        std::mem::drop(row_ids_array);

        let pq_batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids_column, true),
            (SQ_CODE_COLUMN, code_column, false),
        ])?;
        let pq_store =
            ScalarQuantizationStorage::new(sq.num_bits(), metric_type, sq.bounds(), pq_batch)?;

        Some(pq_store)
    } else {
        None
    };

    Ok((hnsw, pq_storage))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::index::{vector::VectorIndexParams, DatasetIndexExt, DatasetIndexInternalExt};
    use arrow_array::RecordBatchIterator;
    use arrow_schema::{Field, Schema};
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

        let idx_params = VectorIndexParams::ivf_pq(2, 8, 2, false, MetricType::L2, 50);
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
            .open_vector_index(&indices[0].name, &indices[0].uuid.to_string())
            .await
            .unwrap();
        let _ivf_idx = idx
            .as_any()
            .downcast_ref::<IVFIndex>()
            .expect("Invalid index type");

        //let indices = /ds.
    }
}
