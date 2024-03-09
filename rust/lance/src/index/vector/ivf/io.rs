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

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use arrow::{
    array::{ArrayBuilder, ArrayDataBuilder},
    datatypes::Float32Type,
};
use arrow_array::{
    cast::AsArray, types::UInt64Type, Array, FixedSizeListArray, RecordBatch, UInt32Array,
};
use futures::{Stream, StreamExt};
use lance_arrow::*;
use lance_core::Error;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_index::vector::pq::ProductQuantizationStorage;
use lance_index::vector::{
    graph::memory::InMemoryVectorStorage,
    hnsw::{builder::HnswBuildParams, HNSWBuilder},
    PART_ID_COLUMN, PQ_CODE_COLUMN,
};
use lance_io::traits::Writer;
use lance_io::{encodings::plain::PlainEncoder, object_store::ObjectStore};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_table::io::manifest::ManifestDescribing;
use snafu::{location, Location};

use super::{IVFIndex, Ivf};
use crate::index::vector::pq::PQIndex;
use crate::Result;
use crate::{dataset::ROW_ID, Dataset};

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
pub(super) async fn write_pq_index_partitions(
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

            let pq_codes = Arc::new(
                batch
                    .column_by_name(PQ_CODE_COLUMN)
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
            pq_array.push(pq_codes);
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

pub(super) async fn write_hnsw_index_partitions(
    dataset: &Dataset,
    column: &str,
    metric_type: MetricType,
    hnsw_params: &HnswBuildParams,
    writer: &mut FileWriter<ManifestDescribing>,
    ivf: &mut Ivf,
    streams: Option<Vec<impl Stream<Item = Result<RecordBatch>>>>,
    existing_indices: Option<&[&IVFIndex]>,
) -> Result<()> {
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

            let pq_codes = Arc::new(
                batch
                    .column_by_name(PQ_CODE_COLUMN)
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
            pq_array.push(pq_codes);
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

        let total_records = row_id_array.iter().map(|a| a.len()).sum::<usize>();
        let offset = writer.tell().await?;

        if total_records > 0 {
            let projection = dataset.schema().project(&[column])?;
            let mut vector_batches = Vec::with_capacity(row_id_array.len());
            for row_ids in &row_id_array {
                let array = dataset
                    .take_rows(row_ids.as_primitive::<UInt64Type>().values(), &projection)
                    .await?
                    .column_by_name(column)
                    .expect("row id column not found")
                    .clone();
                vector_batches.push(array);
            }

            let vector_arrs = vector_batches
                .iter()
                .map(|arr| arr.as_ref())
                .collect::<Vec<_>>();
            let fsl = arrow_select::concat::concat(&vector_arrs).unwrap();
            let mat =
                Arc::new(MatrixView::<Float32Type>::try_from(fsl.as_fixed_size_list()).unwrap());
            let vec_store = Arc::new(InMemoryVectorStorage::new(mat.clone(), metric_type));

            let hnsw = HNSWBuilder::with_params(hnsw_params.clone(), vec_store).build()?;

            // Layout:
            // 1. Batches of HNSW nodes
            // 2. PQ codes
            // 3. Row IDs
            // 3. lance metadata & footer
            hnsw.write(writer).await?;
            let pq_refs = pq_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();

            let pq_offset = writer.tell().await? - offset;
            PlainEncoder::write(&mut writer.object_writer, &pq_refs).await?;
            let row_ids_refs = row_id_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
            PlainEncoder::write(&mut writer.object_writer, row_ids_refs.as_slice()).await?;
            writer.add_metadata("lance:pq", &pq_offset.to_string());
            writer.write_footer().await?;
        }

        let length = writer.tell().await? - offset;
        ivf.add_partition(offset, length as u32);
        log::info!(
            "Wrote partition {} in {} ms",
            part_id,
            start.elapsed().as_millis()
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        index::{vector::VectorIndexParams, DatasetIndexExt, DatasetIndexInternalExt},
        Dataset,
    };
    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::{Field, Schema};
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;
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
