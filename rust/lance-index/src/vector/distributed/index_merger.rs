// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index merging mechanisms for distributed vector index building

use crate::vector::shared::partition_merger::{
    write_unified_ivf_and_index_metadata, SupportedIvfIndexType,
};
use arrow::{compute::concat_batches, datatypes::Float32Type};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt64Array};
use futures::StreamExt as _;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result, ROW_ID_FIELD};
use snafu::location;
use std::ops::Range;
use std::sync::Arc;

use crate::pb;
use crate::vector::flat::index::FlatMetadata;
use crate::vector::ivf::storage::{IvfModel as IvfStorageModel, IVF_METADATA_KEY};
use crate::vector::pq::storage::{transpose, ProductQuantizationMetadata, PQ_METADATA_KEY};
use crate::vector::quantizer::QuantizerMetadata;
use crate::vector::sq::storage::{ScalarQuantizationMetadata, SQ_METADATA_KEY};
use crate::vector::storage::STORAGE_METADATA_KEY;
use crate::vector::{DISTANCE_TYPE_KEY, PQ_CODE_COLUMN, SQ_CODE_COLUMN};
use crate::IndexMetadata as IndexMetaSchema;
use crate::{INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::reader::{FileReader as V2Reader, FileReaderOptions as V2ReaderOptions};
use lance_file::writer::{FileWriter as V2Writer, FileWriter, FileWriterOptions};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::DistanceType;
use prost::Message;
use std::future::Future;
use std::pin::Pin;
use std::sync::LazyLock;

const DEFAULT_PARTITION_WINDOW_SIZE: usize = 512;
const PARTITION_WINDOW_SIZE_ENV: &str = "LANCE_IVF_PQ_MERGE_PARTITION_WINDOW_SIZE";
const DEFAULT_PARTITION_PREFETCH_WINDOW_COUNT: usize = 2;
const PARTITION_PREFETCH_WINDOW_COUNT_ENV: &str =
    "LANCE_IVF_PQ_MERGE_PARTITION_PREFETCH_WINDOW_COUNT";
static PARTITION_WINDOW_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var(PARTITION_WINDOW_SIZE_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PARTITION_WINDOW_SIZE)
});
static PARTITION_PREFETCH_WINDOW_COUNT: LazyLock<usize> = LazyLock::new(|| {
    std::env::var(PARTITION_PREFETCH_WINDOW_COUNT_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PARTITION_PREFETCH_WINDOW_COUNT)
});

/// Strict bitwise equality check for FixedSizeListArray values.
/// Returns true only if length, value_length and all underlying primitive values are equal.
fn fixed_size_list_equal(a: &FixedSizeListArray, b: &FixedSizeListArray) -> bool {
    if a.len() != b.len() || a.value_length() != b.value_length() {
        return false;
    }
    use arrow_schema::DataType;
    match (a.value_type(), b.value_type()) {
        (DataType::Float32, DataType::Float32) => {
            let va = a.values().as_primitive::<Float32Type>();
            let vb = b.values().as_primitive::<Float32Type>();
            va.values() == vb.values()
        }
        (DataType::Float64, DataType::Float64) => {
            let va = a.values().as_primitive::<arrow_array::types::Float64Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float64Type>();
            va.values() == vb.values()
        }
        (DataType::Float16, DataType::Float16) => {
            let va = a.values().as_primitive::<arrow_array::types::Float16Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float16Type>();
            va.values() == vb.values()
        }
        _ => false,
    }
}

/// Relaxed numeric equality check within tolerance to accommodate minor serialization
/// differences while still enforcing global-training invariants.
fn fixed_size_list_almost_equal(a: &FixedSizeListArray, b: &FixedSizeListArray, tol: f32) -> bool {
    if a.len() != b.len() || a.value_length() != b.value_length() {
        return false;
    }
    use arrow_schema::DataType;
    match (a.value_type(), b.value_type()) {
        (DataType::Float32, DataType::Float32) => {
            let va = a.values().as_primitive::<Float32Type>();
            let vb = b.values().as_primitive::<Float32Type>();
            let av = va.values();
            let bv = vb.values();
            if av.len() != bv.len() {
                return false;
            }
            for i in 0..av.len() {
                if (av[i] - bv[i]).abs() > tol {
                    return false;
                }
            }
            true
        }
        (DataType::Float64, DataType::Float64) => {
            let va = a.values().as_primitive::<arrow_array::types::Float64Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float64Type>();
            let av = va.values();
            let bv = vb.values();
            if av.len() != bv.len() {
                return false;
            }
            for i in 0..av.len() {
                if (av[i] - bv[i]).abs() > tol as f64 {
                    return false;
                }
            }
            true
        }
        (DataType::Float16, DataType::Float16) => {
            let va = a.values().as_primitive::<arrow_array::types::Float16Type>();
            let vb = b.values().as_primitive::<arrow_array::types::Float16Type>();
            let av = va.values();
            let bv = vb.values();
            if av.len() != bv.len() {
                return false;
            }
            for i in 0..av.len() {
                let da = av[i].to_f32();
                let db = bv[i].to_f32();
                if (da - db).abs() > tol {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// Initialize schema-level metadata on a writer for a given storage.
///
/// It writes the distance type and the storage metadata (as a vector payload),
/// and optionally the raw storage metadata under a storage-specific metadata
/// key (e.g. [`PQ_METADATA_KEY`] or [`SQ_METADATA_KEY`]).
fn init_writer_for_storage(
    w: &mut FileWriter,
    dt: DistanceType,
    storage_meta_json: &str,
    storage_meta_key: &str,
) -> Result<()> {
    // distance type
    w.add_schema_metadata(DISTANCE_TYPE_KEY, dt.to_string());
    // storage metadata (vector of one entry for future extensibility)
    let meta_vec_json = serde_json::to_string(&vec![storage_meta_json.to_string()])?;
    w.add_schema_metadata(STORAGE_METADATA_KEY, meta_vec_json);
    if !storage_meta_key.is_empty() {
        w.add_schema_metadata(storage_meta_key, storage_meta_json.to_string());
    }
    Ok(())
}

/// Create and initialize a unified writer for FLAT storage.
pub async fn init_writer_for_flat(
    object_store: &lance_io::object_store::ObjectStore,
    aux_out: &object_store::path::Path,
    d0: usize,
    dt: DistanceType,
) -> Result<FileWriter> {
    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(
            crate::vector::flat::storage::FLAT_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                d0 as i32,
            ),
            true,
        ),
    ]);
    let writer = object_store.create(aux_out).await?;
    let mut w = FileWriter::try_new(
        writer,
        LanceSchema::try_from(&arrow_schema)?,
        FileWriterOptions::default(),
    )?;
    let meta_json = serde_json::to_string(&FlatMetadata { dim: d0 })?;
    init_writer_for_storage(&mut w, dt, &meta_json, "")?;
    Ok(w)
}

/// Create and initialize a unified writer for PQ storage.
///
/// This always writes the codebook into the unified file and resets
/// `buffer_index` in the metadata to point at the new location.
pub async fn init_writer_for_pq(
    object_store: &lance_io::object_store::ObjectStore,
    aux_out: &object_store::path::Path,
    dt: DistanceType,
    pm: &ProductQuantizationMetadata,
) -> Result<FileWriter> {
    let num_bytes = if pm.nbits == 4 {
        pm.num_sub_vectors / 2
    } else {
        pm.num_sub_vectors
    };
    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                num_bytes as i32,
            ),
            true,
        ),
    ]);
    let writer = object_store.create(aux_out).await?;
    let mut w = FileWriter::try_new(
        writer,
        LanceSchema::try_from(&arrow_schema)?,
        FileWriterOptions::default(),
    )?;
    let mut pm_init = pm.clone();
    let cb = pm_init.codebook.as_ref().ok_or_else(|| Error::Index {
        message: "PQ codebook missing".to_string(),
        location: snafu::location!(),
    })?;
    let codebook_tensor: pb::Tensor = pb::Tensor::try_from(cb)?;
    let buf = Bytes::from(codebook_tensor.encode_to_vec());
    let pos = w.add_global_buffer(buf).await?;
    pm_init.set_buffer_index(pos);
    let pm_json = serde_json::to_string(&pm_init)?;
    init_writer_for_storage(&mut w, dt, &pm_json, PQ_METADATA_KEY)?;
    Ok(w)
}

/// Create and initialize a unified writer for SQ storage.
pub async fn init_writer_for_sq(
    object_store: &lance_io::object_store::ObjectStore,
    aux_out: &object_store::path::Path,
    dt: DistanceType,
    sq_meta: &ScalarQuantizationMetadata,
) -> Result<FileWriter> {
    let d0 = sq_meta.dim;
    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(
            SQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                d0 as i32,
            ),
            true,
        ),
    ]);
    let writer = object_store.create(aux_out).await?;
    let mut w = FileWriter::try_new(
        writer,
        LanceSchema::try_from(&arrow_schema)?,
        FileWriterOptions::default(),
    )?;
    let meta_json = serde_json::to_string(sq_meta)?;
    init_writer_for_storage(&mut w, dt, &meta_json, SQ_METADATA_KEY)?;
    Ok(w)
}

/// Stream and write a range of rows from reader into writer.
///
/// The caller is responsible for ensuring that `range` corresponds to a
/// contiguous row interval for a single IVF partition.
pub async fn write_partition_rows(
    reader: &V2Reader,
    w: &mut FileWriter,
    range: Range<usize>,
) -> Result<()> {
    let mut stream = reader.read_stream(
        lance_io::ReadBatchParams::Range(range),
        u32::MAX,
        4,
        lance_encoding::decoder::FilterExpression::no_filter(),
    )?;
    use futures::StreamExt as _;
    while let Some(rb) = stream.next().await {
        let rb = rb?;
        w.write_batch(&rb).await?;
    }
    Ok(())
}

/// Transpose the PQ code column for a batch and write it to the unified writer.
///
/// This helper assumes `batch` contains a contiguous range of rows for a single
/// IVF partition.
async fn write_partition_rows_pq_transposed(
    w: &mut FileWriter,
    mut batch: RecordBatch,
) -> Result<()> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(());
    }

    let pq_col = batch
        .column_by_name(PQ_CODE_COLUMN)
        .ok_or_else(|| Error::Index {
            message: format!("PQ column {} missing in auxiliary shard", PQ_CODE_COLUMN),
            location: location!(),
        })?;
    let pq_fsl = pq_col
        .as_fixed_size_list_opt()
        .ok_or_else(|| Error::Index {
            message: format!(
                "PQ column {} is not a FixedSizeList in auxiliary shard, got {}",
                PQ_CODE_COLUMN,
                pq_col.data_type(),
            ),
            location: location!(),
        })?;
    let num_bytes = pq_fsl.value_length() as usize;
    let values = pq_fsl.values().as_primitive::<UInt8Type>();
    let transposed_codes = transpose(values, num_rows, num_bytes);
    let transposed_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
        transposed_codes,
        num_bytes as i32,
    )?);
    batch = batch.replace_column_by_name(PQ_CODE_COLUMN, transposed_fsl)?;

    // Write in reasonably sized chunks to avoid huge batches.
    let batch_size: usize = 10_240;
    for offset in (0..num_rows).step_by(batch_size) {
        let len = std::cmp::min(batch_size, num_rows - offset);
        let slice = batch.slice(offset, len);
        w.write_batch(&slice).await?;
    }
    Ok(())
}

/// Detect and return supported index type from reader and schema.
///
/// This is a lightweight wrapper around SupportedIndexType::detect to keep
/// detection logic self-contained within this module.
fn detect_supported_index_type(
    reader: &V2Reader,
    schema: &ArrowSchema,
) -> Result<SupportedIvfIndexType> {
    SupportedIvfIndexType::detect_from_reader_and_schema(reader, schema)
}

/// Decode the fragment id from an encoded row id.
///
/// Row ids are stored as a 64-bit [RowAddress] where the upper 32 bits encode
/// the fragment id and the lower 32 bits encode the row offset.
fn decode_fragment_id_from_row_id(row_id_u64: u64) -> u32 {
    let addr = RowAddress::new_from_u64(row_id_u64);
    addr.fragment_id()
}

/// Compute a content-derived shard sort key for a partial auxiliary file.
///
/// The key is `(min_fragment_id, min_row_id, parent_dir_name)` where:
/// - `min_fragment_id` is the minimum fragment id observed among the first row
///   of each non-empty IVF partition.
/// - `min_row_id` is the minimum encoded row id (as `u64`) among the same
///   representative rows.
/// - `parent_dir_name` is the `partial_*` directory name extracted from
///   `aux_path` and used only as a final lexicographic tie-breaker.
///
/// This helper reads exactly one row per non-empty partition (the first row in
/// that partition) and never scans entire shards.
async fn compute_shard_content_key(
    sched: &std::sync::Arc<ScanScheduler>,
    _store: &lance_io::object_store::ObjectStore,
    aux_path: &object_store::path::Path,
) -> Result<(u32, u64, String)> {
    let fh = sched
        .open_file(aux_path, &CachedFileSize::unknown())
        .await?;
    let reader = V2Reader::try_open(
        fh,
        None,
        Arc::default(),
        &lance_core::cache::LanceCache::no_cache(),
        V2ReaderOptions::default(),
    )
    .await?;

    // Locate the ROW_ID_FIELD column to decode fragment / row ids.
    let schema_arrow: ArrowSchema = reader.schema().as_ref().into();
    let row_id_idx = schema_arrow
        .fields
        .iter()
        .position(|f| f.name() == ROW_ID_FIELD.name())
        .ok_or_else(|| Error::Index {
            message: "ROW_ID_FIELD missing in auxiliary shard".to_string(),
            location: location!(),
        })?;

    // Read IVF lengths from the global buffer.
    let ivf_idx: u32 = reader
        .metadata()
        .file_schema
        .metadata
        .get(IVF_METADATA_KEY)
        .ok_or_else(|| Error::Index {
            message: "IVF meta missing".to_string(),
            location: location!(),
        })?
        .parse()
        .map_err(|_| Error::Index {
            message: "IVF index parse error".to_string(),
            location: location!(),
        })?;
    let bytes = reader.read_global_buffer(ivf_idx).await?;
    let pb_ivf: pb::Ivf = prost::Message::decode(bytes)?;
    let lengths = pb_ivf.lengths;

    let mut min_fragment_id: Option<u32> = None;
    let mut min_row_id: Option<u64> = None;

    let mut offset: usize = 0;
    for len in &lengths {
        let part_len = *len as usize;
        if part_len > 0 {
            let mut stream = reader.read_stream(
                lance_io::ReadBatchParams::Range(offset..offset + 1),
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )?;
            if let Some(batch_res) = stream.next().await {
                let batch = batch_res?;
                if batch.num_rows() > 0 {
                    let arr = batch
                        .column(row_id_idx)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| Error::Index {
                            message: "ROW_ID_FIELD must be a UInt64 column in auxiliary shard"
                                .to_string(),
                            location: location!(),
                        })?;
                    let row_id_val = arr.value(0);
                    let frag_id = decode_fragment_id_from_row_id(row_id_val);
                    min_fragment_id = Some(match min_fragment_id {
                        Some(cur) => cur.min(frag_id),
                        None => frag_id,
                    });
                    min_row_id = Some(match min_row_id {
                        Some(cur) => cur.min(row_id_val),
                        None => row_id_val,
                    });
                }
            }
        }
        offset += part_len;
    }

    let min_fragment_id = min_fragment_id.unwrap_or(RowAddress::TOMBSTONE_FRAG);
    let min_row_id = min_row_id.unwrap_or(RowAddress::TOMBSTONE_ROW);

    let parent_name = {
        let parts: Vec<_> = aux_path.parts().collect();
        if parts.len() >= 2 {
            parts[parts.len() - 2].as_ref().to_string()
        } else {
            String::new()
        }
    };

    Ok((min_fragment_id, min_row_id, parent_name))
}

#[derive(Debug)]
struct ShardInfo {
    reader: Arc<V2Reader>,
    lengths: Vec<u32>,
    partition_offsets: Vec<usize>,
    total_rows: usize,
    sort_key: (u32, u64, String),
}

#[derive(Debug)]
struct ShardWindowReadJob {
    reader: Arc<V2Reader>,
    window_lengths: Vec<u32>,
    window_total_rows: usize,
    start_offset: usize,
    end_offset: usize,
}

#[derive(Debug)]
struct PartitionWindowBatches {
    window_start: usize,
    per_partition_batches: Vec<Vec<RecordBatch>>,
}

type PartitionWindowFuture = Pin<Box<dyn Future<Output = Result<PartitionWindowBatches>> + Send>>;

struct ShardMergeReader {
    shard_infos: Arc<Vec<ShardInfo>>,
    nlist: usize,
    partition_window_size: usize,
    prefetch_window_count: usize,
    next_window_start: usize,
    in_flight_windows: futures::stream::FuturesOrdered<PartitionWindowFuture>,
    current_window: Option<PartitionWindowBatches>,
    current_partition_offset: usize,
}

impl ShardMergeReader {
    fn new(
        shard_infos: Vec<ShardInfo>,
        nlist: usize,
        partition_window_size: usize,
        prefetch_window_count: usize,
    ) -> Self {
        let mut this = Self {
            shard_infos: Arc::new(shard_infos),
            nlist,
            partition_window_size: partition_window_size.max(1),
            prefetch_window_count: prefetch_window_count.max(1),
            next_window_start: 0,
            in_flight_windows: futures::stream::FuturesOrdered::new(),
            current_window: None,
            current_partition_offset: 0,
        };
        this.fill_prefetch();
        this
    }

    fn fill_prefetch(&mut self) {
        while self.in_flight_windows.len() < self.prefetch_window_count
            && self.next_window_start < self.nlist
        {
            let window_start = self.next_window_start;
            let window_end = std::cmp::min(window_start + self.partition_window_size, self.nlist);
            self.next_window_start = window_end;

            let shard_infos = Arc::clone(&self.shard_infos);
            let nlist = self.nlist;
            let fut: PartitionWindowFuture = Box::pin(async move {
                read_partition_window(shard_infos, nlist, window_start, window_end).await
            });
            self.in_flight_windows.push_back(fut);
        }
    }

    async fn next_partition(&mut self) -> Result<Option<(usize, Vec<RecordBatch>)>> {
        loop {
            if let Some(window) = self.current_window.as_mut() {
                if self.current_partition_offset < window.per_partition_batches.len() {
                    let partition_id = window.window_start + self.current_partition_offset;
                    let batches = std::mem::take(
                        &mut window.per_partition_batches[self.current_partition_offset],
                    );
                    self.current_partition_offset += 1;
                    if self.current_partition_offset == window.per_partition_batches.len() {
                        self.current_window = None;
                        self.current_partition_offset = 0;
                    }
                    self.fill_prefetch();
                    return Ok(Some((partition_id, batches)));
                }
                self.current_window = None;
                self.current_partition_offset = 0;
                continue;
            }

            self.fill_prefetch();
            match self.in_flight_windows.next().await {
                Some(window) => {
                    self.current_window = Some(window?);
                    self.current_partition_offset = 0;
                }
                None => return Ok(None),
            }
        }
    }
}

async fn read_partition_window(
    shard_infos: Arc<Vec<ShardInfo>>,
    nlist: usize,
    window_start: usize,
    window_end: usize,
) -> Result<PartitionWindowBatches> {
    let window_len = window_end - window_start;

    let shard_jobs: Vec<ShardWindowReadJob> = shard_infos
        .iter()
        .map(|shard| {
            let window_lengths = shard.lengths[window_start..window_end].to_vec();
            let window_total_rows = window_lengths.iter().map(|len| *len as usize).sum();
            let start_offset = shard.partition_offsets[window_start];
            let end_offset = if window_end < nlist {
                shard.partition_offsets[window_end]
            } else {
                shard.total_rows
            };

            ShardWindowReadJob {
                reader: Arc::clone(&shard.reader),
                window_lengths,
                window_total_rows,
                start_offset,
                end_offset,
            }
        })
        .collect();

    let shard_parallelism = shard_jobs.len().max(1);
    let mut shard_results_stream = futures::stream::iter(shard_jobs.into_iter().enumerate().map(
        |(shard_idx, shard_job)| async move {
            let per_partition_batches =
                read_shard_window_partitions(shard_job, window_start, window_end, window_len)
                    .await?;
            Ok::<(usize, Vec<Vec<RecordBatch>>), Error>((shard_idx, per_partition_batches))
        },
    ))
    .buffer_unordered(shard_parallelism);

    let mut shard_results: Vec<(usize, Vec<Vec<RecordBatch>>)> =
        Vec::with_capacity(shard_parallelism);
    while let Some(shard_result) = shard_results_stream.next().await {
        shard_results.push(shard_result?);
    }
    shard_results.sort_by_key(|(shard_idx, _)| *shard_idx);

    let mut per_partition_batches: Vec<Vec<RecordBatch>> = vec![Vec::new(); window_len];
    for (_, mut shard_partition_batches) in shard_results {
        for rel_partition in 0..window_len {
            per_partition_batches[rel_partition]
                .append(&mut shard_partition_batches[rel_partition]);
        }
    }

    Ok(PartitionWindowBatches {
        window_start,
        per_partition_batches,
    })
}

async fn read_shard_window_partitions(
    shard_job: ShardWindowReadJob,
    window_start: usize,
    window_end: usize,
    window_len: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    let mut per_partition_batches: Vec<Vec<RecordBatch>> = vec![Vec::new(); window_len];
    if shard_job.window_total_rows == 0 {
        return Ok(per_partition_batches);
    }

    let mut stream = shard_job.reader.read_stream(
        lance_io::ReadBatchParams::Range(shard_job.start_offset..shard_job.end_offset),
        u32::MAX,
        4,
        lance_encoding::decoder::FilterExpression::no_filter(),
    )?;

    let mut rel_partition = 0usize;
    while rel_partition < window_len && shard_job.window_lengths[rel_partition] == 0 {
        rel_partition += 1;
    }
    let mut remaining = if rel_partition < window_len {
        shard_job.window_lengths[rel_partition] as usize
    } else {
        0
    };

    while let Some(rb) = stream.next().await {
        let rb = rb?;
        let mut consumed = 0usize;

        while consumed < rb.num_rows() {
            while rel_partition < window_len && remaining == 0 {
                rel_partition += 1;
                if rel_partition < window_len {
                    remaining = shard_job.window_lengths[rel_partition] as usize;
                }
            }

            if rel_partition >= window_len {
                return Err(Error::Index {
                    message: format!(
                        "Shard has more rows than declared lengths in partition window [{}, {})",
                        window_start, window_end
                    ),
                    location: location!(),
                });
            }

            let to_take = std::cmp::min(remaining, rb.num_rows() - consumed);
            per_partition_batches[rel_partition].push(rb.slice(consumed, to_take));
            consumed += to_take;
            remaining -= to_take;
        }
    }

    while rel_partition < window_len && remaining == 0 {
        rel_partition += 1;
        if rel_partition < window_len {
            remaining = shard_job.window_lengths[rel_partition] as usize;
        }
    }

    if rel_partition != window_len {
        return Err(Error::Index {
            message: format!(
                "Shard has fewer rows than declared lengths in partition window [{}, {})",
                window_start, window_end
            ),
            location: location!(),
        });
    }

    Ok(per_partition_batches)
}

/// Merge all partial_* vector index auxiliary files under `index_dir/{uuid}/partial_*/auxiliary.idx`
/// into `index_dir/{uuid}/auxiliary.idx`.
///
/// Supports IVF_FLAT, IVF_PQ, IVF_SQ, IVF_HNSW_FLAT, IVF_HNSW_PQ, IVF_HNSW_SQ storage types.
/// For PQ and SQ, this assumes all partial indices share the same quantizer/codebook
/// and distance type; it will reuse the first encountered metadata.
pub async fn merge_partial_vector_auxiliary_files(
    object_store: &lance_io::object_store::ObjectStore,
    index_dir: &object_store::path::Path,
) -> Result<()> {
    let mut aux_paths: Vec<object_store::path::Path> = Vec::new();
    let mut stream = object_store.list(Some(index_dir.clone()));
    while let Some(item) = stream.next().await {
        if let Ok(meta) = item {
            if let Some(fname) = meta.location.filename() {
                if fname == INDEX_AUXILIARY_FILE_NAME {
                    // Check parent dir name starts with partial_
                    let parts: Vec<_> = meta.location.parts().collect();
                    if parts.len() >= 2 {
                        let pname = parts[parts.len() - 2].as_ref();
                        if pname.starts_with("partial_") {
                            aux_paths.push(meta.location.clone());
                        }
                    }
                }
            }
        }
    }

    if aux_paths.is_empty() {
        // If a unified auxiliary file already exists at the root, no merge is required.
        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        if object_store.exists(&aux_out).await.unwrap_or(false) {
            log::warn!(
                "No partial_* auxiliary files found under index dir: {}, but unified auxiliary file already exists; skipping merge",
                index_dir
            );
            return Ok(());
        }
        // For certain index types (e.g., FLAT/HNSW-only) the merge may be a no-op in distributed setups
        // where shards were committed directly. In such cases, proceed without error to avoid blocking
        // index manifest merge. PQ/SQ variants still require merging artifacts and will be handled by
        // downstream open logic if missing.
        log::warn!(
            "No partial_* auxiliary files found under index dir: {}; proceeding without merge for index types that do not require auxiliary shards",
            index_dir
        );
        return Ok(());
    }

    // Prepare IVF model and storage metadata aggregation
    let mut distance_type: Option<DistanceType> = None;
    let mut pq_meta: Option<ProductQuantizationMetadata> = None;
    let mut sq_meta: Option<ScalarQuantizationMetadata> = None;
    let mut dim: Option<usize> = None;
    let mut detected_index_type: Option<SupportedIvfIndexType> = None;

    // Prepare output path; we'll create writer once when we know schema
    let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);

    // We'll delay creating the V2 writer until we know the vector schema (dim and quantizer type)
    let mut v2w_opt: Option<V2Writer> = None;

    // We'll also need a scheduler to open readers efficiently
    let sched = ScanScheduler::new(
        Arc::new(object_store.clone()),
        SchedulerConfig::max_bandwidth(object_store),
    );

    // Compute content-derived sort keys for each shard once while opening the
    // auxiliary readers. These keys will be reused both for ordering the
    // enumeration of shards and for per-partition writes.
    let mut shard_keys: Vec<(object_store::path::Path, (u32, u64, String))> =
        Vec::with_capacity(aux_paths.len());
    for aux in aux_paths.into_iter() {
        let key = compute_shard_content_key(&sched, object_store, &aux).await?;
        shard_keys.push((aux, key));
    }

    // Sort shards by their content-derived keys (min_fragment_id, min_row_id,
    // parent_dir_name) to detach from underlying listing order.
    shard_keys.sort_by(|a, b| a.1.cmp(&b.1));

    // Track IVF partition count consistency and accumulate lengths per partition
    let mut nlist_opt: Option<usize> = None;
    let mut accumulated_lengths: Vec<u32> = Vec::new();
    let mut first_centroids: Option<FixedSizeListArray> = None;

    // Track per-shard readers, IVF lengths, and precomputed partition offsets.
    // This avoids reopening each shard file for every partition during merge.
    let mut shard_infos: Vec<ShardInfo> = Vec::new();

    // Iterate over each shard auxiliary file and merge its metadata and collect lengths
    for (aux, key) in &shard_keys {
        let fh = sched.open_file(aux, &CachedFileSize::unknown()).await?;
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await?;
        let meta = reader.metadata();

        // Read distance type
        let dt = meta
            .file_schema
            .metadata
            .get(DISTANCE_TYPE_KEY)
            .ok_or_else(|| Error::Index {
                message: format!("Missing {} in shard", DISTANCE_TYPE_KEY),
                location: location!(),
            })?;
        let dt: DistanceType = DistanceType::try_from(dt.as_str())?;
        if distance_type.is_none() {
            distance_type = Some(dt);
        } else if distance_type.as_ref().map(|v| *v != dt).unwrap_or(false) {
            return Err(Error::Index {
                message: "Distance type mismatch across shards".to_string(),
                location: location!(),
            });
        }

        // Detect index type (first iteration only)
        if detected_index_type.is_none() {
            // Try to derive precise type from sibling partial index.idx metadata if available
            // Try resolve sibling index.idx path by trimming the last component of aux path
            let parent_str = {
                let s = aux.as_ref();
                if let Some((p, _)) = s.trim_end_matches('/').rsplit_once('/') {
                    p.to_string()
                } else {
                    s.to_string()
                }
            };
            let idx_path = object_store::path::Path::from(format!(
                "{}/{}",
                parent_str,
                crate::INDEX_FILE_NAME
            ));
            if object_store.exists(&idx_path).await.unwrap_or(false) {
                let fh2 = sched
                    .open_file(&idx_path, &CachedFileSize::unknown())
                    .await?;
                let idx_reader = V2Reader::try_open(
                    fh2,
                    None,
                    Arc::default(),
                    &lance_core::cache::LanceCache::no_cache(),
                    V2ReaderOptions::default(),
                )
                .await?;
                if let Some(idx_meta_json) = idx_reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(INDEX_METADATA_SCHEMA_KEY)
                {
                    let idx_meta: IndexMetaSchema = serde_json::from_str(idx_meta_json)?;
                    detected_index_type = Some(match idx_meta.index_type.as_str() {
                        "IVF_FLAT" => SupportedIvfIndexType::IvfFlat,
                        "IVF_PQ" => SupportedIvfIndexType::IvfPq,
                        "IVF_SQ" => SupportedIvfIndexType::IvfSq,
                        "IVF_HNSW_FLAT" => SupportedIvfIndexType::IvfHnswFlat,
                        "IVF_HNSW_PQ" => SupportedIvfIndexType::IvfHnswPq,
                        "IVF_HNSW_SQ" => SupportedIvfIndexType::IvfHnswSq,
                        other => {
                            return Err(Error::Index {
                                message: format!(
                                    "Unsupported index type in shard index.idx: {}",
                                    other
                                ),
                                location: location!(),
                            });
                        }
                    });
                }
            }
            // Fallback: infer from auxiliary schema
            if detected_index_type.is_none() {
                let schema_arrow: ArrowSchema = reader.schema().as_ref().into();
                detected_index_type = Some(detect_supported_index_type(&reader, &schema_arrow)?);
            }
        }

        // Read IVF lengths from global buffer
        let ivf_idx: u32 = reader
            .metadata()
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or_else(|| Error::Index {
                message: "IVF meta missing".to_string(),
                location: location!(),
            })?
            .parse()
            .map_err(|_| Error::Index {
                message: "IVF index parse error".to_string(),
                location: location!(),
            })?;
        let bytes = reader.read_global_buffer(ivf_idx).await?;
        let pb_ivf: pb::Ivf = prost::Message::decode(bytes)?;
        let lengths = pb_ivf.lengths.clone();
        let nlist = lengths.len();

        if nlist_opt.is_none() {
            nlist_opt = Some(nlist);
            accumulated_lengths = vec![0; nlist];
            // Try load centroids tensor if present
            if let Some(tensor) = pb_ivf.centroids_tensor.as_ref() {
                let arr = FixedSizeListArray::try_from(tensor)?;
                first_centroids = Some(arr.clone());
                let d0 = arr.value_length() as usize;
                if dim.is_none() {
                    dim = Some(d0);
                }
            }
        } else if nlist_opt.as_ref().map(|v| *v != nlist).unwrap_or(false) {
            return Err(Error::Index {
                message: "IVF partition count mismatch across shards".to_string(),
                location: location!(),
            });
        }

        // Handle logic based on detected index type
        let idx_type = detected_index_type.ok_or_else(|| Error::Index {
            message: "Unable to detect index type".to_string(),
            location: location!(),
        })?;
        match idx_type {
            SupportedIvfIndexType::IvfSq => {
                // Handle Scalar Quantization (SQ) storage for IVF_SQ
                let sq_json = if let Some(sq_json) =
                    reader.metadata().file_schema.metadata.get(SQ_METADATA_KEY)
                {
                    sq_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    // Try to extract SQ metadata from storage metadata
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        // Check if this is SQ metadata by trying to parse it
                        if let Ok(_sq_meta) =
                            serde_json::from_str::<ScalarQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "SQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "SQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "SQ metadata missing".to_string(),
                        location: location!(),
                    });
                };

                let sq_meta_parsed: ScalarQuantizationMetadata = serde_json::from_str(&sq_json)
                    .map_err(|e| Error::Index {
                        message: format!("SQ metadata parse error: {}", e),
                        location: location!(),
                    })?;

                let d0 = sq_meta_parsed.dim;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }

                if sq_meta.is_none() {
                    sq_meta = Some(sq_meta_parsed.clone());
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_sq(object_store, &aux_out, dt, &sq_meta_parsed).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIvfIndexType::IvfPq => {
                // Handle Product Quantization (PQ) storage
                // Load PQ metadata JSON; construct ProductQuantizationMetadata
                let pm_json = if let Some(pm_json) =
                    reader.metadata().file_schema.metadata.get(PQ_METADATA_KEY)
                {
                    pm_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    // Try to extract PQ metadata from storage metadata
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        // Check if this is PQ metadata by trying to parse it
                        if let Ok(_pq_meta) =
                            serde_json::from_str::<ProductQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "PQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "PQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "PQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let mut pm: ProductQuantizationMetadata =
                    serde_json::from_str(&pm_json).map_err(|e| Error::Index {
                        message: format!("PQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                // Load codebook from global buffer if not present
                if pm.codebook.is_none() {
                    let tensor_bytes = reader
                        .read_global_buffer(pm.codebook_position as u32)
                        .await?;
                    let codebook_tensor: crate::pb::Tensor = prost::Message::decode(tensor_bytes)?;
                    pm.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
                }
                let d0 = pm.dimension;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if let Some(existing_pm) = pq_meta.as_ref() {
                    // Enforce structural equality
                    if existing_pm.num_sub_vectors != pm.num_sub_vectors
                        || existing_pm.nbits != pm.nbits
                        || existing_pm.dimension != pm.dimension
                    {
                        return Err(Error::Index {
                            message: format!(
                                "Distributed PQ merge: structural mismatch across shards; first(dim={}, m={}, nbits={}), current(dim={}, m={}, nbits={})",
                                existing_pm.dimension,
                                existing_pm.num_sub_vectors,
                                existing_pm.nbits,
                                pm.dimension,
                                pm.num_sub_vectors,
                                pm.nbits
                            ),
                            location: location!(),
                        });
                    }
                    // Enforce codebook equality with tolerance for minor serialization diffs
                    let existing_cb =
                        existing_pm.codebook.as_ref().ok_or_else(|| Error::Index {
                            message: "PQ codebook missing in first shard".to_string(),
                            location: location!(),
                        })?;
                    let current_cb = pm.codebook.as_ref().ok_or_else(|| Error::Index {
                        message: "PQ codebook missing in shard".to_string(),
                        location: location!(),
                    })?;
                    if !fixed_size_list_equal(existing_cb, current_cb) {
                        const TOL: f32 = 1e-5;
                        if !fixed_size_list_almost_equal(existing_cb, current_cb, TOL) {
                            return Err(Error::Index {
                                message: "PQ codebook content mismatch across shards".to_string(),
                                location: location!(),
                            });
                        } else {
                            log::warn!("PQ codebook differs within tolerance; proceeding with first shard codebook");
                        }
                    }
                }
                if pq_meta.is_none() {
                    pq_meta = Some(pm.clone());
                }
                if v2w_opt.is_none() {
                    let mut pm_for_unified = pm.clone();
                    pm_for_unified.transposed = true;
                    let w = init_writer_for_pq(object_store, &aux_out, dt, &pm_for_unified).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIvfIndexType::IvfFlat => {
                // Handle FLAT storage
                // FLAT: infer dimension from vector column using first shard's schema
                let schema: ArrowSchema = reader.schema().as_ref().into();
                let flat_field = schema
                    .fields
                    .iter()
                    .find(|f| f.name() == crate::vector::flat::storage::FLAT_COLUMN)
                    .ok_or_else(|| Error::Index {
                        message: "FLAT column missing".to_string(),
                        location: location!(),
                    })?;
                let d0 = match flat_field.data_type() {
                    DataType::FixedSizeList(_, sz) => *sz as usize,
                    _ => 0,
                };
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_flat(object_store, &aux_out, d0, dt).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIvfIndexType::IvfHnswFlat => {
                // Treat HNSW_FLAT storage the same as FLAT: create schema with ROW_ID + flat vectors
                // Determine dimension from shard schema (flat column) or fallback to STORAGE_METADATA_KEY
                let schema_arrow: ArrowSchema = reader.schema().as_ref().into();
                // Try to find flat column and derive dim
                let d0 = if let Some(flat_field) = schema_arrow
                    .fields
                    .iter()
                    .find(|f| f.name() == crate::vector::flat::storage::FLAT_COLUMN)
                {
                    match flat_field.data_type() {
                        DataType::FixedSizeList(_, sz) => *sz as usize,
                        _ => 0,
                    }
                } else {
                    // Fallback to STORAGE_METADATA_KEY FlatMetadata
                    if let Some(storage_meta_json) = reader
                        .metadata()
                        .file_schema
                        .metadata
                        .get(STORAGE_METADATA_KEY)
                    {
                        let storage_metadata_vec: Vec<String> =
                            serde_json::from_str(storage_meta_json).map_err(|e| Error::Index {
                                message: format!("Failed to parse storage metadata: {}", e),
                                location: location!(),
                            })?;
                        if let Some(first_meta) = storage_metadata_vec.first() {
                            if let Ok(flat_meta) = serde_json::from_str::<FlatMetadata>(first_meta)
                            {
                                flat_meta.dim
                            } else {
                                return Err(Error::Index {
                                    message: "FLAT metadata missing in storage metadata"
                                        .to_string(),
                                    location: location!(),
                                });
                            }
                        } else {
                            return Err(Error::Index {
                                message: "FLAT metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "FLAT column missing and no storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                };
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_flat(object_store, &aux_out, d0, dt).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIvfIndexType::IvfHnswPq => {
                // Treat HNSW_PQ storage the same as PQ: reuse PQ metadata and schema creation
                let pm_json = if let Some(pm_json) =
                    reader.metadata().file_schema.metadata.get(PQ_METADATA_KEY)
                {
                    pm_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        if let Ok(_pq_meta) =
                            serde_json::from_str::<ProductQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "PQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "PQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "PQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let mut pm: ProductQuantizationMetadata =
                    serde_json::from_str(&pm_json).map_err(|e| Error::Index {
                        message: format!("PQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                if pm.codebook.is_none() {
                    let tensor_bytes = reader
                        .read_global_buffer(pm.codebook_position as u32)
                        .await?;
                    let codebook_tensor: crate::pb::Tensor = prost::Message::decode(tensor_bytes)?;
                    pm.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
                }
                let d0 = pm.dimension;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if let Some(existing_pm) = pq_meta.as_ref() {
                    // Enforce structural equality
                    if existing_pm.num_sub_vectors != pm.num_sub_vectors
                        || existing_pm.nbits != pm.nbits
                        || existing_pm.dimension != pm.dimension
                    {
                        return Err(Error::Index {
                            message: format!(
                                "Distributed PQ merge (HNSW_PQ): structural mismatch across shards; first(dim={}, m={}, nbits={}), current(dim={}, m={}, nbits={})",
                                existing_pm.dimension,
                                existing_pm.num_sub_vectors,
                                existing_pm.nbits,
                                pm.dimension,
                                pm.num_sub_vectors,
                                pm.nbits
                            ),
                            location: location!(),
                        });
                    }
                    // Enforce codebook equality with tolerance for minor serialization diffs
                    let existing_cb =
                        existing_pm.codebook.as_ref().ok_or_else(|| Error::Index {
                            message: "PQ codebook missing in first shard".to_string(),
                            location: location!(),
                        })?;
                    let current_cb = pm.codebook.as_ref().ok_or_else(|| Error::Index {
                        message: "PQ codebook missing in shard".to_string(),
                        location: location!(),
                    })?;
                    if !fixed_size_list_equal(existing_cb, current_cb) {
                        const TOL: f32 = 1e-5;
                        if !fixed_size_list_almost_equal(existing_cb, current_cb, TOL) {
                            return Err(Error::Index {
                                message: "PQ codebook content mismatch across shards".to_string(),
                                location: location!(),
                            });
                        } else {
                            log::warn!("PQ codebook differs within tolerance; proceeding with first shard codebook");
                        }
                    }
                }
                if pq_meta.is_none() {
                    pq_meta = Some(pm.clone());
                }
                if v2w_opt.is_none() {
                    let mut pm_for_unified = pm.clone();
                    pm_for_unified.transposed = true;
                    let w = init_writer_for_pq(object_store, &aux_out, dt, &pm_for_unified).await?;
                    v2w_opt = Some(w);
                }
            }
            SupportedIvfIndexType::IvfHnswSq => {
                // Treat HNSW_SQ storage the same as SQ: reuse SQ metadata and schema creation
                let sq_json = if let Some(sq_json) =
                    reader.metadata().file_schema.metadata.get(SQ_METADATA_KEY)
                {
                    sq_json.clone()
                } else if let Some(storage_meta_json) = reader
                    .metadata()
                    .file_schema
                    .metadata
                    .get(STORAGE_METADATA_KEY)
                {
                    let storage_metadata_vec: Vec<String> = serde_json::from_str(storage_meta_json)
                        .map_err(|e| Error::Index {
                            message: format!("Failed to parse storage metadata: {}", e),
                            location: location!(),
                        })?;
                    if let Some(first_meta) = storage_metadata_vec.first() {
                        if let Ok(_sq_meta) =
                            serde_json::from_str::<ScalarQuantizationMetadata>(first_meta)
                        {
                            first_meta.clone()
                        } else {
                            return Err(Error::Index {
                                message: "SQ metadata missing in storage metadata".to_string(),
                                location: location!(),
                            });
                        }
                    } else {
                        return Err(Error::Index {
                            message: "SQ metadata missing in storage metadata".to_string(),
                            location: location!(),
                        });
                    }
                } else {
                    return Err(Error::Index {
                        message: "SQ metadata missing".to_string(),
                        location: location!(),
                    });
                };
                let sq_meta_parsed: ScalarQuantizationMetadata = serde_json::from_str(&sq_json)
                    .map_err(|e| Error::Index {
                        message: format!("SQ metadata parse error: {}", e),
                        location: location!(),
                    })?;
                let d0 = sq_meta_parsed.dim;
                dim.get_or_insert(d0);
                if let Some(dprev) = dim {
                    if dprev != d0 {
                        return Err(Error::Index {
                            message: "Dimension mismatch across shards".to_string(),
                            location: location!(),
                        });
                    }
                }
                if sq_meta.is_none() {
                    sq_meta = Some(sq_meta_parsed.clone());
                }
                if v2w_opt.is_none() {
                    let w = init_writer_for_sq(object_store, &aux_out, dt, &sq_meta_parsed).await?;
                    v2w_opt = Some(w);
                }
            }
        }

        let mut partition_offsets = Vec::with_capacity(nlist);
        let mut running_offset = 0usize;
        for len in &lengths {
            partition_offsets.push(running_offset);
            running_offset = running_offset.saturating_add(*len as usize);
        }

        // Accumulate overall lengths per partition for unified IVF model.
        for pid in 0..nlist {
            let part_len = lengths[pid];
            accumulated_lengths[pid] = accumulated_lengths[pid].saturating_add(part_len);
        }

        // Keep one opened reader per shard and reuse it during partition merge.
        shard_infos.push(ShardInfo {
            reader: Arc::new(reader),
            lengths,
            partition_offsets,
            total_rows: running_offset,
            sort_key: key.clone(),
        });
    }

    // Re-sort shard_infos using content-derived keys to decouple per-partition
    // write ordering from discovery order.
    shard_infos.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));

    // Write rows grouped by partition across all shards to ensure contiguous ranges per partition

    if v2w_opt.is_none() {
        return Err(Error::Index {
            message: "Failed to initialize unified writer".to_string(),
            location: location!(),
        });
    }
    let nlist = nlist_opt.ok_or_else(|| Error::Index {
        message: "Missing IVF partition count".to_string(),
        location: location!(),
    })?;
    let idx_type_final = detected_index_type.ok_or_else(|| Error::Index {
        message: "Unable to detect index type".to_string(),
        location: location!(),
    })?;

    match idx_type_final {
        SupportedIvfIndexType::IvfPq | SupportedIvfIndexType::IvfHnswPq => {
            // For PQ-backed indices, transpose PQ codes while merging partitions
            // so that the unified file stores column-major PQ codes.
            let partition_window_size = *PARTITION_WINDOW_SIZE;
            let prefetch_window_count = *PARTITION_PREFETCH_WINDOW_COUNT;
            let mut shard_merge_reader = ShardMergeReader::new(
                shard_infos,
                nlist,
                partition_window_size,
                prefetch_window_count,
            );

            while let Some((pid, batches)) = shard_merge_reader.next_partition().await? {
                if accumulated_lengths[pid] == 0 {
                    continue;
                }
                if batches.is_empty() {
                    return Err(Error::Index {
                        message: format!("No merged batches found for non-empty partition {}", pid),
                        location: location!(),
                    });
                }

                let schema = batches[0].schema();
                let partition_batch = concat_batches(&schema, batches.iter())?;
                if let Some(w) = v2w_opt.as_mut() {
                    write_partition_rows_pq_transposed(w, partition_batch).await?;
                }
            }
        }
        _ => {
            for pid in 0..nlist {
                for shard in shard_infos.iter() {
                    let part_len = shard.lengths[pid] as usize;
                    if part_len == 0 {
                        continue;
                    }
                    let offset = shard.partition_offsets[pid];
                    if let Some(w) = v2w_opt.as_mut() {
                        write_partition_rows(shard.reader.as_ref(), w, offset..offset + part_len)
                            .await?;
                    }
                }
            }
        }
    }

    // Write unified IVF metadata into global buffer & set schema metadata
    if let Some(w) = v2w_opt.as_mut() {
        let mut ivf_model = if let Some(c) = first_centroids {
            IvfStorageModel::new(c, None)
        } else {
            IvfStorageModel::empty()
        };
        for len in accumulated_lengths.iter() {
            ivf_model.add_partition(*len);
        }
        let dt2 = distance_type.ok_or_else(|| Error::Index {
            message: "Distance type missing".to_string(),
            location: location!(),
        })?;
        write_unified_ivf_and_index_metadata(w, &ivf_model, dt2, idx_type_final).await?;
        w.finish().await?;
    } else {
        return Err(Error::Index {
            message: "Failed to initialize unified writer".to_string(),
            location: location!(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array};
    use arrow_schema::Field;
    use bytes::Bytes;
    use futures::StreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::address::RowAddress;
    use lance_core::ROW_ID_FIELD;
    use lance_file::writer::FileWriterOptions as V2WriterOptions;
    use lance_io::object_store::ObjectStore;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use lance_io::utils::CachedFileSize;
    use lance_linalg::distance::DistanceType;
    use object_store::path::Path;
    use prost::Message;

    async fn write_flat_partial_aux(
        store: &ObjectStore,
        aux_path: &Path,
        dim: i32,
        lengths: &[u32],
        base_row_id: u64,
        distance_type: DistanceType,
    ) -> Result<usize> {
        let arrow_schema = ArrowSchema::new(vec![
            (*ROW_ID_FIELD).clone(),
            Field::new(
                crate::vector::flat::storage::FLAT_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]);

        let writer = store.create(aux_path).await?;
        let mut v2w = V2Writer::try_new(
            writer,
            lance_core::datatypes::Schema::try_from(&arrow_schema)?,
            V2WriterOptions::default(),
        )?;

        // Distance type metadata for this shard.
        v2w.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());

        // IVF metadata: only lengths are needed by the merger.
        let ivf_meta = pb::Ivf {
            centroids: Vec::new(),
            offsets: Vec::new(),
            lengths: lengths.to_vec(),
            centroids_tensor: None,
            loss: None,
        };
        let buf = Bytes::from(ivf_meta.encode_to_vec());
        let pos = v2w.add_global_buffer(buf).await?;
        v2w.add_schema_metadata(IVF_METADATA_KEY, pos.to_string());

        // Build row ids and vectors grouped by partition so that ranges match lengths.
        let total_rows: usize = lengths.iter().map(|v| *v as usize).sum();
        let mut row_ids = Vec::with_capacity(total_rows);
        let mut values = Vec::with_capacity(total_rows * dim as usize);

        let mut current_row_id = base_row_id;
        for (pid, len) in lengths.iter().enumerate() {
            for _ in 0..*len {
                row_ids.push(current_row_id);
                current_row_id += 1;
                for d in 0..dim {
                    // Simple deterministic payload; only layout matters for merge.
                    values.push(pid as f32 + d as f32 * 0.01);
                }
            }
        }

        let row_id_arr = UInt64Array::from(row_ids);
        let value_arr = Float32Array::from(values);
        let fsl = FixedSizeListArray::try_new_from_values(value_arr, dim).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(row_id_arr), Arc::new(fsl)],
        )
        .unwrap();

        v2w.write_batch(&batch).await?;
        v2w.finish().await?;
        Ok(total_rows)
    }

    #[tokio::test]
    async fn test_merge_ivf_flat_success_basic() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];
        let dim = 2_i32;

        write_flat_partial_aux(&object_store, &aux0, dim, &lengths0, 0, DistanceType::L2)
            .await
            .unwrap();
        write_flat_partial_aux(&object_store, &aux1, dim, &lengths1, 100, DistanceType::L2)
            .await
            .unwrap();

        merge_partial_vector_auxiliary_files(&object_store, &index_dir)
            .await
            .unwrap();

        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        assert!(object_store.exists(&aux_out).await.unwrap());

        // Use ScanScheduler to obtain a FileScheduler (required by V2Reader::try_open)
        let sched = ScanScheduler::new(
            Arc::new(object_store.clone()),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let fh = sched
            .open_file(&aux_out, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await
        .unwrap();
        let meta = reader.metadata();

        // Validate IVF lengths aggregation.
        let ivf_idx: u32 = meta
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .unwrap()
            .parse()
            .unwrap();
        let bytes = reader.read_global_buffer(ivf_idx).await.unwrap();
        let pb_ivf: pb::Ivf = prost::Message::decode(bytes).unwrap();
        let expected_lengths: Vec<u32> = lengths0
            .iter()
            .zip(lengths1.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        assert_eq!(pb_ivf.lengths, expected_lengths);

        // Validate index metadata schema.
        let idx_meta_json = meta
            .file_schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .unwrap();
        let idx_meta: IndexMetaSchema = serde_json::from_str(idx_meta_json).unwrap();
        assert_eq!(idx_meta.index_type, "IVF_FLAT");
        assert_eq!(idx_meta.distance_type, DistanceType::L2.to_string());

        // Validate total number of rows.
        let mut total_rows = 0usize;
        let mut stream = reader
            .read_stream(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .unwrap();
        while let Some(batch) = stream.next().await {
            total_rows += batch.unwrap().num_rows();
        }
        let expected_total: usize = expected_lengths.iter().map(|v| *v as usize).sum();
        assert_eq!(total_rows, expected_total);
    }

    #[tokio::test]
    async fn test_merge_distance_type_mismatch() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths = vec![2_u32, 2_u32];
        let dim = 2_i32;

        write_flat_partial_aux(&object_store, &aux0, dim, &lengths, 0, DistanceType::L2)
            .await
            .unwrap();
        write_flat_partial_aux(
            &object_store,
            &aux1,
            dim,
            &lengths,
            100,
            DistanceType::Cosine,
        )
        .await
        .unwrap();

        let res = merge_partial_vector_auxiliary_files(&object_store, &index_dir).await;
        match res {
            Err(Error::Index { message, .. }) => {
                assert!(
                    message.contains("Distance type mismatch"),
                    "unexpected message: {}",
                    message
                );
            }
            other => panic!(
                "expected Error::Index for distance type mismatch, got {:?}",
                other
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn write_pq_partial_aux(
        store: &ObjectStore,
        aux_path: &Path,
        nbits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        lengths: &[u32],
        base_row_id: u64,
        distance_type: DistanceType,
        codebook: &FixedSizeListArray,
    ) -> Result<usize> {
        let num_bytes = if nbits == 4 {
            // Two 4-bit codes per byte.
            num_sub_vectors / 2
        } else {
            num_sub_vectors
        };

        let arrow_schema = ArrowSchema::new(vec![
            (*ROW_ID_FIELD).clone(),
            Field::new(
                crate::vector::PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    num_bytes as i32,
                ),
                true,
            ),
        ]);

        let writer = store.create(aux_path).await?;
        let mut v2w = V2Writer::try_new(
            writer,
            lance_core::datatypes::Schema::try_from(&arrow_schema)?,
            V2WriterOptions::default(),
        )?;

        // Distance type metadata for this shard.
        v2w.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());

        // PQ metadata with codebook stored in a global buffer.
        let mut pq_meta = ProductQuantizationMetadata {
            codebook_position: 0,
            nbits,
            num_sub_vectors,
            dimension,
            codebook: Some(codebook.clone()),
            codebook_tensor: Vec::new(),
            transposed: true,
        };

        let codebook_tensor: pb::Tensor = pb::Tensor::try_from(codebook)?;
        let codebook_buf = Bytes::from(codebook_tensor.encode_to_vec());
        let codebook_pos = v2w.add_global_buffer(codebook_buf).await?;
        pq_meta.codebook_position = codebook_pos as usize;

        let pq_meta_json = serde_json::to_string(&pq_meta)?;
        v2w.add_schema_metadata(PQ_METADATA_KEY, pq_meta_json);

        // IVF metadata: only lengths are needed by the merger.
        let ivf_meta = pb::Ivf {
            centroids: Vec::new(),
            offsets: Vec::new(),
            lengths: lengths.to_vec(),
            centroids_tensor: None,
            loss: None,
        };
        let buf = Bytes::from(ivf_meta.encode_to_vec());
        let ivf_pos = v2w.add_global_buffer(buf).await?;
        v2w.add_schema_metadata(IVF_METADATA_KEY, ivf_pos.to_string());

        // Build row ids and PQ codes grouped by partition so that ranges match lengths.
        let total_rows: usize = lengths.iter().map(|v| *v as usize).sum();
        let mut row_ids = Vec::with_capacity(total_rows);
        let mut codes = Vec::with_capacity(total_rows * num_bytes);

        let mut current_row_id = base_row_id;
        for (pid, len) in lengths.iter().enumerate() {
            for _ in 0..*len {
                row_ids.push(current_row_id);
                current_row_id += 1;
                for b in 0..num_bytes {
                    // Simple deterministic payload; merge only cares about layout.
                    codes.push((pid + b) as u8);
                }
            }
        }

        let row_id_arr = UInt64Array::from(row_ids);
        let codes_arr = UInt8Array::from(codes);
        let codes_fsl =
            FixedSizeListArray::try_new_from_values(codes_arr, num_bytes as i32).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(row_id_arr), Arc::new(codes_fsl)],
        )
        .unwrap();

        v2w.write_batch(&batch).await?;
        v2w.finish().await?;
        Ok(total_rows)
    }

    #[tokio::test]
    async fn test_merge_ivf_pq_success() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid_pq");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];

        // PQ parameters.
        let nbits = 4_u32;
        let num_sub_vectors = 2_usize;
        let dimension = 8_usize;

        // Deterministic PQ codebook shared by both shards.
        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors;
        let total_values = num_codebook_vectors * dimension;
        let values = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook = FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap();

        // Non-overlapping row id ranges across shards.
        write_pq_partial_aux(
            &object_store,
            &aux0,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths0,
            0,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        write_pq_partial_aux(
            &object_store,
            &aux1,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths1,
            1_000,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        // Merge PQ auxiliary files.
        merge_partial_vector_auxiliary_files(&object_store, &index_dir)
            .await
            .unwrap();

        // 3) Unified auxiliary file exists.
        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        assert!(object_store.exists(&aux_out).await.unwrap());

        // Open merged auxiliary file.
        let sched = ScanScheduler::new(
            Arc::new(object_store.clone()),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let fh = sched
            .open_file(&aux_out, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await
        .unwrap();
        let meta = reader.metadata();

        // 4) Unified IVF metadata lengths equal shard-wise sums.
        let ivf_idx: u32 = meta
            .file_schema
            .metadata
            .get(IVF_METADATA_KEY)
            .unwrap()
            .parse()
            .unwrap();
        let bytes = reader.read_global_buffer(ivf_idx).await.unwrap();
        let pb_ivf: pb::Ivf = prost::Message::decode(bytes).unwrap();
        let expected_lengths: Vec<u32> = lengths0
            .iter()
            .zip(lengths1.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        assert_eq!(pb_ivf.lengths, expected_lengths);

        // 5) Index metadata schema reports IVF_PQ and correct distance type.
        let idx_meta_json = meta
            .file_schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .unwrap();
        let idx_meta: IndexMetaSchema = serde_json::from_str(idx_meta_json).unwrap();
        assert_eq!(idx_meta.index_type, "IVF_PQ");
        assert_eq!(idx_meta.distance_type, DistanceType::L2.to_string());

        // 6) PQ metadata and codebook are preserved.
        let pq_meta_json = meta.file_schema.metadata.get(PQ_METADATA_KEY).unwrap();
        let pq_meta: ProductQuantizationMetadata = serde_json::from_str(pq_meta_json).unwrap();
        assert_eq!(pq_meta.nbits, nbits);
        assert_eq!(pq_meta.num_sub_vectors, num_sub_vectors);
        assert_eq!(pq_meta.dimension, dimension);

        let codebook_pos = pq_meta.codebook_position as u32;
        let cb_bytes = reader.read_global_buffer(codebook_pos).await.unwrap();
        let cb_tensor: pb::Tensor = prost::Message::decode(cb_bytes).unwrap();
        let merged_codebook = FixedSizeListArray::try_from(&cb_tensor).unwrap();

        assert!(fixed_size_list_equal(&codebook, &merged_codebook));
    }

    #[tokio::test]
    async fn test_merge_ivf_pq_codebook_mismatch() {
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid_pq_mismatch");

        let partial0 = index_dir.child("partial_0");
        let partial1 = index_dir.child("partial_1");
        let aux0 = partial0.child(INDEX_AUXILIARY_FILE_NAME);
        let aux1 = partial1.child(INDEX_AUXILIARY_FILE_NAME);

        let lengths0 = vec![2_u32, 1_u32];
        let lengths1 = vec![1_u32, 2_u32];

        // PQ parameters.
        let nbits = 4_u32;
        let num_sub_vectors = 2_usize;
        let dimension = 8_usize;

        // Base PQ codebook for shard 0.
        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors;
        let total_values = num_codebook_vectors * dimension;
        let values0 = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook0 = FixedSizeListArray::try_new_from_values(values0, dimension as i32).unwrap();

        // Different PQ codebook for shard 1 with values shifted beyond tolerance.
        let values1 = Float32Array::from_iter((0..total_values).map(|v| v as f32 + 1.0));
        let codebook1 = FixedSizeListArray::try_new_from_values(values1, dimension as i32).unwrap();

        // Non-overlapping row id ranges across shards.
        write_pq_partial_aux(
            &object_store,
            &aux0,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths0,
            0,
            DistanceType::L2,
            &codebook0,
        )
        .await
        .unwrap();

        write_pq_partial_aux(
            &object_store,
            &aux1,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths1,
            1_000,
            DistanceType::L2,
            &codebook1,
        )
        .await
        .unwrap();

        let res = merge_partial_vector_auxiliary_files(&object_store, &index_dir).await;
        match res {
            Err(Error::Index { message, .. }) => {
                assert!(
                    message.contains("PQ codebook content mismatch"),
                    "unexpected message: {}",
                    message
                );
            }
            other => panic!(
                "expected Error::Index with PQ codebook content mismatch, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_merge_partial_order_tie_breaker() {
        // Two partial directories that map to the same (min_fragment_id, dataset_version)
        // but differ in their parent directory name. This exercises the third
        // lexicographic tie-breaker component of the sort key.
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/uuid_tie");

        let partial_a = index_dir.child("partial_1_10");
        let partial_b = index_dir.child("partial_1_10b");
        let aux_a = partial_a.child(INDEX_AUXILIARY_FILE_NAME);
        let aux_b = partial_b.child(INDEX_AUXILIARY_FILE_NAME);

        // Equal-length shards to simulate the tie scenario where per-partition
        // row counts alone cannot disambiguate ordering.
        let lengths = vec![2_u32, 2_u32];

        // PQ parameters shared by both shards.
        let nbits = 4_u32;
        let num_sub_vectors = 2_usize;
        let dimension = 8_usize;

        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors;
        let total_values = num_codebook_vectors * dimension;
        let values = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook = FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap();

        // Shard A: base_row_id = 0.
        write_pq_partial_aux(
            &object_store,
            &aux_a,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths,
            0,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        // Shard B: base_row_id = 1_000, identical lengths and PQ metadata.
        write_pq_partial_aux(
            &object_store,
            &aux_b,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths,
            1_000,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        // Merge must succeed and produce a unified auxiliary file.
        merge_partial_vector_auxiliary_files(&object_store, &index_dir)
            .await
            .unwrap();

        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        assert!(object_store.exists(&aux_out).await.unwrap());

        // Open merged auxiliary file and verify that the per-partition write
        // order follows the lexicographic parent-dir tiebreaker: rows from
        // `partial_1_10` (row ids starting at 0) should precede rows from
        // `partial_1_10b` (row ids starting at 1_000) for the first partition.
        let sched = ScanScheduler::new(
            Arc::new(object_store.clone()),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let fh = sched
            .open_file(&aux_out, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await
        .unwrap();

        let mut stream = reader
            .read_stream(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .unwrap();

        let mut row_ids = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for i in 0..arr.len() {
                row_ids.push(arr.value(i));
            }
        }

        // We expect two partitions with aggregated lengths [4, 4].
        assert_eq!(row_ids.len(), 8);
        let first_partition_ids = &row_ids[..4];
        assert_eq!(first_partition_ids, &[0, 1, 1_000, 1_001]);
    }

    #[tokio::test]
    async fn test_merge_content_key_order_invariance() {
        // Two partial directories whose content-derived keys
        // (min_fragment_id, min_row_id) are identical; ordering is determined
        // solely by the parent directory name as a lexicographic tie-breaker.
        let object_store = ObjectStore::memory();
        let index_dir = Path::from("index/content_key");

        let partial_a = index_dir.child("partial_content_a");
        let partial_b = index_dir.child("partial_content_b");
        let aux_a = partial_a.child(INDEX_AUXILIARY_FILE_NAME);
        let aux_b = partial_b.child(INDEX_AUXILIARY_FILE_NAME);

        // Equal-length shards so per-partition lengths alone cannot disambiguate
        // ordering.
        let lengths = vec![2_u32, 2_u32];

        // PQ parameters shared by both shards.
        let nbits = 4_u32;
        let num_sub_vectors = 2_usize;
        let dimension = 8_usize;

        let num_centroids = 1_usize << nbits;
        let num_codebook_vectors = num_centroids * num_sub_vectors;
        let total_values = num_codebook_vectors * dimension;
        let values = Float32Array::from_iter((0..total_values).map(|v| v as f32));
        let codebook = FixedSizeListArray::try_new_from_values(values, dimension as i32).unwrap();

        // Use a RowAddress-encoded base so both shards have the same
        // (fragment_id, row_offset) for their first row, hence identical
        // content-derived numeric keys.
        let base_addr: u64 = RowAddress::new_from_parts(1, 5).into();

        write_pq_partial_aux(
            &object_store,
            &aux_a,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths,
            base_addr,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        write_pq_partial_aux(
            &object_store,
            &aux_b,
            nbits,
            num_sub_vectors,
            dimension,
            &lengths,
            base_addr,
            DistanceType::L2,
            &codebook,
        )
        .await
        .unwrap();

        // Merge must succeed and produce a unified auxiliary file.
        merge_partial_vector_auxiliary_files(&object_store, &index_dir)
            .await
            .unwrap();

        let aux_out = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        assert!(object_store.exists(&aux_out).await.unwrap());

        // Open merged auxiliary file and inspect row id layout.
        let sched = ScanScheduler::new(
            Arc::new(object_store.clone()),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let fh = sched
            .open_file(&aux_out, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            V2ReaderOptions::default(),
        )
        .await
        .unwrap();

        let mut stream = reader
            .read_stream(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                4,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .unwrap();

        let mut row_ids = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for i in 0..arr.len() {
                row_ids.push(arr.value(i));
            }
        }

        // Two shards, each contributing `sum(lengths)` rows.
        let expected_total_rows: usize = lengths.iter().map(|v| *v as usize).sum::<usize>() * 2;
        assert_eq!(row_ids.len(), expected_total_rows);

        let first_partition_rows = lengths[0] as usize * 2;
        let (p0, p1) = row_ids.split_at(first_partition_rows);

        let base = base_addr;
        // For partition 0 we expect rows from `partial_content_a` first, then
        // from `partial_content_b`.
        let expected_p0 = vec![base, base + 1, base, base + 1];
        assert_eq!(p0, expected_p0.as_slice());

        // For partition 1 the pattern continues with offsets +2, +3.
        let expected_p1 = vec![base + 2, base + 3, base + 2, base + 3];
        assert_eq!(p1, expected_p1.as_slice());
    }
}
