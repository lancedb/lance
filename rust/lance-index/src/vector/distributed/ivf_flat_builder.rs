// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    cast::AsArray, Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_file::v2::writer::{FileWriter as V2FileWriter, FileWriterOptions as V2WriterOptions};
use lance_io::object_store::ObjectStore;
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_linalg::distance::DistanceType;
use object_store::path::Path;
use snafu::location;
use tempfile::TempDir;

use crate::vector::flat::index::{FlatMetadata, FlatQuantizer};
use crate::vector::ivf::storage::{IvfModel as IvfStorageModel, IVF_METADATA_KEY};
use crate::vector::ivf::{new_ivf_transformer, IvfBuildParams};
use crate::vector::quantizer::Quantization;
use crate::vector::storage::STORAGE_METADATA_KEY;
use crate::vector::{DISTANCE_TYPE_KEY, PART_ID_COLUMN};
use prost::Message;

use super::communicator::Communicator;

/// Parameters to orchestrate distributed IVF_FLAT building on a single process worker
#[derive(Clone, Debug)]
pub struct DistributedIvfFlatParams {
    pub nlist: usize,
    pub distance_type: DistanceType,
    /// Optional pretrained centroids; if None, caller should have trained and provided
    /// via upstream flow (we keep capability to pass here to minimize churn).
    pub pretrained_centroids: Option<Arc<FixedSizeListArray>>,
    /// Writer options for temporary per-worker output
    pub shard_root: Option<Path>,
}

impl DistributedIvfFlatParams {
    pub fn new(nlist: usize, distance_type: DistanceType) -> Self {
        Self {
            nlist,
            distance_type,
            pretrained_centroids: None,
            shard_root: None,
        }
    }
}

/// Build distributed IVF_FLAT index
///
/// - local_vectors: this worker's shard of vectors (FixedSizeList<Float32, dim>)
/// - global_row_offset: caller-provided row_id base to guarantee uniqueness across workers
///   (e.g. prefix with rank * 1_000_000_000 or accumulate sizes). If None, we use
///   rank-based prefix to avoid collisions in tests.
/// - ivf_params: IvfBuildParams (used for future parity; currently we consume max_iters only.)
/// - dist: Distance metric
/// - comm: communicator
/// - out_dir: unified output directory to write auxiliary.idx (rank0 only); others pass same path
///
/// Return: path to the unified auxiliary file (rank0) or Ok(None) for non-root ranks
pub async fn build_ivf_flat_distributed(
    local_vectors: &FixedSizeListArray,
    global_row_offset: Option<u64>,
    _ivf_params: &IvfBuildParams,
    dist: DistanceType,
    params: &DistributedIvfFlatParams,
    comm: &dyn Communicator,
    out_dir: &Path,
) -> Result<Option<Path>> {
    let world = comm.world_size();
    let rank = comm.rank();

    if params.nlist == 0 {
        return Err(Error::invalid_input("nlist must be > 0", location!()));
    }
    if let Some(ref c) = params.pretrained_centroids {
        if c.len() != params.nlist {
            return Err(Error::invalid_input(
                format!("centroids.len()={} != nlist={}", c.len(), params.nlist),
                location!(),
            ));
        }
    }

    // Prepare centroids: either provided or error out (training should have happened upstream)
    let centroids = match &params.pretrained_centroids {
        Some(c) => c.as_ref().clone(),
        None => {
            // Minimal viable path: training is expected upstream in this MR step.
            // We short-circuit with an explicit error to guide usage.
            return Err(Error::invalid_input(
                "pretrained centroids not provided; please train and pass centroids.",
                location!(),
            ));
        }
    };

    // Broadcast centroids: no-op for LocalCommunicator but keep API parity.
    // We clone into Vec<Vec<f32>> for interface, then back; to avoid extra copying we skip and
    // rely on caller to pass identical centroids on all ranks for LocalCommunicator.
    let dim = centroids.value_length() as usize;

    // Compute local partition ids using the single-node transformer
    let ivf = new_ivf_transformer(centroids.clone(), dist, vec![]);
    let part_ids: UInt32Array = ivf.compute_partitions(local_vectors)?;

    // Create a temporary per-worker shard lance file under shard_root or system temp
    let shard_root = match &params.shard_root {
        Some(p) => p.clone(),
        None => {
            let tmp = TempDir::new()?.keep();
            Path::from_filesystem_path(tmp).map_err(|e| Error::IO {
                source: Box::new(e),
                location: location!(),
            })?
        }
    };
    let per_worker_dir = shard_root.child(format!("ivf_flat_shard_rank{}", rank));

    let store = ObjectStore::local();
    // Build the batch schema: [ROW_ID, PART_ID, FLAT]
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(PART_ID_COLUMN, DataType::UInt32, true),
        FlatQuantizer::new(dim, dist).field(),
    ]));

    // Construct row ids with unique base per worker.
    // If global_row_offset is None, compute a deterministic prefix sum of shard sizes
    // based on rank order using communicator allreduce (no new dependency, std-only).
    let base: u64 = if let Some(off) = global_row_offset {
        off
    } else {
        let local_len = local_vectors.len();
        // Build a fake allreduce with k=world and dim=1 where each rank contributes its local_len
        // at index `rank`. The result gives per-rank counts across all workers.
        let mut local_sums = vec![vec![0f32; 1]; world];
        let mut local_cnt = vec![0usize; world];
        local_sums[rank][0] = local_len as f32;
        local_cnt[rank] = local_len;
        let (_gs, gc) = comm.allreduce_sums_counts(&local_sums, &local_cnt);
        // Compute prefix sum of counts for ranks < self.rank
        gc.iter().take(rank).sum::<usize>() as u64
    };
    let row_ids =
        UInt64Array::from_iter_values((0..local_vectors.len()).map(|i| base + (i as u64)));

    // Replace vector column with FLAT (for FLAT quantizer, it's identical slice)
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(row_ids) as ArrayRef,
            Arc::new(part_ids) as ArrayRef,
            Arc::new(local_vectors.clone()) as ArrayRef,
        ],
    )?;

    // Write per-worker shard file (unsorted buffer)
    let shard_unsorted = per_worker_dir.child("unsorted.lance");
    let writer = store.create(&shard_unsorted).await?;
    let lance_schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref())?;
    let mut v2w = V2FileWriter::try_new(writer, lance_schema, V2WriterOptions::default())?;
    v2w.write_batch(&batch).await?;
    v2w.finish().await?;

    // Barrier before merge
    comm.barrier();

    // Merge stage: only rank 0 performs merge
    if rank != 0 {
        // Non-root returns after barrier; unified file will be at out_dir/auxiliary.idx
        return Ok(None);
    }

    // Collect all shard paths deterministically (rank0 assumes same directory layout)
    let mut shard_paths = Vec::with_capacity(world);
    for r in 0..world {
        let per = shard_root.child(format!("ivf_flat_shard_rank{}", r));
        shard_paths.push(per.child("unsorted.lance"));
    }

    let aux_path =
        merge_ivf_flat_shards(&shard_paths, centroids.clone(), dist, out_dir, dim).await?;

    // Final barrier to let other ranks proceed after merge
    comm.barrier();

    Ok(Some(aux_path))
}

/// Merge multiple per-worker IVF_FLAT shards into a unified auxiliary.idx
///
/// Each shard file is expected to contain columns: [ROW_ID, PART_ID_COLUMN, FLAT]
/// We will:
/// - read all shards
/// - group rows by PART_ID in ascending order, concatenate batches per partition in the
///   natural order of files
/// - write a single v2 Lance file with schema [ROW_ID, FLAT]
/// - set metadata: DISTANCE_TYPE_KEY, STORAGE_METADATA_KEY (FlatMetadata vec), IVF_METADATA_KEY
///   (points to a global buffer containing serialized Ivf protobuf)
#[allow(clippy::needless_range_loop)]
async fn merge_ivf_flat_shards(
    shard_files: &[Path],
    centroids: FixedSizeListArray,
    dist: DistanceType,
    out_dir: &Path,
    dim: usize,
) -> Result<Path> {
    use arrow_array::Float32Array;
    use futures::TryStreamExt as _;
    use lance_file::v2::reader::{FileReader as V2Reader, FileReaderOptions};
    use lance_io::{
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };

    let store = ObjectStore::local();

    // Prepare per-list buffers
    let nlist = centroids.len();
    let mut per_list_rows: Vec<Vec<(u64, ArrayRef)>> = vec![Vec::new(); nlist];
    let mut per_list_counts: Vec<u32> = vec![0; nlist];
    // Track row id ranges per shard for basic conflict detection
    let mut shard_ranges: Vec<(u64, u64, Path)> = Vec::new();

    for path in shard_files {
        // If shard file doesn't exist (e.g. empty worker), skip gracefully
        if !store.exists(path).await? {
            continue;
        }
        let sched = ScanScheduler::new(
            Arc::new(store.clone()),
            SchedulerConfig::max_bandwidth(&store),
        );
        let fh = sched.open_file(path, &CachedFileSize::unknown()).await?;
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;
        // Read full file
        let _schema: ArrowSchema = reader.schema().as_ref().into();
        let mut stream = reader.read_stream(
            lance_io::ReadBatchParams::RangeFull,
            u32::MAX,
            4,
            lance_encoding::decoder::FilterExpression::no_filter(),
        )?;
        // Track min/max row id within this shard
        let mut shard_min: Option<u64> = None;
        let mut shard_max: Option<u64> = None;
        while let Some(batch) = stream.try_next().await? {
            // Validate & extract columns
            let row_ids = batch
                .column_by_name(ROW_ID)
                .ok_or_else(|| Error::Index {
                    message: format!("missing {} in shard", ROW_ID),
                    location: location!(),
                })?
                .as_primitive::<arrow_array::types::UInt64Type>();
            let part_ids = batch
                .column_by_name(PART_ID_COLUMN)
                .ok_or_else(|| Error::Index {
                    message: format!("missing {} in shard", PART_ID_COLUMN),
                    location: location!(),
                })?
                .as_primitive::<arrow_array::types::UInt32Type>();
            let flat = batch
                .column_by_name(crate::vector::flat::storage::FLAT_COLUMN)
                .ok_or_else(|| Error::Index {
                    message: "missing FLAT column in shard".to_string(),
                    location: location!(),
                })?
                .as_fixed_size_list();

            // Iterate rows and push to per-list buffers (batching by contiguous list segments could be optimized)
            for i in 0..batch.num_rows() {
                let pid = part_ids.value(i) as usize;
                if pid >= nlist {
                    continue;
                }
                let rid = row_ids.value(i);
                // Update local min/max for this shard
                shard_min = Some(match shard_min {
                    Some(m) => m.min(rid),
                    None => rid,
                });
                shard_max = Some(match shard_max {
                    Some(m) => m.max(rid),
                    None => rid,
                });
                let vec_arr = flat.value(i); // ArrayRef for vector length dim
                per_list_rows[pid].push((rid, vec_arr));
                per_list_counts[pid] = per_list_counts[pid].saturating_add(1);
            }
        }
        if let (Some(minv), Some(maxv)) = (shard_min, shard_max) {
            shard_ranges.push((minv, maxv, path.clone()));
        }
    }

    // Basic interval overlap detection across shard row id ranges
    if shard_ranges.len() > 1 {
        shard_ranges.sort_by_key(|(minv, _, _)| *minv);
        let mut prev_min = shard_ranges[0].0;
        let mut prev_max = shard_ranges[0].1;
        let mut prev_path = shard_ranges[0].2.clone();
        for (minv, maxv, path) in shard_ranges.iter().skip(1) {
            if *minv <= prev_max {
                return Err(Error::Index {
                    message: format!(
                        "row id ranges overlap: [{}-{}] ({}) vs [{}-{}] ({})",
                        prev_min, prev_max, prev_path, *minv, *maxv, path
                    ),
                    location: location!(),
                });
            }
            if *maxv > prev_max {
                prev_max = *maxv;
                prev_path = path.clone();
            }
            prev_min = *minv;
        }
    }

    // Prepare unified writer: auxiliary.idx with [ROW_ID, FLAT] columns
    let aux_path = out_dir.child(crate::INDEX_AUXILIARY_FILE_NAME);
    let writer = store.create(&aux_path).await?;

    let arrow_schema = ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        FlatQuantizer::new(dim, dist).field(),
    ]);
    let mut v2w = V2FileWriter::try_new(
        writer,
        lance_core::datatypes::Schema::try_from(&arrow_schema)?,
        V2WriterOptions::default(),
    )?;

    // Set basic metadata: distance type
    v2w.add_schema_metadata(DISTANCE_TYPE_KEY, dist.to_string());

    // Flat quantizer metadata list (single entry)
    let flat_meta = FlatMetadata { dim };
    let meta_json = serde_json::to_string(&flat_meta)?;
    let meta_vec_json = serde_json::to_string(&vec![meta_json])?;
    v2w.add_schema_metadata(STORAGE_METADATA_KEY, meta_vec_json);

    // Compute IVF offsets/lengths while writing rows ordered by partition id
    let mut ivf_model = IvfStorageModel::new(centroids.clone(), None);

    // For each partition, write rows in appended batches
    // We'll batch in chunks to avoid giant single batches in extreme cases; tests are small.
    const CHUNK: usize = 8192;
    for pid in 0..nlist {
        let rows = &per_list_rows[pid];
        let total = rows.len();
        if total == 0 {
            ivf_model.add_partition(0);
            continue;
        }
        let mut written = 0usize;
        while written < total {
            let end = (written + CHUNK).min(total);
            let slice = &rows[written..end];
            let row_ids = UInt64Array::from_iter_values(slice.iter().map(|(rid, _)| *rid));
            // Build vectors fsl by concatenating items (already FSL rows)
            // We must build a FixedSizeListArray from values; easiest path: collect to f32 vec
            let mut values: Vec<f32> = Vec::with_capacity((end - written) * dim);
            for (_, arr) in slice.iter() {
                let vals = arr.as_primitive::<arrow_array::types::Float32Type>();
                values.extend_from_slice(vals.values());
            }
            let f32_vals = Float32Array::from(values);
            let vectors = Arc::new(FixedSizeListArray::try_new_from_values(
                f32_vals, dim as i32,
            )?); // FixedSizeListArrayExt
            let batch = RecordBatch::try_new(
                Arc::new(arrow_schema.clone()),
                vec![Arc::new(row_ids) as ArrayRef, vectors as ArrayRef],
            )?;
            v2w.write_batch(&batch).await?;
            written = end;
        }
        ivf_model.add_partition(total as u32);
    }

    // Write IVF metadata (as a global buffer) and reference in schema metadata
    let pb_ivf: crate::pb::Ivf = (&ivf_model).try_into()?; // prost::Message::encode_to_vec
    let ivf_bytes = Bytes::from(pb_ivf.encode_to_vec());
    let ivf_buf_index = v2w.add_global_buffer(ivf_bytes).await?; // index into global buffers table
    v2w.add_schema_metadata(IVF_METADATA_KEY, ivf_buf_index.to_string());

    // Finish writer
    v2w.finish().await?;

    Ok(aux_path)
}

/// Helper to wrap a FixedSizeListArray into a RecordBatchStream with PART_ID assigned.
pub fn fsl_with_partitions_to_stream(
    vectors: FixedSizeListArray,
    part_ids: UInt32Array,
) -> Result<Box<dyn RecordBatchStream + Unpin + 'static>> {
    if vectors.len() != part_ids.len() {
        return Err(Error::invalid_input(
            format!(
                "vectors.len() {} != part_ids.len() {}",
                vectors.len(),
                part_ids.len()
            ),
            location!(),
        ));
    }
    let dim = vectors.value_length();
    let schema = Arc::new(ArrowSchema::new(vec![
        (*ROW_ID_FIELD).clone(),
        Field::new(PART_ID_COLUMN, DataType::UInt32, true),
        FlatQuantizer::new(dim as usize, DistanceType::L2).field(),
    ]));
    // Caller will project/rename row_id if needed; here just provide a placeholder stream
    let empty_row_ids =
        Arc::new(UInt64Array::from_iter_values(0..vectors.len() as u64)) as ArrayRef;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            empty_row_ids,
            Arc::new(part_ids) as ArrayRef,
            Arc::new(vectors) as ArrayRef,
        ],
    )?;
    let stream = futures::stream::iter(vec![Ok(batch)]);
    Ok(Box::new(RecordBatchStreamAdapter::new(schema, stream)))
}
