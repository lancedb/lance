// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SuffixArrayIndexPlugin implementing ScalarIndexPlugin.
//!
//! Handles training (building) and loading of suffix array indices.
//! For corpora that exceed `MAX_SEGMENT_BYTES` (~512 MB) the build
//! automatically splits the data into multiple segments, each with
//! its own suffix array.  This keeps individual suffix arrays within
//! the i32 addressing limit while providing transparent aggregation
//! at query time.
//!
//! Segment SA construction is parallelized using rayon to saturate
//! available CPU cores, providing near-linear speedup on multi-core
//! machines.

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::{Error, Result};
use prost::Message;
use rayon::prelude::*;
use serde::Deserialize;
use tracing::info;

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::progress::IndexBuildProgress;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{CreatedIndex, IndexStore, ScalarIndex};

use super::builder::{build_suffix_array, compact_suffix_array, compute_pointer_width, release_memory_to_os};
use super::index::SuffixArrayIndex;

const SUFFIX_ARRAY_INDEX_VERSION: u32 = 0;

/// Maximum text bytes per segment (~512 MB).
/// Each segment needs its own suffix array with i32 pointers, so must be < 2 GB.
/// 512 MB balances SA working set size vs. parallelism granularity.
const MAX_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

/// Maximum number of parallel SA builder threads.
/// Each builder uses ~5×N memory (N = segment size), so 4 threads × 512MB
/// = ~10 GB for SA construction. Combined with segment text buffers (~19 GB
/// for large datasets), total stays within typical 96 GB cgroup limits.
/// Configurable via LANCE_SA_PARALLEL_BUILDERS env var.
fn max_parallel_builders() -> usize {
    std::env::var("LANCE_SA_PARALLEL_BUILDERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}


/// Result of building a suffix array for a single segment.
struct BuiltSegment {
    seg_idx: usize,
    text: Vec<u8>,
    compact_sa: Vec<u8>,
    pointer_width: usize,
    corpus_bytes: u64,
    /// Cumulative byte offsets marking the end of each document.
    /// `doc_offsets[i]` is the byte offset in `text` where document `i` ends.
    /// Document `i` spans `[doc_offsets[i-1], doc_offsets[i])` (with doc_offsets[-1] = 0).
    doc_offsets: Vec<u64>,
}

/// Plugin for creating and loading suffix array indices.
#[derive(Debug, Default)]
pub struct SuffixArrayIndexPlugin;

/// Parameters parsed from the JSON params string for SA index creation.
#[derive(Debug, Clone, Deserialize, Default)]
struct SuffixArrayBuildParams {
    /// When true, lowercase all text at build time for case-insensitive matching.
    #[serde(default)]
    case_insensitive: bool,
    /// Separator token ID placed between documents in token-level indices.
    /// Defaults to 0xFFFF for u16 tokens and 0xFFFFFFFF for u32 tokens.
    #[serde(default)]
    separator_token_id: Option<u32>,
}

/// Training request that carries SA build parameters.
struct SuffixArrayTrainingRequest {
    criteria: TrainingCriteria,
    params: SuffixArrayBuildParams,
}

impl TrainingRequest for SuffixArrayTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// Write one pre-built segment's (text, compact_sa, doc_offsets) to the index store.
async fn write_segment(
    index_store: &dyn IndexStore,
    built: &BuiltSegment,
    num_segments: usize,
) -> Result<pb::SuffixArraySegmentInfo> {
    let seg_idx = built.seg_idx;

    // Choose file names: single-segment uses legacy names for backwards compat
    let (tok_name, sa_name, offsets_name) = if num_segments <= 1 {
        (
            "tokenized.bin".to_string(),
            "suffix_array.bin".to_string(),
            "doc_offsets.bin".to_string(),
        )
    } else {
        (
            format!("segment_{seg_idx}_tokenized.bin"),
            format!("segment_{seg_idx}_suffix_array.bin"),
            format!("segment_{seg_idx}_doc_offsets.bin"),
        )
    };

    // Write tokenized text
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "data",
        arrow_schema::DataType::LargeBinary,
        false,
    )]));
    let batch = arrow_array::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![built.text.as_slice()]))],
    )?;
    let mut writer = index_store.new_index_file(&tok_name, schema.clone()).await?;
    writer.write_record_batch(batch).await?;
    writer.finish().await?;

    // Write suffix array
    let sa_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "data",
        arrow_schema::DataType::LargeBinary,
        false,
    )]));
    let sa_batch = arrow_array::RecordBatch::try_new(
        sa_schema.clone(),
        vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
            built.compact_sa.as_slice(),
        ]))],
    )?;
    let mut sa_writer = index_store.new_index_file(&sa_name, sa_schema).await?;
    sa_writer.write_record_batch(sa_batch).await?;
    sa_writer.finish().await?;

    // Write document offsets (u64 array, one per document)
    let offsets_bytes: Vec<u8> = built.doc_offsets.iter()
        .flat_map(|o| o.to_le_bytes())
        .collect();
    let offsets_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "data",
        arrow_schema::DataType::LargeBinary,
        false,
    )]));
    let offsets_batch = arrow_array::RecordBatch::try_new(
        offsets_schema.clone(),
        vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
            offsets_bytes.as_slice(),
        ]))],
    )?;
    let mut offsets_writer = index_store.new_index_file(&offsets_name, offsets_schema).await?;
    offsets_writer.write_record_batch(offsets_batch).await?;
    offsets_writer.finish().await?;

    // Return freed memory to the OS. Without this, glibc retains freed pages
    // in the process address space, causing RSS to grow monotonically across
    // segments and eventually OOM during the commit phase.
    release_memory_to_os();

    Ok(pb::SuffixArraySegmentInfo {
        pointer_width: built.pointer_width as u32,
        total_entries: built.corpus_bytes,
        corpus_bytes: built.corpus_bytes,
        num_documents: built.doc_offsets.len() as u64,
    })
}

/// Extract text bytes from a column array, appending to `out`.
/// Also records the cumulative byte offset after each document in `doc_offsets`.
/// When `case_insensitive` is true, text is lowercased before appending.
/// For list<int16/int32> columns, each element's tokens are serialized as
/// fixed-width little-endian integers with a separator token between documents.
/// Returns the number of non-null rows processed.
fn extract_bytes(
    col: &dyn arrow_array::Array,
    out: &mut Vec<u8>,
    doc_offsets: &mut Vec<u64>,
    case_insensitive: bool,
    separator_token_id: Option<u32>,
) -> Result<u64> {
    let mut docs = 0u64;
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<arrow_array::StringArray>().ok_or_else(|| {
                Error::invalid_input("Failed to downcast to StringArray")
            })?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    if case_insensitive {
                        out.extend_from_slice(arr.value(i).to_lowercase().as_bytes());
                    } else {
                        out.extend_from_slice(arr.value(i).as_bytes());
                    }
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<arrow_array::LargeStringArray>().ok_or_else(|| {
                Error::invalid_input("Failed to downcast to LargeStringArray")
            })?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    if case_insensitive {
                        out.extend_from_slice(arr.value(i).to_lowercase().as_bytes());
                    } else {
                        out.extend_from_slice(arr.value(i).as_bytes());
                    }
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::Binary => {
            let arr = col.as_any().downcast_ref::<arrow_array::BinaryArray>().ok_or_else(|| {
                Error::invalid_input("Failed to downcast to BinaryArray")
            })?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.extend_from_slice(arr.value(i));
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::LargeBinary => {
            let arr = col.as_any().downcast_ref::<arrow_array::LargeBinaryArray>().ok_or_else(|| {
                Error::invalid_input("Failed to downcast to LargeBinaryArray")
            })?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.extend_from_slice(arr.value(i));
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::List(_) | DataType::LargeList(_) => {
            docs = extract_token_list_bytes(col, out, doc_offsets, separator_token_id)?;
        }
        other => {
            return Err(Error::invalid_input(format!(
                "Unsupported column type for suffix array index: {other:?}"
            )));
        }
    }
    Ok(docs)
}

/// Extract token IDs from a list column and serialize them as fixed-width
/// little-endian bytes. Inserts a separator token between documents.
///
/// Supports List<Int16>, List<UInt16>, List<Int32>, List<UInt32> and
/// their LargeList variants. Returns the number of non-null rows.
fn extract_token_list_bytes(
    col: &dyn arrow_array::Array,
    out: &mut Vec<u8>,
    doc_offsets: &mut Vec<u64>,
    separator_token_id: Option<u32>,
) -> Result<u64> {
    use arrow_array::{Int16Array, Int32Array, Int64Array, UInt16Array, UInt32Array, UInt64Array};

    // Determine element type and token width from the list's value type.
    let (elem_type, values_ref) = if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
        (list_arr.values().data_type().clone(), list_arr.values().clone())
    } else if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
        (list_arr.values().data_type().clone(), list_arr.values().clone())
    } else {
        return Err(Error::invalid_input("Expected List or LargeList array"));
    };

    // Int64/UInt64 are downcast to 4-byte tokens (validated below).
    let token_width: usize = match &elem_type {
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Int64 | DataType::UInt64 => 4,
        other => {
            return Err(Error::invalid_input(format!(
                "Unsupported list element type for token SA: {other:?}"
            )));
        }
    };

    let sep = separator_token_id.unwrap_or_else(|| {
        if token_width == 2 { 0xFFFF } else { 0xFFFF_FFFF }
    });
    let sep_bytes: Vec<u8> = sep.to_le_bytes()[..token_width].to_vec();

    let num_rows = col.len();
    let mut docs = 0u64;

    // Helper: get (start, end) offsets for row i, or None if null.
    let get_offsets = |i: usize| -> Option<(usize, usize)> {
        if col.is_null(i) {
            return None;
        }
        if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            Some((start, end))
        } else if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            Some((start, end))
        } else {
            None
        }
    };

    for i in 0..num_rows {
        let Some((start, end)) = get_offsets(i) else { continue };

        // Insert separator between documents (not before the first)
        if docs > 0 {
            out.extend_from_slice(&sep_bytes);
        }

        // Serialize each token as little-endian bytes
        match &elem_type {
            DataType::Int16 => {
                let arr = values_ref.as_any().downcast_ref::<Int16Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::UInt16 => {
                let arr = values_ref.as_any().downcast_ref::<UInt16Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::Int32 => {
                let arr = values_ref.as_any().downcast_ref::<Int32Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::UInt32 => {
                let arr = values_ref.as_any().downcast_ref::<UInt32Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::Int64 => {
                let arr = values_ref.as_any().downcast_ref::<Int64Array>().unwrap();
                for j in start..end {
                    let v = arr.value(j);
                    if v < i32::MIN as i64 || v > i32::MAX as i64 {
                        return Err(Error::invalid_input(format!(
                            "Token value {v} at row {i} exceeds i32 range. \
                             Cast your column to list<int32> or use smaller token IDs."
                        )));
                    }
                    out.extend_from_slice(&(v as i32).to_le_bytes());
                }
            }
            DataType::UInt64 => {
                let arr = values_ref.as_any().downcast_ref::<UInt64Array>().unwrap();
                for j in start..end {
                    let v = arr.value(j);
                    if v > u32::MAX as u64 {
                        return Err(Error::invalid_input(format!(
                            "Token value {v} at row {i} exceeds u32 range. \
                             Cast your column to list<uint32> or use smaller token IDs."
                        )));
                    }
                    out.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }

        doc_offsets.push(out.len() as u64);
        docs += 1;
    }

    Ok(docs)
}

#[async_trait]
impl ScalarIndexPlugin for SuffixArrayIndexPlugin {
    fn name(&self) -> &str {
        "SuffixArray"
    }

    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        // Validate field type: accept text/binary or list of integer tokens
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {}
            DataType::List(inner) | DataType::LargeList(inner) => {
                match inner.data_type() {
                    DataType::Int16 | DataType::UInt16 | DataType::Int32 | DataType::UInt32 => {}
                    // Int64/UInt64 accepted — values validated at build time and
                    // stored as 4-byte (i32) tokens.
                    DataType::Int64 | DataType::UInt64 => {}
                    other => {
                        return Err(Error::invalid_input(format!(
                            "Suffix array index on list columns requires Int16, UInt16, Int32, UInt32, \
                             Int64, or UInt64 elements, got {other:?}"
                        )));
                    }
                }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "Suffix array index requires Utf8, LargeUtf8, Binary, LargeBinary, or \
                     List<Int16/Int32/Int64> column, got {other:?}"
                )));
            }
        }
        let build_params: SuffixArrayBuildParams = if params.is_empty() {
            SuffixArrayBuildParams::default()
        } else {
            serde_json::from_str(params).map_err(|e| {
                Error::invalid_input(format!(
                    "Failed to parse suffix array index params: {e}"
                ))
            })?
        };
        Ok(Box::new(SuffixArrayTrainingRequest {
            criteria: TrainingCriteria::new(TrainingOrdering::None),
            params: build_params,
        }))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        SUFFIX_ARRAY_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        // Extract build params from the training request
        let build_params = request
            .as_any()
            .downcast_ref::<SuffixArrayTrainingRequest>()
            .map(|r| r.params.clone())
            .unwrap_or_default();
        let case_insensitive = build_params.case_insensitive;
        let separator_token_id = build_params.separator_token_id;

        if case_insensitive {
            info!("Building case-insensitive suffix array index (lowercasing all text)");
        }

        let schema = data.schema();
        let value_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "value")
            .ok_or_else(|| Error::invalid_input("Training data stream missing 'value' column"))?;

        // Detect token_width from the column type. For text/binary columns it's 1 (byte-level).
        // For list<int16/uint16> columns it's 2, for list<int32/uint32> it's 4.
        let value_field = &schema.fields()[value_col_idx];
        let token_width: u32 = match value_field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => 1,
            DataType::List(inner) | DataType::LargeList(inner) => {
                match inner.data_type() {
                    DataType::Int16 | DataType::UInt16 => 2,
                    DataType::Int32 | DataType::UInt32 => 4,
                    // Int64/UInt64 values are validated and stored as 4-byte tokens
                    DataType::Int64 | DataType::UInt64 => 4,
                    other => {
                        return Err(Error::invalid_input(format!(
                            "Unsupported list element type for suffix array: {other:?}"
                        )));
                    }
                }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "Unsupported column type for suffix array index: {other:?}"
                )));
            }
        };

        if token_width > 1 {
            let sep = separator_token_id.unwrap_or_else(|| {
                if token_width == 2 { 0xFFFF } else { 0xFFFF_FFFF }
            });
            info!(
                token_width,
                separator_token_id = sep,
                "Building token-level suffix array index"
            );
        }

        // ── Streaming pipeline: read → batch → build SA → write → repeat ──
        //
        // Instead of reading ALL text into memory first (which can OOM on large
        // datasets, especially when reading from S3), we interleave reading with
        // building and writing. We accumulate just enough text segments for one
        // batch, then build their suffix arrays, write them, free memory, and
        // continue reading from the stream.
        //
        // Peak memory ≈ batch_size × (512MB text + 2.5GB SA working set) ≈ 10-20GB
        // instead of ≈ 19GB text + batch × SA working set ≈ 40-80GB.

        let parallel_threads = max_parallel_builders();
        let mut stream = data;
        let mut current_bytes: Vec<u8> = Vec::new();
        let mut current_doc_offsets: Vec<u64> = Vec::new();
        let mut pending_segments: Vec<(Vec<u8>, Vec<u64>)> = Vec::new();
        let mut total_documents: u64 = 0;
        let mut total_corpus_bytes: u64 = 0;
        let mut segment_infos: Vec<pb::SuffixArraySegmentInfo> = Vec::new();
        let mut seg_counter: usize = 0;
        let mut batch_idx: usize = 0;
        let build_start = std::time::Instant::now();

        // Helper closure to process a batch of segments
        async fn process_batch(
            pending: &mut Vec<(Vec<u8>, Vec<u64>)>,
            parallel_threads: usize,
            seg_counter: &mut usize,
            batch_idx: &mut usize,
            segment_infos: &mut Vec<pb::SuffixArraySegmentInfo>,
            index_store: &dyn IndexStore,
        ) -> Result<()> {
            if pending.is_empty() {
                return Ok(());
            }

            let batch_size = parallel_threads.min(pending.len());
            let batch: Vec<(Vec<u8>, Vec<u64>)> = pending.drain(..batch_size).collect();
            let seg_offset = *seg_counter;
            *seg_counter += batch_size;

            info!(
                batch = *batch_idx,
                batch_size,
                seg_offset,
                remaining = pending.len(),
                "Building batch"
            );

            // Build this batch's SAs in parallel
            let built_batch = tokio::task::spawn_blocking(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(batch_size)
                    .build()
                    .map_err(|e| Error::io(
                        format!("Failed to create rayon thread pool: {e}"),
                    ))?;

                pool.install(|| {
                    batch.into_par_iter()
                        .enumerate()
                        .map(|(i, (text, doc_offsets))| {
                            let seg_idx = seg_offset + i;
                            let seg_start = std::time::Instant::now();
                            let corpus_bytes = text.len() as u64;
                            let pointer_width = compute_pointer_width(corpus_bytes);

                            let sa = build_suffix_array(&text);
                            let compact_sa = compact_suffix_array(&sa, pointer_width)?;
                            drop(sa);
                            release_memory_to_os();

                            let seg_elapsed = seg_start.elapsed();
                            info!(
                                segment = seg_idx,
                                bytes = corpus_bytes,
                                docs = doc_offsets.len(),
                                secs = seg_elapsed.as_secs_f64(),
                                mb_per_sec = corpus_bytes as f64 / seg_elapsed.as_secs_f64() / 1e6,
                                "Segment SA built"
                            );

                            Ok(BuiltSegment {
                                seg_idx,
                                text,
                                compact_sa,
                                pointer_width,
                                corpus_bytes,
                                doc_offsets,
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
            }).await.map_err(|e| Error::io(
                format!("SA builder task panicked: {e}"),
            ))??;

            // Always use multi-segment naming (segment_N_*) since we're streaming
            // and don't know the final segment count during writes.
            // For single-segment results, we set num_segments=1 in metadata but
            // still use multi-segment file naming, which load_multi handles.
            let force_multi = 2; // Forces segment_N_ naming in write_segment
            for built in &built_batch {
                let info = write_segment(index_store, built, force_multi).await?;
                segment_infos.push(info);
            }
            drop(built_batch);
            release_memory_to_os();

            *batch_idx += 1;
            Ok(())
        }

        // Read text from stream, accumulating segments. When we have enough
        // for a full batch, process it immediately.
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let col = batch.column(value_col_idx);
            let docs = extract_bytes(col.as_ref(), &mut current_bytes, &mut current_doc_offsets, case_insensitive, separator_token_id)?;
            total_documents += docs;

            // Split off complete segments as we accumulate data.
            // When splitting, doc_offsets need to be adjusted: offsets that fall
            // within the split-off portion stay with that segment (rebased to 0),
            // offsets beyond the split point remain in current_doc_offsets (rebased
            // to the new segment start).
            while current_bytes.len() >= MAX_SEGMENT_BYTES {
                let split_point = MAX_SEGMENT_BYTES as u64;
                let rest = current_bytes.split_off(MAX_SEGMENT_BYTES);

                // Partition doc_offsets: offsets <= split_point go with this segment,
                // the rest stay in current. Note: since offsets are cumulative and
                // sorted, we can binary search.
                let split_idx = current_doc_offsets.partition_point(|&o| o <= split_point);

                let seg_offsets: Vec<u64> = current_doc_offsets[..split_idx].to_vec();
                // Rebase remaining offsets relative to the new segment start
                let remaining_offsets: Vec<u64> = current_doc_offsets[split_idx..]
                    .iter()
                    .map(|&o| o - split_point)
                    .collect();

                let segment = std::mem::replace(&mut current_bytes, rest);
                current_doc_offsets = remaining_offsets;
                total_corpus_bytes += segment.len() as u64;
                pending_segments.push((segment, seg_offsets));

                // When we have a full batch, process it immediately
                if pending_segments.len() >= parallel_threads {
                    process_batch(
                        &mut pending_segments,
                        parallel_threads,
                        &mut seg_counter,
                        &mut batch_idx,
                        &mut segment_infos,
                        index_store,
                    ).await?;
                }
            }
        }

        // Push remaining bytes as the final segment
        if !current_bytes.is_empty() {
            total_corpus_bytes += current_bytes.len() as u64;
            pending_segments.push((current_bytes, current_doc_offsets));
        } else if !current_doc_offsets.is_empty() {
            // Edge case: last doc ended exactly at MAX_SEGMENT_BYTES
            // doc_offsets are empty vec (all docs were in split segment)
        }

        // Process any remaining segments
        while !pending_segments.is_empty() {
            process_batch(
                &mut pending_segments,
                parallel_threads,
                &mut seg_counter,
                &mut batch_idx,
                &mut segment_infos,
                index_store,
            ).await?;
        }

        let num_segments = segment_infos.len();

        // Handle empty dataset
        if num_segments == 0 {
            // Write a single empty segment
            let empty_segment = BuiltSegment {
                seg_idx: 0,
                text: Vec::new(),
                compact_sa: Vec::new(),
                pointer_width: 1,
                corpus_bytes: 0,
                doc_offsets: Vec::new(),
            };
            let info = write_segment(index_store, &empty_segment, 1).await?;
            segment_infos.push(info);
        }

        let num_segments = segment_infos.len();
        let build_elapsed = build_start.elapsed();
        info!(
            total_corpus_bytes,
            total_documents,
            num_segments,
            build_secs = build_elapsed.as_secs_f64(),
            parallel_threads,
            batches = batch_idx,
            "Streaming build complete"
        );

        // Build protobuf details.
        // Since streaming always uses multi-segment file naming (segment_N_*),
        // we always populate the segments array, even for single-segment indices.
        // This ensures load_multi is used at read time, which matches the
        // file naming convention.
        let details = pb::SuffixArrayIndexDetails {
            token_width,
            pointer_width: 0,
            total_tokens: 0,
            total_documents,
            corpus_bytes: total_corpus_bytes,
            tokenizer_name: None,
            vocab_size: None,
            separator_token_id: if token_width > 1 {
                let sep = separator_token_id.unwrap_or_else(|| {
                    if token_width == 2 { 0xFFFF } else { 0xFFFF_FFFF }
                });
                Some(sep)
            } else {
                None
            },
            num_segments: num_segments as u32,
            segments: segment_infos,
            case_insensitive,
        };

        let index_details = prost_types::Any::from_msg(&details)?;

        Ok(CreatedIndex {
            index_details,
            index_version: SUFFIX_ARRAY_INDEX_VERSION,
            files: Some(index_store.list_files_with_sizes().await?),
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &lance_core::cache::LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let details = pb::SuffixArrayIndexDetails::decode(&*index_details.value)?;
        let case_insensitive = details.case_insensitive;
        let token_width = details.token_width.max(1) as u8;

        if !details.segments.is_empty() {
            // Multi-segment index (v1+) — also used for single-segment indices
            // built with streaming pipeline that always uses segment_N_ naming.
            info!(
                num_segments = details.segments.len(),
                case_insensitive,
                token_width,
                "Loading multi-segment suffix array index"
            );
            Ok(SuffixArrayIndex::load_multi(
                index_store,
                &details.segments,
                case_insensitive,
                token_width,
            ).await? as Arc<dyn ScalarIndex>)
        } else {
            // Single-segment index (v0 / backwards compat)
            let pointer_width = details.pointer_width as u8;
            let total_entries = details.total_tokens;
            Ok(SuffixArrayIndex::load(
                index_store,
                frag_reuse_index,
                cache,
                pointer_width,
                total_entries,
                case_insensitive,
                token_width,
            )
            .await? as Arc<dyn ScalarIndex>)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::suffix_array::builder::{
        build_suffix_array, compact_suffix_array, compute_pointer_width,
    };
    use crate::scalar::suffix_array::query;

    #[test]
    fn test_sa_construction_small() {
        let data = b"abcabc";
        let sa = build_suffix_array(data);
        assert_eq!(sa.len(), data.len());

        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at position {i}: {s1:?} >= {s2:?}");
        }
    }

    #[test]
    fn test_binary_search_correctness() {
        let data = b"mississippi";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"issi"), 2);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"ss"), 2);
        assert_eq!(
            query::count(data, &compact, ptr_width, n, b"mississippi"),
            1
        );
        assert_eq!(query::count(data, &compact, ptr_width, n, b"z"), 0);
    }

    #[test]
    fn test_count_returns_correct_counts() {
        let data = b"the cat sat on the mat";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"the"), 2);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"at"), 3);
        assert_eq!(query::count(data, &compact, ptr_width, n, b" "), 5);
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let sa = build_suffix_array(data);
        assert!(sa.is_empty());

        let ptr_width = compute_pointer_width(0);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        assert_eq!(query::count(data, &compact, ptr_width, 0, b"anything"), 0);
    }

    #[test]
    fn test_single_character_data() {
        let data = b"x";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"x"), 1);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"y"), 0);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"xx"), 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_train_load_query_roundtrip() {
        use std::sync::Arc;

        use arrow_array::{RecordBatch, StringArray, UInt64Array};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use futures::stream;
        use lance_core::cache::LanceCache;
        use lance_core::utils::tempfile::TempDir;
        use lance_io::object_store::ObjectStore;

        use crate::metrics::NoOpMetricsCollector;
        use crate::progress::NoopIndexBuildProgress;
        use crate::scalar::lance_format::LanceIndexStore;
        use crate::scalar::registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME};

        use super::super::query::SuffixArrayQuery;
        use super::SuffixArrayIndexPlugin;

        let texts = vec![
            "the quick brown fox",
            "jumps over the lazy dog",
            "the cat sat on the mat",
        ];
        let text_array = StringArray::from(texts.clone());
        let row_ids = UInt64Array::from_iter_values(0..3u64);
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(text_array), Arc::new(row_ids)],
        )
        .unwrap();

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::once(std::future::ready(Ok(batch))),
        ));

        let plugin = SuffixArrayIndexPlugin;
        let field = Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false);
        let request = plugin.new_training_request("", &field).unwrap();

        let tmpdir = Arc::new(TempDir::default());
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let progress = Arc::new(NoopIndexBuildProgress);
        let created = plugin
            .train_index(data_stream, store.as_ref(), request, None, progress)
            .await
            .unwrap();

        assert!(created.files.is_some());
        let files = created.files.as_ref().unwrap();
        assert!(files.iter().any(|f| f.path.contains("tokenized")));
        assert!(files.iter().any(|f| f.path.contains("suffix_array")));

        let loaded_index = plugin
            .load_index(
                store.clone(),
                &created.index_details,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();

        let metrics = NoOpMetricsCollector;

        // Count "the" — should appear 4 times
        let count_query = SuffixArrayQuery::Count {
            query_bytes: b"the".to_vec(),
        };
        let result = loaded_index.search(&count_query, &metrics).await.unwrap();
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                let row_addrs: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|addr| addr.into())
                    .collect();
                assert!(!row_addrs.is_empty(), "Expected count > 0 for 'the' query");
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }

        // Count "xyz" — should not appear
        let count_query = SuffixArrayQuery::Count {
            query_bytes: b"xyz".to_vec(),
        };
        let result = loaded_index.search(&count_query, &metrics).await.unwrap();
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                let row_addrs: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|addr| addr.into())
                    .collect();
                assert!(row_addrs.is_empty(), "Expected count = 0 for 'xyz' query");
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }

        // Search for positions of "fox"
        let search_query = SuffixArrayQuery::Search {
            query_bytes: b"fox".to_vec(),
            max_results: 10,
        };
        let result = loaded_index.search(&search_query, &metrics).await.unwrap();
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                let row_addrs: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|addr| addr.into())
                    .collect();
                assert_eq!(
                    row_addrs.len(),
                    1,
                    "Expected exactly 1 match for 'fox', got {:?}",
                    row_addrs
                );
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_multi_segment_roundtrip() {
        // Verify that a multi-segment index produces the same counts as a
        // single-segment index over the same corpus.
        use bytes::Bytes;

        use crate::metrics::NoOpMetricsCollector;
        use crate::scalar::ScalarIndex;

        use super::super::builder::{build_suffix_array, compact_suffix_array, compute_pointer_width};
        use super::super::index::{SuffixArrayIndex, SuffixArraySegment};
        use super::super::query::SuffixArrayQuery;

        // Build a corpus split across two segments
        let text1 = b"the cat sat on the mat";
        let text2 = b"the dog ran in the park";

        let sa1 = build_suffix_array(text1);
        let pw1 = compute_pointer_width(text1.len() as u64);
        let csa1 = compact_suffix_array(&sa1, pw1).unwrap();

        let sa2 = build_suffix_array(text2);
        let pw2 = compute_pointer_width(text2.len() as u64);
        let csa2 = compact_suffix_array(&sa2, pw2).unwrap();

        let index = SuffixArrayIndex::from_segments(vec![
            SuffixArraySegment {
                tokenized: Bytes::from(text1.to_vec()),
                suffix_array: Bytes::from(csa1),
                pointer_width: pw1 as u8,
                total_entries: text1.len() as u64,
                doc_offsets: vec![text1.len() as u64],
            },
            SuffixArraySegment {
                tokenized: Bytes::from(text2.to_vec()),
                suffix_array: Bytes::from(csa2),
                pointer_width: pw2 as u8,
                total_entries: text2.len() as u64,
                doc_offsets: vec![text2.len() as u64],
            },
        ]);

        // "the" appears 2 times in text1 + 2 times in text2 = 4
        assert_eq!(index.total_count(b"the"), 4);
        // "cat" appears once in text1 only
        assert_eq!(index.total_count(b"cat"), 1);
        // "dog" appears once in text2 only
        assert_eq!(index.total_count(b"dog"), 1);
        // "xyz" appears nowhere
        assert_eq!(index.total_count(b"xyz"), 0);

        // Test prob across segments
        let prob_result = index.compute_prob(b"the ", b"cat");
        // "the " appears: text1 has "the cat" and "the mat" = 2, text2 has "the dog" and "the park" = 2, total = 4
        assert_eq!(prob_result.prompt_cnt, 4);
        // "the cat" appears once (text1 only)
        assert_eq!(prob_result.cont_cnt, 1);
        assert!((prob_result.prob - 0.25).abs() < 1e-10);

        // Test ntd across segments
        let ntd_result = index.compute_ntd(b"the ", None);
        assert_eq!(ntd_result.prompt_cnt, 4);
        // Should have 4 distinct next bytes: 'c', 'm', 'd', 'p'
        assert_eq!(ntd_result.distribution.len(), 4);

        // Test infgram_prob across segments
        let igp = index.compute_infgram_prob(b"the ", b"cat");
        assert_eq!(igp.prob_result.prompt_cnt, 4);
        assert_eq!(igp.prob_result.cont_cnt, 1);
        assert_eq!(igp.effective_suffix_len, 4);

        // Test via ScalarIndex::search
        let metrics = NoOpMetricsCollector;
        let count_query = SuffixArrayQuery::Count {
            query_bytes: b"the".to_vec(),
        };
        let result = index.search(&count_query, &metrics).await.unwrap();
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                let count: u64 = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .sum();
                assert_eq!(count, 4, "Expected 4 occurrences of 'the' across 2 segments");
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }
    }

    #[test]
    fn test_plugin_rejects_invalid_field_type() {
        use crate::scalar::registry::ScalarIndexPlugin;
        use arrow_schema::{DataType, Field};

        let plugin = super::SuffixArrayIndexPlugin;
        let field = Field::new("value", DataType::Int32, false);
        let result = plugin.new_training_request("", &field);
        assert!(result.is_err(), "Int32 field should be rejected");
    }

    #[test]
    fn test_plugin_accepts_valid_field_types() {
        use crate::scalar::registry::ScalarIndexPlugin;
        use arrow_schema::{DataType, Field};

        let plugin = super::SuffixArrayIndexPlugin;
        for dt in &[
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
        ] {
            let field = Field::new("value", dt.clone(), false);
            assert!(
                plugin.new_training_request("", &field).is_ok(),
                "Should accept {:?}",
                dt
            );
        }
    }
}
