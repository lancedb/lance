// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): child node stored as a tabular Lance v2 file.
//!
//! "Child nodes are Lance files, not protobuf." A child owns a contiguous
//! fragment-id range and stores one row per fragment: `frag_id` plus the
//! serialized `DataFragment`. No table-level metadata. The child's own ε-buffer
//! would live in a global buffer of the same file (recursion-ready; unused at
//! the two levels this prototype benchmarks).

use std::sync::Arc;

use arrow_array::{BinaryArray, RecordBatch, UInt64Array, cast::AsArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use prost::Message;

use crate::format::Fragment;
use crate::format::pb;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::ScanScheduler;
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

const READ_BATCH_ROWS: u32 = 16 * 1024;
const READ_BATCH_READAHEAD: u32 = 16;

/// Reads and writes child nodes as Lance v2 files against an object store.
pub struct ChildIo {
    object_store: Arc<ObjectStore>,
    scheduler: Arc<ScanScheduler>,
    cache: Arc<LanceCache>,
}

/// Result of writing a child node.
pub struct ChildWriteResult {
    pub num_rows: u64,
    pub byte_size: u64,
}

fn child_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("frag_id", DataType::UInt64, false),
        ArrowField::new("fragment_pb", DataType::Binary, false),
    ]))
}

impl ChildIo {
    pub fn new(
        object_store: Arc<ObjectStore>,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
    ) -> Self {
        Self {
            object_store,
            scheduler,
            cache,
        }
    }

    /// Write `fragments` (assumed sorted by id) to a child Lance file at `path`.
    pub async fn write(&self, path: &Path, fragments: &[Fragment]) -> Result<ChildWriteResult> {
        let arrow_schema = child_arrow_schema();
        let ids = UInt64Array::from_iter_values(fragments.iter().map(|f| f.id));
        let encoded: Vec<Vec<u8>> = fragments
            .iter()
            .map(|f| pb::DataFragment::from(f).encode_to_vec())
            .collect();
        let pbs = BinaryArray::from_iter_values(encoded.iter().map(|v| v.as_slice()));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(ids), Arc::new(pbs)])?;

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref())?;
        let writer = self.object_store.create(path).await?;
        let mut file_writer =
            FileWriter::try_new(writer, lance_schema, FileWriterOptions::default())?;
        file_writer.write_batch(&batch).await?;
        let summary = file_writer.finish().await?;
        Ok(ChildWriteResult {
            num_rows: summary.num_rows,
            byte_size: summary.size_bytes,
        })
    }

    /// Read a child Lance file at `path` back into a fragment list.
    ///
    /// `known_size` (from the child ref) avoids a size-discovery HEAD.
    pub async fn read(&self, path: &Path, known_size: Option<u64>) -> Result<Vec<Fragment>> {
        let size = known_size
            .map(CachedFileSize::new)
            .unwrap_or_else(CachedFileSize::unknown);
        let file_scheduler = self.scheduler.open_file(path, &size).await?;
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &self.cache,
            FileReaderOptions::default(),
        )
        .await?;

        let mut fragments = Vec::with_capacity(reader.num_rows() as usize);
        let mut stream = reader
            .read_stream(
                ReadBatchParams::RangeFull,
                READ_BATCH_ROWS,
                READ_BATCH_READAHEAD,
                FilterExpression::no_filter(),
            )
            .await?;
        while let Some(batch) = stream.try_next().await? {
            let pbs = batch
                .column_by_name("fragment_pb")
                .ok_or_else(|| Error::invalid_input("child file missing fragment_pb column"))?
                .as_binary::<i32>();
            for value in pbs.iter() {
                let bytes =
                    value.ok_or_else(|| Error::invalid_input("null fragment_pb in child file"))?;
                let fragment = Fragment::try_from(pb::DataFragment::decode(bytes)?)?;
                fragments.push(fragment);
            }
        }
        Ok(fragments)
    }
}
