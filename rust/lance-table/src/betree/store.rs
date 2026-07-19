// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): copy-on-write node IO.
//!
//! Nodes are immutable object-store files; every rewrite produces a new file
//! (uuid-named), so flush/split/merge just write new files and the parent
//! repoints. Leaves are tabular Lance v2 files (`frag_id`, `fragment_pb`);
//! internal nodes and the root are protobuf objects.

use std::sync::Arc;

use arrow_array::{BinaryArray, RecordBatch, UInt64Array, cast::AsArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use prost::Message;

use crate::betree::node::{self, InternalNode};
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
use object_store::{GetOptions, ObjectStore as OSObjectStore, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const READ_BATCH_ROWS: u32 = 16 * 1024;
const READ_BATCH_READAHEAD: u32 = 16;

/// A written node: its parent `ChildRef` (with logical byte size for tree logic)
/// plus the actual bytes written to storage (for write-amplification accounting).
pub struct Written {
    pub child_ref: pb::ChildRef,
    pub io_bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct VersionHint {
    version: u64,
}

/// Reads and writes Bε-tree node files against an object store.
pub struct NodeStore {
    object_store: Arc<ObjectStore>,
    base: Path,
    scheduler: Arc<ScanScheduler>,
    cache: Arc<LanceCache>,
}

fn leaf_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("frag_id", DataType::UInt64, false),
        ArrowField::new("fragment_pb", DataType::Binary, false),
    ]))
}

impl NodeStore {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
    ) -> Self {
        Self {
            object_store,
            base,
            scheduler,
            cache,
        }
    }

    fn leaf_path(&self) -> Path {
        self.base
            .clone()
            .join("_bt")
            .join("leaf")
            .join(format!("{}.lance", Uuid::new_v4()))
    }
    fn node_path(&self) -> Path {
        self.base
            .clone()
            .join("_bt")
            .join("node")
            .join(format!("{}.node", Uuid::new_v4()))
    }
    fn root_path(&self, version: u64) -> Path {
        self.base
            .clone()
            .join("_bt")
            .join("root")
            .join(format!("{version}.root"))
    }
    fn hint_path(&self) -> Path {
        self.base
            .clone()
            .join("_bt")
            .join("root")
            .join("latest_hint.json")
    }

    /// Write a leaf (sorted fragments) as a Lance file. Returns a leaf `ChildRef`
    /// (with logical byte size) and the actual file bytes written.
    pub async fn write_leaf(&self, fragments: &[Fragment]) -> Result<Written> {
        let arrow_schema = leaf_arrow_schema();
        let ids = UInt64Array::from_iter_values(fragments.iter().map(|f| f.id));
        let encoded: Vec<Vec<u8>> = fragments
            .iter()
            .map(|f| pb::DataFragment::from(f).encode_to_vec())
            .collect();
        let pbs = BinaryArray::from_iter_values(encoded.iter().map(|v| v.as_slice()));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(ids), Arc::new(pbs)])?;

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref())?;
        let path = self.leaf_path();
        let writer = self.object_store.create(&path).await?;
        let mut file_writer =
            FileWriter::try_new(writer, lance_schema, FileWriterOptions::default())?;
        file_writer.write_batch(&batch).await?;
        let summary = file_writer.finish().await?;

        let logical = node::leaf_logical_bytes(fragments);
        Ok(Written {
            child_ref: node::leaf_ref(path.to_string(), fragments, logical),
            io_bytes: summary.size_bytes,
        })
    }

    /// Read a leaf Lance file back into a fragment list.
    pub async fn read_leaf(&self, child: &pb::ChildRef) -> Result<Vec<Fragment>> {
        let path = Path::from(child.node_path.as_str());
        let file_scheduler = self
            .scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await?;
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
                .ok_or_else(|| Error::invalid_input("leaf file missing fragment_pb column"))?
                .as_binary::<i32>();
            for value in pbs.iter() {
                let bytes =
                    value.ok_or_else(|| Error::invalid_input("null fragment_pb in leaf file"))?;
                fragments.push(Fragment::try_from(pb::DataFragment::decode(bytes)?)?);
            }
        }
        Ok(fragments)
    }

    /// Write an internal node (children + buffer) as a protobuf object.
    pub async fn write_internal(
        &self,
        children: Vec<pb::ChildRef>,
        buffer: Vec<pb::TaggedAction>,
    ) -> Result<Written> {
        let logical = node::internal_logical_bytes(&children, &buffer);
        let path = self.node_path();
        let node = pb::InternalNode {
            children: children.clone(),
            buffer,
        };
        let bytes = node.encode_to_vec();
        let io_bytes = bytes.len() as u64;
        self.object_store
            .inner
            .put_opts(&path, PutPayload::from(bytes), PutOptions::default())
            .await?;
        Ok(Written {
            child_ref: node::internal_ref(path.to_string(), &children, logical),
            io_bytes,
        })
    }

    /// Read an internal node.
    pub async fn read_internal(&self, child: &pb::ChildRef) -> Result<InternalNode> {
        let path = Path::from(child.node_path.as_str());
        let bytes = self
            .object_store
            .inner
            .get_opts(&path, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let node = pb::InternalNode::decode(bytes)?;
        Ok(InternalNode {
            children: node.children,
            buffer: node.buffer,
        })
    }

    /// Write the root object and update the latest-version hint. Returns bytes written.
    pub async fn write_root(&self, root: &pb::BeTreeRoot) -> Result<u64> {
        let bytes = root.encode_to_vec();
        let size = bytes.len() as u64;
        self.object_store
            .inner
            .put_opts(
                &self.root_path(root.version),
                PutPayload::from(bytes),
                PutOptions::default(),
            )
            .await?;
        let hint = serde_json::to_vec(&VersionHint {
            version: root.version,
        })
        .map_err(|e| Error::invalid_input(format!("failed to encode hint: {e}")))?;
        self.object_store
            .inner
            .put_opts(
                &self.hint_path(),
                PutPayload::from(hint),
                PutOptions::default(),
            )
            .await?;
        Ok(size)
    }

    pub async fn read_latest_version(&self) -> Result<u64> {
        let bytes = self
            .object_store
            .inner
            .get_opts(&self.hint_path(), GetOptions::default())
            .await?
            .bytes()
            .await?;
        let hint: VersionHint = serde_json::from_slice(&bytes)
            .map_err(|e| Error::invalid_input(format!("failed to decode hint: {e}")))?;
        Ok(hint.version)
    }

    pub async fn read_root(&self, version: u64) -> Result<pb::BeTreeRoot> {
        let bytes = self
            .object_store
            .inner
            .get_opts(&self.root_path(version), GetOptions::default())
            .await?
            .bytes()
            .await?;
        Ok(pb::BeTreeRoot::decode(bytes)?)
    }
}
