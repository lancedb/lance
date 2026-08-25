// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): copy-on-write node IO.
//!
//! Nodes are immutable object-store files; every rewrite produces a new file
//! (uuid-named), so flush/split/merge just write new files and the parent
//! repoints. Internal nodes and the root are protobuf objects.
//!
//! Leaves are tabular Lance v2 files with **one row per data file** and each
//! `DataFile` field in its own column (path, versions, size, base_id, field
//! ids). Decomposing the fragment into per-field columns — rather than one
//! opaque `DataFragment` protobuf blob per row — lets Lance compress each column
//! independently (identical file versions RLE to ~nothing, sizes cluster, paths
//! dictionary/FSST-encode), which is the columnar win @Xuanwo measured. A
//! fragment's `id`/`physical_rows` are repeated on each of its rows (RLE-cheap).
//!
//! Limitation (prototype): a fragment must have ≥1 data file to round-trip (it
//! has no row otherwise), and overlays / deletion / row-id metadata are not
//! persisted here — neither occurs in the benchmark workload.

use std::num::NonZero;
use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt32Type, UInt64Type};
use arrow_array::{Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use prost::Message;

use crate::betree::node::{self, InternalNode};
use crate::format::pb;
use crate::format::{DataFile, Fragment};
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

fn int_list_type() -> DataType {
    DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true)))
}

/// Columnar leaf schema: one row per data file, each `DataFile` field its own column.
fn leaf_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("frag_id", DataType::UInt64, false),
        ArrowField::new("physical_rows", DataType::UInt64, false), // 0 = unknown
        ArrowField::new("path", DataType::Utf8, false),
        ArrowField::new("field_ids", int_list_type(), false),
        ArrowField::new("column_indices", int_list_type(), false),
        ArrowField::new("major_version", DataType::UInt32, false),
        ArrowField::new("minor_version", DataType::UInt32, false),
        ArrowField::new("file_size_bytes", DataType::UInt64, false), // 0 = unknown
        ArrowField::new("base_id", DataType::UInt32, true),          // null = None
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

    /// Write a leaf (sorted fragments) as a columnar Lance file: one row per data
    /// file. Returns a leaf `ChildRef` (logical byte size) + actual bytes written.
    pub async fn write_leaf(&self, fragments: &[Fragment]) -> Result<Written> {
        let num_files: usize = fragments.iter().map(|f| f.files.len()).sum();
        let mut frag_ids = Vec::with_capacity(num_files);
        let mut physical = Vec::with_capacity(num_files);
        let mut paths = Vec::with_capacity(num_files);
        let mut major = Vec::with_capacity(num_files);
        let mut minor = Vec::with_capacity(num_files);
        let mut sizes = Vec::with_capacity(num_files);
        let mut base_ids: Vec<Option<u32>> = Vec::with_capacity(num_files);
        let mut field_builder = ListBuilder::new(Int32Builder::new());
        let mut col_builder = ListBuilder::new(Int32Builder::new());

        for f in fragments {
            let pr = f.physical_rows.unwrap_or(0) as u64;
            for df in &f.files {
                frag_ids.push(f.id);
                physical.push(pr);
                paths.push(df.path.clone());
                major.push(df.file_major_version);
                minor.push(df.file_minor_version);
                sizes.push(df.file_size_bytes.get().map(|n| n.get()).unwrap_or(0));
                base_ids.push(df.base_id);
                for &v in df.fields.iter() {
                    field_builder.values().append_value(v);
                }
                field_builder.append(true);
                for &v in df.column_indices.iter() {
                    col_builder.values().append_value(v);
                }
                col_builder.append(true);
            }
        }

        let arrow_schema = leaf_arrow_schema();
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(frag_ids)),
                Arc::new(UInt64Array::from(physical)),
                Arc::new(StringArray::from(paths)),
                Arc::new(field_builder.finish()),
                Arc::new(col_builder.finish()),
                Arc::new(UInt32Array::from(major)),
                Arc::new(UInt32Array::from(minor)),
                Arc::new(UInt64Array::from(sizes)),
                Arc::new(UInt32Array::from(base_ids)),
            ],
        )?;

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

    /// Read a columnar leaf back into a fragment list (rows grouped by frag_id).
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

        let mut fragments: Vec<Fragment> = Vec::with_capacity(child.num_keys as usize);
        let mut stream = reader
            .read_stream(
                ReadBatchParams::RangeFull,
                READ_BATCH_ROWS,
                READ_BATCH_READAHEAD,
                FilterExpression::no_filter(),
            )
            .await?;
        while let Some(batch) = stream.try_next().await? {
            let col = |name: &str| {
                batch
                    .column_by_name(name)
                    .ok_or_else(|| Error::invalid_input(format!("leaf missing column {name}")))
            };
            let frag_ids = col("frag_id")?.as_primitive::<UInt64Type>();
            let physical = col("physical_rows")?.as_primitive::<UInt64Type>();
            let paths = col("path")?.as_string::<i32>();
            let field_ids = col("field_ids")?.as_list::<i32>();
            let col_indices = col("column_indices")?.as_list::<i32>();
            let major = col("major_version")?.as_primitive::<UInt32Type>();
            let minor = col("minor_version")?.as_primitive::<UInt32Type>();
            let sizes = col("file_size_bytes")?.as_primitive::<UInt64Type>();
            let base_ids = col("base_id")?.as_primitive::<UInt32Type>();

            for row in 0..batch.num_rows() {
                let fid = frag_ids.value(row);
                if fragments.last().map(|f| f.id) != Some(fid) {
                    let mut frag = Fragment::new(fid);
                    let pr = physical.value(row);
                    frag.physical_rows = (pr != 0).then_some(pr as usize);
                    fragments.push(frag);
                }
                let fields = field_ids
                    .value(row)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec();
                let cols = col_indices
                    .value(row)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec();
                let base = (!base_ids.is_null(row)).then(|| base_ids.value(row));
                let df = DataFile::new(
                    paths.value(row),
                    fields,
                    cols,
                    major.value(row),
                    minor.value(row),
                    NonZero::new(sizes.value(row)),
                    base,
                );
                fragments.last_mut().unwrap().files.push(df);
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
