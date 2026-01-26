// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemTable flush to persistent storage.

use std::sync::Arc;

use bytes::Bytes;
use lance_core::{Error, Result};
use lance_index::mem_wal::{FlushedGeneration, RegionManifest};
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;
use lance_io::object_store::ObjectStore;
use lance_table::format::IndexMetadata;
use log::info;
use object_store::path::Path;
use snafu::location;
use uuid::Uuid;

use super::super::index::MemIndexConfig;
use super::super::memtable::MemTable;
use crate::dataset::mem_wal::manifest::RegionManifestStore;
use crate::dataset::mem_wal::util::{flushed_memtable_path, generate_random_hash};
use crate::Dataset;

#[derive(Debug, Clone)]
pub struct FlushResult {
    pub generation: FlushedGeneration,
    pub rows_flushed: usize,
    pub covered_wal_entry_position: u64,
}

pub struct MemTableFlusher {
    object_store: Arc<ObjectStore>,
    base_path: Path,
    base_uri: String,
    region_id: Uuid,
    manifest_store: Arc<RegionManifestStore>,
}

impl MemTableFlusher {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: impl Into<String>,
        region_id: Uuid,
        manifest_store: Arc<RegionManifestStore>,
    ) -> Self {
        Self {
            object_store,
            base_path,
            base_uri: base_uri.into(),
            region_id,
            manifest_store,
        }
    }

    /// Construct a full URI for a path within the base dataset.
    fn path_to_uri(&self, path: &Path) -> String {
        // Remove base_path prefix from path to get relative path
        let path_str = path.as_ref();
        let base_str = self.base_path.as_ref();

        let relative = if let Some(stripped) = path_str.strip_prefix(base_str) {
            stripped.trim_start_matches('/')
        } else {
            path_str
        };

        // Combine base_uri with relative path
        let base = self.base_uri.trim_end_matches('/');
        if relative.is_empty() {
            base.to_string()
        } else {
            format!("{}/{}", base, relative)
        }
    }

    /// Flush the MemTable to storage (data files, indexes, bloom filter).
    pub async fn flush(&self, memtable: &MemTable, epoch: u64) -> Result<FlushResult> {
        self.manifest_store.check_fenced(epoch).await?;

        if memtable.row_count() == 0 {
            return Err(Error::invalid_input(
                "Cannot flush empty MemTable",
                location!(),
            ));
        }

        if !memtable.all_flushed_to_wal() {
            return Err(Error::invalid_input(
                "MemTable has unflushed fragments - WAL flush required first",
                location!(),
            ));
        }

        let random_hash = generate_random_hash();
        let generation = memtable.generation();
        let gen_folder_name = format!("{}_gen_{}", random_hash, generation);
        let gen_path =
            flushed_memtable_path(&self.base_path, &self.region_id, &random_hash, generation);

        info!(
            "Flushing MemTable generation {} to {} ({} rows, {} batches)",
            generation,
            gen_path,
            memtable.row_count(),
            memtable.batch_count()
        );

        self.write_data_file(&gen_path, memtable).await?;

        let bloom_path = gen_path.child("bloom_filter.bin");
        self.write_bloom_filter(&bloom_path, memtable.bloom_filter())
            .await?;

        let last_wal_entry_position = memtable.last_flushed_wal_entry_position();
        let new_manifest = self
            .update_manifest(epoch, generation, &gen_folder_name, last_wal_entry_position)
            .await?;

        info!(
            "Flushed generation {} for region {} (manifest version {})",
            generation, self.region_id, new_manifest.version
        );

        Ok(FlushResult {
            generation: FlushedGeneration {
                generation,
                path: gen_folder_name,
            },
            rows_flushed: memtable.row_count(),
            covered_wal_entry_position: last_wal_entry_position,
        })
    }

    async fn write_data_file(&self, path: &Path, memtable: &MemTable) -> Result<()> {
        use arrow_array::RecordBatchIterator;

        use crate::dataset::WriteParams;

        if memtable.row_count() == 0 {
            return Ok(());
        }

        let batches = memtable.scan_batches().await?;
        if batches.is_empty() {
            return Ok(());
        }

        let uri = self.path_to_uri(path);
        let reader =
            RecordBatchIterator::new(batches.into_iter().map(Ok), memtable.schema().clone());

        // Use very large max_rows_per_file to ensure 1 fragment per flushed memtable
        let write_params = WriteParams {
            max_rows_per_file: usize::MAX,
            ..Default::default()
        };
        Dataset::write(reader, &uri, Some(write_params)).await?;

        Ok(())
    }

    async fn write_bloom_filter(
        &self,
        path: &Path,
        bloom: &lance_index::scalar::bloomfilter::sbbf::Sbbf,
    ) -> Result<()> {
        let data = bloom.to_bytes();
        self.object_store
            .inner
            .put(path, Bytes::from(data).into())
            .await
            .map_err(|e| Error::io(format!("Failed to write bloom filter: {}", e), location!()))?;
        Ok(())
    }

    /// Flush the MemTable to storage with indexes.
    pub async fn flush_with_indexes(
        &self,
        memtable: &MemTable,
        epoch: u64,
        index_configs: &[MemIndexConfig],
    ) -> Result<FlushResult> {
        self.manifest_store.check_fenced(epoch).await?;

        if memtable.row_count() == 0 {
            return Err(Error::invalid_input(
                "Cannot flush empty MemTable",
                location!(),
            ));
        }

        if !memtable.all_flushed_to_wal() {
            return Err(Error::invalid_input(
                "MemTable has unflushed fragments - WAL flush required first",
                location!(),
            ));
        }

        let random_hash = generate_random_hash();
        let generation = memtable.generation();
        let gen_folder_name = format!("{}_gen_{}", random_hash, generation);
        let gen_path =
            flushed_memtable_path(&self.base_path, &self.region_id, &random_hash, generation);

        info!(
            "Flushing MemTable generation {} with indexes to {} ({} rows, {} batches)",
            generation,
            gen_path,
            memtable.row_count(),
            memtable.batch_count()
        );

        self.write_data_file(&gen_path, memtable).await?;

        let created_indexes = self
            .create_indexes(&gen_path, index_configs, memtable.indexes())
            .await?;
        if !created_indexes.is_empty() {
            info!(
                "Created {} BTree indexes on flushed generation {}",
                created_indexes.len(),
                generation
            );
        }

        // Create IVF-PQ indexes and commit them to the dataset
        if let Some(registry) = memtable.indexes() {
            let uri = self.path_to_uri(&gen_path);
            let mut dataset = Dataset::open(&uri).await?;

            for config in index_configs {
                if let MemIndexConfig::IvfPq(ivf_pq_config) = config {
                    if let Some(mem_index) = registry.get_ivf_pq(&ivf_pq_config.name) {
                        let mut index_meta = self
                            .create_ivf_pq_index(&gen_path, ivf_pq_config, mem_index)
                            .await?;

                        // Fix up the index metadata with correct field index
                        let schema = dataset.schema();
                        let field_idx = schema
                            .field(&ivf_pq_config.column)
                            .map(|f| f.id)
                            .unwrap_or(0);
                        index_meta.fields = vec![field_idx];
                        index_meta.dataset_version = dataset.version().version;
                        // Calculate fragment_bitmap from dataset fragments
                        let fragment_ids: roaring::RoaringBitmap = dataset
                            .get_fragments()
                            .iter()
                            .map(|f| f.id() as u32)
                            .collect();
                        index_meta.fragment_bitmap = Some(fragment_ids);

                        // Commit the index to the dataset
                        use crate::dataset::transaction::{Operation, Transaction};
                        let transaction = Transaction::new(
                            index_meta.dataset_version,
                            Operation::CreateIndex {
                                new_indices: vec![index_meta],
                                removed_indices: vec![],
                            },
                            None,
                        );
                        dataset
                            .apply_commit(transaction, &Default::default(), &Default::default())
                            .await?;

                        info!(
                            "Created IVF-PQ index '{}' on flushed generation {}",
                            ivf_pq_config.name, generation
                        );
                    }
                }
            }

            // Create FTS indexes
            self.create_fts_indexes(&uri, index_configs, &mut dataset)
                .await?;
        }

        let bloom_path = gen_path.child("bloom_filter.bin");
        self.write_bloom_filter(&bloom_path, memtable.bloom_filter())
            .await?;

        let last_wal_entry_position = memtable.last_flushed_wal_entry_position();
        let new_manifest = self
            .update_manifest(epoch, generation, &gen_folder_name, last_wal_entry_position)
            .await?;

        info!(
            "Flushed generation {} for region {} (manifest version {})",
            generation, self.region_id, new_manifest.version
        );

        Ok(FlushResult {
            generation: FlushedGeneration {
                generation,
                path: gen_folder_name,
            },
            rows_flushed: memtable.row_count(),
            covered_wal_entry_position: last_wal_entry_position,
        })
    }

    /// Create BTree indexes on the flushed dataset.
    async fn create_indexes(
        &self,
        gen_path: &Path,
        index_configs: &[MemIndexConfig],
        mem_indexes: Option<&super::super::index::IndexStore>,
    ) -> Result<Vec<IndexMetadata>> {
        use arrow_array::RecordBatchIterator;

        use crate::index::CreateIndexBuilder;

        let uri = self.path_to_uri(gen_path);

        let btree_configs: Vec<_> = index_configs
            .iter()
            .filter_map(|c| match c {
                MemIndexConfig::BTree(cfg) => Some(cfg),
                MemIndexConfig::IvfPq(_) => None,
                MemIndexConfig::Fts(_) => None,
            })
            .collect();

        if btree_configs.is_empty() {
            return Ok(vec![]);
        }

        let mut dataset = Dataset::open(&uri).await?;
        let mut created_indexes = Vec::new();

        for btree_cfg in btree_configs {
            let params = ScalarIndexParams::default();
            let mut builder = CreateIndexBuilder::new(
                &mut dataset,
                &[btree_cfg.column.as_str()],
                IndexType::BTree,
                &params,
            )
            .name(btree_cfg.name.clone());

            if let Some(registry) = mem_indexes {
                if let Some(btree_index) = registry.get_btree(&btree_cfg.name) {
                    let training_batches = btree_index.to_training_batches(8192)?;
                    if !training_batches.is_empty() {
                        let schema = training_batches[0].schema();
                        let reader =
                            RecordBatchIterator::new(training_batches.into_iter().map(Ok), schema);
                        builder = builder.preprocessed_data(Box::new(reader));
                    }
                }
            }

            let index_meta = builder.execute_uncommitted().await?;
            created_indexes.push(index_meta.clone());

            use crate::dataset::transaction::{Operation, Transaction};
            let transaction = Transaction::new(
                index_meta.dataset_version,
                Operation::CreateIndex {
                    new_indices: vec![index_meta],
                    removed_indices: vec![],
                },
                None,
            );
            dataset
                .apply_commit(transaction, &Default::default(), &Default::default())
                .await?;
        }

        Ok(created_indexes)
    }

    /// Create FTS (Full-Text Search) indexes on the flushed dataset.
    ///
    /// Uses the standard InvertedIndexBuilder with the same tokenizer parameters
    /// that were used for the in-memory FTS index.
    async fn create_fts_indexes(
        &self,
        _uri: &str,
        index_configs: &[MemIndexConfig],
        dataset: &mut Dataset,
    ) -> Result<()> {
        use crate::index::CreateIndexBuilder;

        let fts_configs: Vec<_> = index_configs
            .iter()
            .filter_map(|c| match c {
                MemIndexConfig::Fts(cfg) => Some(cfg),
                _ => None,
            })
            .collect();

        if fts_configs.is_empty() {
            return Ok(());
        }

        for fts_cfg in fts_configs {
            let mut builder = CreateIndexBuilder::new(
                dataset,
                &[fts_cfg.column.as_str()],
                IndexType::Inverted,
                &fts_cfg.params,
            )
            .name(fts_cfg.name.clone());

            let index_meta = builder.execute_uncommitted().await?;

            use crate::dataset::transaction::{Operation, Transaction};
            let transaction = Transaction::new(
                index_meta.dataset_version,
                Operation::CreateIndex {
                    new_indices: vec![index_meta.clone()],
                    removed_indices: vec![],
                },
                None,
            );
            dataset
                .apply_commit(transaction, &Default::default(), &Default::default())
                .await?;

            info!(
                "Created FTS index '{}' on column '{}'",
                fts_cfg.name, fts_cfg.column
            );
        }

        Ok(())
    }

    /// Create an IVF-PQ index from in-memory data.
    ///
    /// Writes the index files directly using the pre-computed partition assignments
    /// and PQ codes from the in-memory index.
    async fn create_ivf_pq_index(
        &self,
        gen_path: &Path,
        config: &super::super::index::IvfPqIndexConfig,
        mem_index: &super::super::index::IvfPqMemIndex,
    ) -> Result<IndexMetadata> {
        use arrow_schema::{Field, Schema as ArrowSchema};
        use lance_core::ROW_ID;
        use lance_file::writer::FileWriter;
        use lance_index::pb;
        use lance_index::vector::flat::index::FlatIndex;
        use lance_index::vector::ivf::storage::IVF_METADATA_KEY;
        use lance_index::vector::quantizer::{
            Quantization, QuantizationMetadata, QuantizerMetadata,
        };
        use lance_index::vector::storage::STORAGE_METADATA_KEY;
        use lance_index::vector::v3::subindex::IvfSubIndex;
        use lance_index::vector::{DISTANCE_TYPE_KEY, PQ_CODE_COLUMN};
        use lance_index::{
            IndexMetadata as IndexMetaSchema, INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME,
            INDEX_METADATA_SCHEMA_KEY,
        };
        use prost::Message;
        use std::sync::Arc;

        let index_uuid = uuid::Uuid::new_v4();
        let index_dir = gen_path.child("_indices").child(index_uuid.to_string());

        // Get partition data from in-memory index
        let partition_batches = mem_index.to_partition_batches()?;
        let ivf_model = mem_index.ivf_model();
        let pq = mem_index.pq();
        let distance_type = mem_index.distance_type();

        // Create storage file schema: _rowid, __pq_code
        let pq_code_len = pq.num_sub_vectors * pq.num_bits as usize / 8;
        let storage_schema: ArrowSchema = ArrowSchema::new(vec![
            Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            Field::new(
                PQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(Field::new("item", arrow_schema::DataType::UInt8, false)),
                    pq_code_len as i32,
                ),
                false,
            ),
        ]);

        // Create index file schema (FlatIndex schema)
        let index_schema: ArrowSchema = FlatIndex::schema().as_ref().clone();

        // Create file writers
        let storage_path = index_dir.child(INDEX_AUXILIARY_FILE_NAME);
        let index_path = index_dir.child(INDEX_FILE_NAME);

        let mut storage_writer = FileWriter::try_new(
            self.object_store.create(&storage_path).await?,
            (&storage_schema).try_into()?,
            Default::default(),
        )?;
        let mut index_writer = FileWriter::try_new(
            self.object_store.create(&index_path).await?,
            (&index_schema).try_into()?,
            Default::default(),
        )?;

        // Track IVF partitions for both files
        let mut storage_ivf = lance_index::vector::ivf::storage::IvfModel::empty();

        // Get centroids (required for IVF index)
        let centroids = ivf_model
            .centroids
            .clone()
            .ok_or_else(|| Error::io("IVF model has no centroids", location!()))?;
        let mut index_ivf = lance_index::vector::ivf::storage::IvfModel::new(centroids, None);
        let mut partition_index_metadata = Vec::with_capacity(ivf_model.num_partitions());

        // Create a map of partition_id -> batch for quick lookup
        let partition_map: std::collections::HashMap<usize, _> =
            partition_batches.into_iter().collect();

        // Write each partition
        for part_id in 0..ivf_model.num_partitions() {
            if let Some(batch) = partition_map.get(&part_id) {
                // Transpose PQ codes for storage (column-major layout)
                let transposed_batch = transpose_pq_batch(batch, pq_code_len)?;

                // Write storage data
                storage_writer.write_batch(&transposed_batch).await?;
                storage_ivf.add_partition(transposed_batch.num_rows() as u32);

                // FlatIndex is empty (no additional sub-index data needed for IVF-PQ)
                index_ivf.add_partition(0);
                partition_index_metadata.push(String::new());
            } else {
                // Empty partition
                storage_ivf.add_partition(0);
                index_ivf.add_partition(0);
                partition_index_metadata.push(String::new());
            }
        }

        // Write storage file metadata
        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());

        // Write PQ metadata
        let pq_metadata = pq.metadata(Some(QuantizationMetadata {
            codebook_position: Some(0),
            codebook: None,
            transposed: true,
        }));
        if let Some(extra_metadata) = pq_metadata.extra_metadata()? {
            let idx = storage_writer.add_global_buffer(extra_metadata).await?;
            let mut pq_meta = pq_metadata;
            pq_meta.set_buffer_index(idx);
            let storage_partition_metadata = vec![serde_json::to_string(&pq_meta)?];
            storage_writer.add_schema_metadata(
                STORAGE_METADATA_KEY,
                serde_json::to_string(&storage_partition_metadata)?,
            );
        }

        // Write index file metadata
        let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
        let index_metadata = IndexMetaSchema {
            index_type: "IVF_PQ".to_string(),
            distance_type: distance_type.to_string(),
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
            FlatIndex::metadata_key(),
            serde_json::to_string(&partition_index_metadata)?,
        );

        // Finish writing
        storage_writer.finish().await?;
        index_writer.finish().await?;

        // Create index metadata for commit
        // Vector indices need index_details set for retain_supported_indices() to keep them
        let index_details = Some(std::sync::Arc::new(prost_types::Any {
            type_url: "type.googleapis.com/lance.index.VectorIndexDetails".to_string(),
            value: vec![],
        }));
        let index_meta = IndexMetadata {
            uuid: index_uuid,
            name: config.name.clone(),
            fields: vec![0], // Will be updated when committing
            dataset_version: 0,
            fragment_bitmap: None,
            index_details,
            base_id: None,
            created_at: Some(chrono::Utc::now()),
            index_version: 1,
        };

        Ok(index_meta)
    }

    /// Update the region manifest with the new flushed generation.
    async fn update_manifest(
        &self,
        epoch: u64,
        generation: u64,
        gen_path: &str,
        covered_wal_entry_position: u64,
    ) -> Result<RegionManifest> {
        let gen_path = gen_path.to_string();

        self.manifest_store
            .commit_update(epoch, |current| {
                let mut flushed_generations = current.flushed_generations.clone();
                flushed_generations.push(FlushedGeneration {
                    generation,
                    path: gen_path.clone(),
                });

                RegionManifest {
                    version: current.version + 1,
                    replay_after_wal_entry_position: covered_wal_entry_position,
                    wal_entry_position_last_seen: current
                        .wal_entry_position_last_seen
                        .max(covered_wal_entry_position),
                    current_generation: generation + 1,
                    flushed_generations,
                    ..current.clone()
                }
            })
            .await
    }
}

/// Transpose PQ codes in a batch from row-major to column-major layout.
///
/// The storage format expects PQ codes to be transposed for efficient distance computation.
fn transpose_pq_batch(
    batch: &arrow_array::RecordBatch,
    pq_code_len: usize,
) -> Result<arrow_array::RecordBatch> {
    use arrow_array::cast::AsArray;
    use arrow_array::FixedSizeListArray;
    use arrow_schema::Field;
    use lance_core::ROW_ID;
    use lance_index::vector::pq::storage::transpose;
    use lance_index::vector::PQ_CODE_COLUMN;
    use std::sync::Arc;

    let row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| Error::io("Missing _rowid column in partition batch", location!()))?;

    let pq_codes = batch
        .column_by_name(PQ_CODE_COLUMN)
        .ok_or_else(|| Error::io("Missing __pq_code column in partition batch", location!()))?;

    let pq_codes_fsl = pq_codes.as_fixed_size_list();
    let codes_flat = pq_codes_fsl
        .values()
        .as_primitive::<arrow_array::types::UInt8Type>();

    // Transpose from row-major to column-major
    let transposed = transpose(codes_flat, pq_code_len, batch.num_rows());
    // Use non-nullable inner field to match the schema
    let inner_field = Arc::new(Field::new("item", arrow_schema::DataType::UInt8, false));
    let transposed_fsl = Arc::new(
        FixedSizeListArray::try_new(inner_field, pq_code_len as i32, Arc::new(transposed), None)
            .map_err(|e| {
                Error::io(
                    format!("Failed to create transposed PQ array: {}", e),
                    location!(),
                )
            })?,
    );

    arrow_array::RecordBatch::try_new(batch.schema(), vec![row_ids.clone(), transposed_fsl])
        .map_err(|e| {
            Error::io(
                format!("Failed to create transposed batch: {}", e),
                location!(),
            )
        })
}

/// Message to trigger flush of a frozen memtable to Lance storage.
pub struct TriggerMemTableFlush {
    /// The frozen memtable to flush.
    pub memtable: Arc<MemTable>,
    /// Optional channel to notify when flush completes.
    pub done: Option<tokio::sync::oneshot::Sender<Result<FlushResult>>>,
}

impl std::fmt::Debug for TriggerMemTableFlush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriggerMemTableFlush")
            .field("memtable_gen", &self.memtable.generation())
            .field("memtable_rows", &self.memtable.row_count())
            .field("has_done", &self.done.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, String, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, uri, temp_dir)
    }

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, num_rows: usize) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
                Arc::new(StringArray::from_iter_values(
                    (0..num_rows).map(|i| format!("name_{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_flusher_requires_wal_flush() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = Arc::new(RegionManifestStore::new(
            store.clone(),
            &base_path,
            region_id,
            2,
        ));

        // Claim region
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();
        memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Not flushed to WAL yet
        assert!(!memtable.all_flushed_to_wal());

        let flusher = MemTableFlusher::new(store, base_path, base_uri, region_id, manifest_store);
        let result = flusher.flush(&memtable, epoch).await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unflushed fragments"));
    }

    #[tokio::test]
    async fn test_flusher_empty_memtable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = Arc::new(RegionManifestStore::new(
            store.clone(),
            &base_path,
            region_id,
            2,
        ));

        // Claim region
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let memtable = MemTable::new(schema, 1, vec![]).unwrap();

        let flusher = MemTableFlusher::new(store, base_path, base_uri, region_id, manifest_store);
        let result = flusher.flush(&memtable, epoch).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty MemTable"));
    }

    #[tokio::test]
    async fn test_flusher_success() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = Arc::new(RegionManifestStore::new(
            store.clone(),
            &base_path,
            region_id,
            2,
        ));

        // Claim region
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();
        let frag_id = memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Simulate WAL flush
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);
        assert!(memtable.all_flushed_to_wal());

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path,
            base_uri,
            region_id,
            manifest_store.clone(),
        );
        let result = flusher.flush(&memtable, epoch).await.unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, 10);
        assert_eq!(result.covered_wal_entry_position, 1);

        // Verify manifest was updated
        let updated_manifest = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(updated_manifest.version, 2);
        assert_eq!(updated_manifest.replay_after_wal_entry_position, 1);
        assert_eq!(updated_manifest.current_generation, 2);
        assert_eq!(updated_manifest.flushed_generations.len(), 1);
    }

    #[tokio::test]
    async fn test_flusher_with_btree_index() {
        use super::super::super::index::{BTreeIndexConfig, IndexStore};
        use lance_index::DatasetIndexExt;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = Arc::new(RegionManifestStore::new(
            store.clone(),
            &base_path,
            region_id,
            2,
        ));

        // Claim region
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Create index config for the 'id' column (field_id = 0)
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_btree".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry so preprocessed data path is used
        let registry = IndexStore::from_configs(&index_configs, 100_000, 8).unwrap();
        memtable.set_indexes(registry);

        let frag_id = memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Simulate WAL flush
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            region_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs)
            .await
            .unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, 10);

        // Verify the flushed dataset has the BTree index
        // result.generation.path is just the folder name, construct full URI
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, region_id, result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "id_btree");

        // Verify query results are correct
        // The test data has ids 0-9, so querying for id = 5 should return 1 row
        let batch = dataset
            .scan()
            .filter("id = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 5);

        // Verify the query plan uses the BTree index
        let mut scan = dataset.scan();
        scan.filter("id = 5").unwrap();
        scan.prefilter(true);
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "LanceRead: ...full_filter=id = Int32(5)...
  ScalarIndexQuery: query=[id = 5]@id_btree",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_flusher_with_ivf_pq_index() {
        use super::super::super::index::{IndexStore, IvfPqIndexConfig};
        use arrow_array::{FixedSizeListArray, Float32Array};
        use lance_arrow::FixedSizeListArrayExt;
        use lance_index::vector::ivf::storage::IvfModel;
        use lance_index::vector::kmeans::{train_kmeans, KMeansParams};
        use lance_index::vector::pq::PQBuildParams;
        use lance_index::DatasetIndexExt;
        use lance_linalg::distance::DistanceType;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = Arc::new(RegionManifestStore::new(
            store.clone(),
            &base_path,
            region_id,
            2,
        ));

        // Claim region
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Create schema with vector column
        // Use 300 vectors to satisfy PQ training requirement (min 256)
        let vector_dim = 8;
        let num_vectors = 300;
        let num_partitions = 4;
        let num_sub_vectors = 2;

        let vector_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    vector_dim as i32,
                ),
                false,
            ),
        ]));

        // Generate random vectors for training and testing
        let vectors: Vec<f32> = (0..num_vectors * vector_dim)
            .map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()) * 0.5)
            .collect();
        let vectors_array = Float32Array::from(vectors);

        // Train IVF centroids using KMeans
        let kmeans_params = KMeansParams::new(None, 10, 1, DistanceType::L2);
        let kmeans = train_kmeans::<arrow_array::types::Float32Type>(
            &vectors_array,
            kmeans_params,
            vector_dim,
            num_partitions,
            num_vectors, // sample_size
        )
        .unwrap();

        // Create centroids as FixedSizeListArray
        let centroids_flat = kmeans
            .centroids
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("Centroids should be Float32Array")
            .clone();
        let centroids_fsl =
            FixedSizeListArray::try_new_from_values(centroids_flat, vector_dim as i32).unwrap();

        let ivf_model = IvfModel::new(centroids_fsl, None);

        // Train PQ codebook
        let vectors_fsl =
            FixedSizeListArray::try_new_from_values(vectors_array.clone(), vector_dim as i32)
                .unwrap();
        let pq_params = PQBuildParams::new(num_sub_vectors, 8);
        let pq = pq_params.build(&vectors_fsl, DistanceType::L2).unwrap();

        // Create index config (field_id = 1 for vector column)
        let index_configs = vec![MemIndexConfig::IvfPq(Box::new(IvfPqIndexConfig {
            name: "vector_ivf_pq".to_string(),
            field_id: 1,
            column: "vector".to_string(),
            ivf_model: ivf_model.clone(),
            pq: pq.clone(),
            distance_type: DistanceType::L2,
        }))];

        // Create MemTable with vector schema
        let mut memtable = MemTable::new(vector_schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry
        let mut registry = IndexStore::from_configs(&index_configs, 100_000, 8).unwrap();

        // Also need to add the IVF-PQ index to the registry for preprocessing
        registry.add_ivf_pq(
            "vector_ivf_pq".to_string(),
            1, // field_id for vector column
            "vector".to_string(),
            ivf_model,
            pq,
            DistanceType::L2,
        );
        memtable.set_indexes(registry);

        // Create test batch with vectors
        let ids = Int32Array::from_iter_values(0..num_vectors as i32);
        // Use the field from the schema to ensure nullability matches
        let inner_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vectors_fsl_data = FixedSizeListArray::try_new(
            inner_field,
            vector_dim as i32,
            Arc::new(vectors_array),
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            vector_schema.clone(),
            vec![Arc::new(ids), Arc::new(vectors_fsl_data)],
        )
        .unwrap();

        let frag_id = memtable.insert(batch).await.unwrap();

        // Simulate WAL flush
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            region_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs)
            .await
            .unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, num_vectors);

        // Verify the flushed dataset has the IVF-PQ index
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, region_id, result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vector_ivf_pq");

        // Create a query vector (use first vector from the dataset)
        let query_vector: Vec<f32> = (0..vector_dim)
            .map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()) * 0.5)
            .collect();
        let query_array = Float32Array::from(query_vector);

        // Verify ANN query returns correct results
        let batch = dataset
            .scan()
            .nearest("vector", &query_array, 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        // Should return 10 nearest neighbors
        assert_eq!(batch.num_rows(), 10);

        // Verify distances are non-negative and sorted in ascending order (nearest first)
        let distance_col = batch
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!(
            distance_col.value(0) >= 0.0,
            "First distance should be non-negative"
        );
        for i in 1..10 {
            assert!(
                distance_col.value(i - 1) <= distance_col.value(i),
                "Distances should be sorted: {} > {}",
                distance_col.value(i - 1),
                distance_col.value(i)
            );
        }

        // Verify returned IDs are valid (within range 0..num_vectors)
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..10 {
            let id = id_col.value(i);
            assert!(
                id >= 0 && id < num_vectors as i32,
                "ID {} should be in range [0, {})",
                id,
                num_vectors
            );
        }

        // Verify the query plan uses the IVF-PQ index
        let mut scan = dataset.scan();
        scan.nearest("vector", &query_array, 10).unwrap();
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "ProjectionExec: expr=[id@2 as id, vector@3 as vector, _distance@0 as _distance]
  Take: ...
    CoalesceBatchesExec: ...
      SortExec: TopK...
        ANNSubIndex: name=vector_ivf_pq, k=10, deltas=1, metric=L2
          ANNIvfPartition: ...deltas=1",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_flusher_with_fts_index() {
        use super::super::super::index::{FtsIndexConfig, IndexStore};
        use arrow_array::StringArray;
        use arrow_schema::{DataType, Field, Schema as ArrowSchema};
        use lance_index::DatasetIndexExt;
        use std::sync::Arc;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = Arc::new(RegionManifestStore::new(
            store.clone(),
            &base_path,
            region_id,
            2,
        ));

        // Claim region
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Create schema with text column
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));

        // Create FTS index config (field_id = 1 for text column)
        let index_configs = vec![MemIndexConfig::Fts(FtsIndexConfig::new(
            "text_fts".to_string(),
            1,
            "text".to_string(),
        ))];

        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry
        let registry = IndexStore::from_configs(&index_configs, 100_000, 8).unwrap();
        memtable.set_indexes(registry);

        // Create test batch with text data
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    "hello world",
                    "quick brown fox",
                    "lazy dog jumps",
                ])),
            ],
        )
        .unwrap();

        let frag_id = memtable.insert(batch).await.unwrap();

        // Simulate WAL flush
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            region_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs)
            .await
            .unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, 3);

        // Verify the flushed dataset has the FTS index
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, region_id, result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "text_fts");

        // Verify FTS query returns correct results
        // Searching for "hello" should find the first document
        use lance_index::scalar::FullTextSearchQuery;
        let batch = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(
            id_col.value(0),
            1,
            "Should find document with 'hello world'"
        );

        // Searching for "fox" should find the second document
        let batch = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("fox".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(
            id_col.value(0),
            2,
            "Should find document with 'quick brown fox'"
        );

        // Verify the query plan uses the FTS index
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            .unwrap();
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "ProjectionExec: expr=[id@2 as id, text@3 as text, _score@1 as _score]
  Take: ...
    CoalesceBatchesExec: ...
      MatchQuery: column=text, query=hello",
        )
        .await
        .unwrap();
    }
}
