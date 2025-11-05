// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary Index
//!

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use arrow_schema::{DataType, Schema};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::cache::{CacheKey, UnsizedCacheKey};
use lance_core::utils::address::RowAddress;
use lance_core::utils::parse::str_is_truthy;
use lance_core::utils::tracing::{
    IO_TYPE_OPEN_FRAG_REUSE, IO_TYPE_OPEN_MEM_WAL, IO_TYPE_OPEN_SCALAR, IO_TYPE_OPEN_VECTOR,
    TRACE_IO_EVENTS,
};
use lance_file::reader::FileReader;
use lance_file::v2;
use lance_file::v2::reader::FileReaderOptions;
use lance_index::frag_reuse::{FragReuseIndex, FRAG_REUSE_INDEX_NAME};
use lance_index::mem_wal::{MemWalIndex, MEM_WAL_INDEX_NAME};
use lance_index::optimize::OptimizeOptions;
use lance_index::pb::index::Implementation;
use lance_index::scalar::expression::{
    IndexInformationProvider, MultiQueryParser, ScalarQueryParser,
};
use lance_index::scalar::inverted::InvertedIndexPlugin;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{TrainingCriteria, TrainingOrdering};
use lance_index::scalar::{CreatedIndex, ScalarIndex};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::sq::ScalarQuantizer;
pub use lance_index::IndexParams;
use lance_index::{
    is_system_index,
    metrics::{MetricsCollector, NoOpMetricsCollector},
    ScalarIndexCriteria,
};
use lance_index::{pb, vector::VectorIndex, Index, IndexType, INDEX_FILE_NAME};
use lance_index::{DatasetIndexExt, INDEX_METADATA_SCHEMA_KEY, VECTOR_INDEX_VERSION};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Reader;
use lance_io::utils::{
    read_last_block, read_message, read_message_from_buf, read_metadata_offset, read_version,
    CachedFileSize,
};
use lance_table::format::IndexMetadata;
use lance_table::format::{Fragment, SelfDescribingFileReader};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringBitmap;
use scalar::index_matches_criteria;
use serde_json::json;
use snafu::location;
use tracing::{info, instrument};
use uuid::Uuid;
use vector::ivf::v2::IVFIndex;
use vector::utils::get_vector_type;

pub(crate) mod append;
mod create;
pub mod frag_reuse;
pub mod mem_wal;
pub mod prefilter;
pub mod scalar;
pub mod vector;

use self::append::merge_indices;
use self::vector::remap_vector_index;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::optimize::remapping::RemapResult;
use crate::dataset::optimize::RemappedIndex;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
use crate::index::mem_wal::open_mem_wal_index;
pub use crate::index::prefilter::{FilterLoader, PreFilter};
use crate::index::scalar::{fetch_index_details, load_training_data, IndexDetails};
use crate::session::index_caches::{FragReuseIndexKey, IndexMetadataKey};
use crate::{dataset::Dataset, Error, Result};
pub use create::CreateIndexBuilder;

// Cache keys for different index types
#[derive(Debug, Clone)]
pub struct ScalarIndexCacheKey<'a> {
    pub uuid: &'a str,
    pub fri_uuid: Option<&'a Uuid>,
}

impl<'a> ScalarIndexCacheKey<'a> {
    pub fn new(uuid: &'a str, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl UnsizedCacheKey for ScalarIndexCacheKey<'_> {
    type ValueType = dyn ScalarIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.into()
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorIndexCacheKey<'a> {
    pub uuid: &'a str,
    pub fri_uuid: Option<&'a Uuid>,
}

impl<'a> VectorIndexCacheKey<'a> {
    pub fn new(uuid: &'a str, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl UnsizedCacheKey for VectorIndexCacheKey<'_> {
    type ValueType = dyn VectorIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.into()
        }
    }
}

#[derive(Debug, Clone)]
pub struct FragReuseIndexCacheKey<'a> {
    pub uuid: &'a str,
    pub fri_uuid: Option<&'a Uuid>,
}

impl<'a> FragReuseIndexCacheKey<'a> {
    pub fn new(uuid: &'a str, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for FragReuseIndexCacheKey<'_> {
    type ValueType = FragReuseIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.into()
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemWalCacheKey<'a> {
    pub uuid: &'a Uuid,
    pub fri_uuid: Option<&'a Uuid>,
}

impl<'a> MemWalCacheKey<'a> {
    pub fn new(uuid: &'a Uuid, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for MemWalCacheKey<'_> {
    type ValueType = MemWalIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("{}-{}", self.uuid, fri_uuid).into()
        } else {
            self.uuid.to_string().into()
        }
    }
}

// Whether to auto-migrate a dataset when we encounter corruption.
fn auto_migrate_corruption() -> bool {
    static LANCE_AUTO_MIGRATION: OnceLock<bool> = OnceLock::new();
    *LANCE_AUTO_MIGRATION.get_or_init(|| {
        std::env::var("LANCE_AUTO_MIGRATION")
            .ok()
            .map(|s| str_is_truthy(&s))
            .unwrap_or(true)
    })
}

/// Builds index.
#[async_trait]
pub trait IndexBuilder {
    fn index_type() -> IndexType;

    async fn build(&self) -> Result<()>;
}

pub(crate) async fn remap_index(
    dataset: &Dataset,
    index_id: &Uuid,
    row_id_map: &HashMap<u64, Option<u64>>,
) -> Result<RemapResult> {
    // Load indices from the disk.
    let indices = dataset.load_indices().await?;
    let matched = indices
        .iter()
        .find(|i| i.uuid == *index_id)
        .ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", index_id),
            location: location!(),
        })?;

    if matched.fields.len() > 1 {
        return Err(Error::Index {
            message: "Remapping indices with multiple fields is not supported".to_string(),
            location: location!(),
        });
    }

    if row_id_map.values().all(|v| v.is_none()) {
        let deleted_bitmap = RoaringBitmap::from_iter(
            row_id_map
                .keys()
                .map(|row_id| RowAddress::new_from_u64(*row_id))
                .map(|addr| addr.fragment_id()),
        );
        if Some(deleted_bitmap) == matched.fragment_bitmap {
            // If remap deleted all rows, we can just return the same index ID.
            // This can happen if there is a bug where the index is covering empty
            // fragment that haven't been cleaned up. They should be cleaned up
            // outside of this function.
            return Ok(RemapResult::Keep(*index_id));
        }
    }

    let field_id = matched
        .fields
        .first()
        .expect("An index existed with no fields");
    let field_path = dataset.schema().field_path(*field_id)?;

    let new_id = Uuid::new_v4();

    let generic = dataset
        .open_generic_index(&field_path, &index_id.to_string(), &NoOpMetricsCollector)
        .await?;

    let created_index = match generic.index_type() {
        it if it.is_scalar() => {
            let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_id.to_string())?;

            let scalar_index = dataset
                .open_scalar_index(&field_path, &index_id.to_string(), &NoOpMetricsCollector)
                .await?;
            if !scalar_index.can_remap() {
                return Ok(RemapResult::Drop);
            }

            match scalar_index.index_type() {
                IndexType::Inverted => {
                    let inverted_index = scalar_index
                        .as_any()
                        .downcast_ref::<lance_index::scalar::inverted::InvertedIndex>()
                        .ok_or(Error::Index {
                            message: "expected inverted index".to_string(),
                            location: location!(),
                        })?;
                    if inverted_index.is_legacy() {
                        log::warn!("reindex because of legacy format, index_type: {}, index_id: {}, field: {}",
                            scalar_index.index_type(),
                            index_id,
                            field_path
                        );
                        let training_data = load_training_data(
                            dataset,
                            &field_path,
                            &TrainingCriteria::new(TrainingOrdering::None),
                            None,
                            true, // Legacy reindexing should always train
                            None,
                        )
                        .await?;
                        InvertedIndexPlugin::train_inverted_index(
                            training_data,
                            &new_store,
                            inverted_index.params().clone(),
                            None,
                        )
                        .await?
                    } else {
                        scalar_index.remap(row_id_map, &new_store).await?
                    }
                }
                _ => scalar_index.remap(row_id_map, &new_store).await?,
            }
        }
        it if it.is_vector() => {
            remap_vector_index(
                Arc::new(dataset.clone()),
                &field_path,
                index_id,
                &new_id,
                matched,
                row_id_map,
            )
            .await?;
            CreatedIndex {
                index_details: prost_types::Any::from_msg(
                    &lance_table::format::pb::VectorIndexDetails::default(),
                )
                .unwrap(),
                index_version: VECTOR_INDEX_VERSION,
            }
        }
        _ => {
            return Err(Error::Index {
                message: format!("Index type {} is not supported", generic.index_type()),
                location: location!(),
            });
        }
    };

    Ok(RemapResult::Remapped(RemappedIndex {
        old_id: *index_id,
        new_id,
        index_details: created_index.index_details,
        index_version: created_index.index_version,
    }))
}

#[derive(Debug)]
pub struct ScalarIndexInfo {
    indexed_columns: HashMap<String, (DataType, Box<MultiQueryParser>)>,
}

impl IndexInformationProvider for ScalarIndexInfo {
    fn get_index(&self, col: &str) -> Option<(&DataType, &dyn ScalarQueryParser)> {
        self.indexed_columns
            .get(col)
            .map(|(ty, parser)| (ty, parser.as_ref() as &dyn ScalarQueryParser))
    }
}

async fn open_index_proto(reader: &dyn Reader) -> Result<pb::Index> {
    let file_size = reader.size().await?;
    let tail_bytes = read_last_block(reader).await?;
    let metadata_pos = read_metadata_offset(&tail_bytes)?;
    let proto: pb::Index = if metadata_pos < file_size - tail_bytes.len() {
        // We have not read the metadata bytes yet.
        read_message(reader, metadata_pos).await?
    } else {
        let offset = tail_bytes.len() - (file_size - metadata_pos);
        read_message_from_buf(&tail_bytes.slice(offset..))?
    };
    Ok(proto)
}

fn vector_index_details() -> prost_types::Any {
    let details = lance_table::format::pb::VectorIndexDetails::default();
    prost_types::Any::from_msg(&details).unwrap()
}

#[async_trait]
impl DatasetIndexExt for Dataset {
    type IndexBuilder<'a> = CreateIndexBuilder<'a>;

    /// Create a builder for creating an index on columns.
    ///
    /// This returns a builder that can be configured with additional options
    /// before awaiting to execute.
    ///
    /// # Examples
    ///
    /// Create a scalar BTREE index:
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance_index::{DatasetIndexExt, IndexType, scalar::ScalarIndexParams};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let params = ScalarIndexParams::default();
    /// dataset
    ///     .create_index_builder(&["id"], IndexType::BTree, &params)
    ///     .name("id_index".to_string())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Create an empty index that will be populated later:
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance_index::{DatasetIndexExt, IndexType, scalar::ScalarIndexParams};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let params = ScalarIndexParams::default();
    /// dataset
    ///     .create_index_builder(&["category"], IndexType::Bitmap, &params)
    ///     .train(false)  // Create empty index
    ///     .replace(true)  // Replace if exists
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn create_index_builder<'a>(
        &'a mut self,
        columns: &[&str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> CreateIndexBuilder<'a> {
        CreateIndexBuilder::new(self, columns, index_type, params)
    }

    #[instrument(skip_all)]
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<()> {
        // Use the builder pattern with default train=true for backward compatibility
        let mut builder = self.create_index_builder(columns, index_type, params);

        if let Some(name) = name {
            builder = builder.name(name);
        }

        builder.replace(replace).await
    }

    async fn drop_index(&mut self, name: &str) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::IndexNotFound {
                identity: format!("name={}", name),
                location: location!(),
            });
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: indices.clone(),
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn prewarm_index(&self, name: &str) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::IndexNotFound {
                identity: format!("name={}", name),
                location: location!(),
            });
        }

        let index = self
            .open_generic_index(name, &indices[0].uuid.to_string(), &NoOpMetricsCollector)
            .await?;
        index.prewarm().await?;

        Ok(())
    }

    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let metadata_key = IndexMetadataKey {
            version: self.version().version,
        };
        let indices = match self.index_cache.get_with_key(&metadata_key).await {
            Some(indices) => indices,
            None => {
                let mut loaded_indices = read_manifest_indexes(
                    &self.object_store,
                    &self.manifest_location,
                    &self.manifest,
                )
                .await?;
                retain_supported_indices(&mut loaded_indices);
                let loaded_indices = Arc::new(loaded_indices);
                self.index_cache
                    .insert_with_key(&metadata_key, loaded_indices.clone())
                    .await;
                loaded_indices
            }
        };

        if let Some(frag_reuse_index_meta) =
            indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        {
            let uuid = frag_reuse_index_meta.uuid.to_string();
            let fri_key = FragReuseIndexKey { uuid: &uuid };
            let frag_reuse_index = self
                .index_cache
                .get_or_insert_with_key(fri_key, || async move {
                    let index_details =
                        load_frag_reuse_index_details(self, frag_reuse_index_meta).await?;
                    open_frag_reuse_index(frag_reuse_index_meta.uuid, index_details.as_ref()).await
                })
                .await?;
            let mut indices = indices.as_ref().clone();
            indices.iter_mut().for_each(|idx| {
                if let Some(bitmap) = idx.fragment_bitmap.as_mut() {
                    frag_reuse_index.remap_fragment_bitmap(bitmap).unwrap();
                }
            });
            Ok(Arc::new(indices))
        } else {
            Ok(indices)
        }
    }

    async fn commit_existing_index(
        &mut self,
        index_name: &str,
        column: &str,
        index_id: Uuid,
    ) -> Result<()> {
        let Some(field) = self.schema().field(column) else {
            return Err(Error::Index {
                message: format!("CreateIndex: column '{column}' does not exist"),
                location: location!(),
            });
        };

        // TODO: We will need some way to determine the index details here.  Perhaps
        // we can load the index itself and get the details that way.

        let new_idx = IndexMetadata {
            uuid: index_id,
            name: index_name.to_string(),
            fields: vec![field.id],
            dataset_version: self.manifest.version,
            fragment_bitmap: Some(self.get_fragments().iter().map(|f| f.id() as u32).collect()),
            index_details: None,
            index_version: 0,
            created_at: Some(chrono::Utc::now()),
            base_id: None, // New indices don't have base_id (they're not from shallow clone)
        };

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: ScalarIndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>> {
        let indices = self.load_indices().await?;

        let mut indices = indices
            .iter()
            .filter(|idx| {
                // We shouldn't have any indices with empty fields, but just in case, log an error
                // but don't fail the operation (we might not be using that index)
                if idx.fields.is_empty() {
                    if idx.name != FRAG_REUSE_INDEX_NAME {
                        log::error!("Index {} has no fields", idx.name);
                    }
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        // This sorting & chunking is only needed to provide some backwards compatibility behavior for
        // old versions of Lance that don't write index details.
        //
        // TODO: At some point we should just fail if the index details are missing and ask the user to
        // retrain the index.
        indices.sort_by_key(|idx| idx.fields[0]);
        let indice_by_field = indices.into_iter().chunk_by(|idx| idx.fields[0]);
        for (field_id, indices) in &indice_by_field {
            let indices = indices.collect::<Vec<_>>();
            let has_multiple = indices.len() > 1;
            for idx in indices {
                let field = self.schema().field_by_id(field_id);
                if let Some(field) = field {
                    if index_matches_criteria(idx, &criteria, field, has_multiple, self.schema())? {
                        let non_empty = idx.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                            bitmap.intersection_len(self.fragment_bitmap.as_ref()) > 0
                        });
                        let is_fts_index = if let Some(details) = &idx.index_details {
                            IndexDetails(details.clone()).supports_fts()
                        } else {
                            false
                        };
                        // FTS indices must always be returned even if empty, because FTS queries
                        // require an index to exist. The query execution will handle the empty
                        // bitmap appropriately and fall back to scanning unindexed data.
                        // Other index types can be skipped if empty since they're optional optimizations.
                        if non_empty || is_fts_index {
                            return Ok(Some(idx.clone()));
                        }
                    }
                }
            }
        }
        return Ok(None);
    }

    #[instrument(skip_all)]
    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()> {
        let dataset = Arc::new(self.clone());
        let indices = self.load_indices().await?;

        let indices_to_optimize = options
            .index_names
            .as_ref()
            .map(|names| names.iter().collect::<HashSet<_>>());
        let name_to_indices = indices
            .iter()
            .filter(|idx| {
                indices_to_optimize
                    .as_ref()
                    .is_none_or(|names| names.contains(&idx.name))
                    && !is_system_index(idx)
            })
            .map(|idx| (idx.name.clone(), idx))
            .into_group_map();

        let mut new_indices = vec![];
        let mut removed_indices = vec![];
        for deltas in name_to_indices.values() {
            let Some(res) = merge_indices(dataset.clone(), deltas.as_slice(), options).await?
            else {
                continue;
            };

            let last_idx = deltas.last().expect("Delta indices should not be empty");
            let new_idx = IndexMetadata {
                uuid: res.new_uuid,
                name: last_idx.name.clone(), // Keep the same name
                fields: last_idx.fields.clone(),
                dataset_version: self.manifest.version,
                fragment_bitmap: Some(res.new_fragment_bitmap),
                index_details: Some(Arc::new(res.new_index_details)),
                index_version: res.new_index_version,
                created_at: Some(chrono::Utc::now()),
                base_id: None, // Mew merged index file locates in the cloned dataset.
            };
            removed_indices.extend(res.removed_indices.iter().map(|&idx| idx.clone()));
            if deltas.len() > res.removed_indices.len() {
                new_indices.extend(
                    deltas[0..(deltas.len() - res.removed_indices.len())]
                        .iter()
                        .map(|&idx| idx.clone()),
                );
            }
            new_indices.push(new_idx);
        }

        if new_indices.is_empty() {
            return Ok(());
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn index_statistics(&self, index_name: &str) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.is_empty() {
            return Err(Error::IndexNotFound {
                identity: format!("name={}", index_name),
                location: location!(),
            });
        }

        if index_name == FRAG_REUSE_INDEX_NAME {
            let index = self
                .open_frag_reuse_index(&NoOpMetricsCollector)
                .await?
                .expect("FragmentReuse index does not exist");
            return serde_json::to_string(&index.statistics()?).map_err(|e| Error::Index {
                message: format!("Failed to serialize index statistics: {}", e),
                location: location!(),
            });
        }

        if index_name == MEM_WAL_INDEX_NAME {
            let index = self
                .open_mem_wal_index(&NoOpMetricsCollector)
                .await?
                .expect("MemWal index does not exist");
            return serde_json::to_string(&index.statistics()?).map_err(|e| Error::Index {
                message: format!("Failed to serialize index statistics: {}", e),
                location: location!(),
            });
        }

        let field_id = metadatas[0].fields[0];
        let field_path = self.schema().field_path(field_id)?;

        // Open all delta indices
        let indices = stream::iter(metadatas.iter())
            .then(|m| {
                let field_path = field_path.clone();
                async move {
                    self.open_generic_index(&field_path, &m.uuid.to_string(), &NoOpMetricsCollector)
                        .await
                }
            })
            .try_collect::<Vec<_>>()
            .await?;

        // Stastistics for each delta index.
        let indices_stats = indices
            .iter()
            .map(|idx| idx.statistics())
            .collect::<Result<Vec<_>>>()?;

        let index_type = indices[0].index_type().to_string();

        let indexed_fragments_per_delta = self.indexed_fragments(index_name).await?;

        let res = indexed_fragments_per_delta
            .iter()
            .map(|frags| {
                let mut sum = 0;
                for frag in frags.iter() {
                    sum += frag.num_rows().ok_or_else(|| Error::Internal {
                        message: "Fragment should have row counts, please upgrade lance and \
                                      trigger a single write to fix this"
                            .to_string(),
                        location: location!(),
                    })?;
                }
                Ok(sum)
            })
            .collect::<Result<Vec<_>>>();

        async fn migrate_and_recompute(ds: &Dataset, index_name: &str) -> Result<String> {
            let mut ds = ds.clone();
            log::warn!(
                "Detecting out-dated fragment metadata, migrating dataset. \
                        To disable migration, set LANCE_AUTO_MIGRATION=false"
            );
            ds.delete("false").await.map_err(|err| {
                Error::Execution {
                    message: format!("Failed to migrate dataset while calculating index statistics. \
                            To disable migration, set LANCE_AUTO_MIGRATION=false. Original error: {}", err),
                    location: location!(),
                }
            })?;
            ds.index_statistics(index_name).await
        }

        let num_indexed_rows_per_delta = match res {
            Ok(rows) => rows,
            Err(Error::Internal { message, .. })
                if auto_migrate_corruption() && message.contains("trigger a single write") =>
            {
                return migrate_and_recompute(self, index_name).await;
            }
            Err(e) => return Err(e),
        };

        let mut fragment_ids = HashSet::new();
        for frags in indexed_fragments_per_delta.iter() {
            for frag in frags.iter() {
                if !fragment_ids.insert(frag.id) {
                    if auto_migrate_corruption() {
                        return migrate_and_recompute(self, index_name).await;
                    } else {
                        return Err(Error::Internal {
                            message:
                                "Overlap in indexed fragments. Please upgrade to lance >= 0.23.0 \
                                  and trigger a single write to fix this"
                                    .to_string(),
                            location: location!(),
                        });
                    }
                }
            }
        }
        let num_indexed_fragments = fragment_ids.len();

        let num_unindexed_fragments = self.fragments().len() - num_indexed_fragments;
        let num_indexed_rows: usize = num_indexed_rows_per_delta.iter().cloned().sum();
        let num_unindexed_rows = self.count_rows(None).await? - num_indexed_rows;

        // Calculate updated_at as max(created_at) from all index metadata
        let updated_at = metadatas
            .iter()
            .filter_map(|m| m.created_at)
            .max()
            .map(|dt| dt.timestamp_millis() as u64);

        let stats = json!({
            "index_type": index_type,
            "name": index_name,
            "num_indices": metadatas.len(),
            "indices": indices_stats,
            "num_indexed_fragments": num_indexed_fragments,
            "num_indexed_rows": num_indexed_rows,
            "num_unindexed_fragments": num_unindexed_fragments,
            "num_unindexed_rows": num_unindexed_rows,
            "num_indexed_rows_per_delta": num_indexed_rows_per_delta,
            "updated_at_timestamp_ms": updated_at,
        });

        serde_json::to_string(&stats).map_err(|e| Error::Index {
            message: format!("Failed to serialize index statistics: {}", e),
            location: location!(),
        })
    }

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream> {
        let indices = self.load_indices_by_name(index_name).await?;
        if indices.is_empty() {
            return Err(Error::IndexNotFound {
                identity: format!("name={}", index_name),
                location: location!(),
            });
        }
        let column = self.schema().field_by_id(indices[0].fields[0]).unwrap();

        let mut schema: Option<Arc<Schema>> = None;
        let mut partition_streams = Vec::with_capacity(indices.len());
        for index in indices {
            let index = self
                .open_vector_index(&column.name, &index.uuid.to_string(), &NoOpMetricsCollector)
                .await?;

            let stream = index
                .partition_reader(partition_id, with_vector, &NoOpMetricsCollector)
                .await?;
            if schema.is_none() {
                schema = Some(stream.schema());
            }
            partition_streams.push(stream);
        }

        match schema {
            Some(schema) => {
                let merged = stream::select_all(partition_streams);
                let stream = RecordBatchStreamAdapter::new(schema, merged);
                Ok(Box::pin(stream))
            }
            None => Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::new(Schema::empty()),
                stream::empty(),
            ))),
        }
    }
}

pub(crate) fn retain_supported_indices(indices: &mut Vec<IndexMetadata>) {
    indices.retain(|idx| {
        let max_supported_version = idx
            .index_details
            .as_ref()
            .map(|details| {
                IndexDetails(details.clone())
                    .index_version()
                    // If we don't know how to read the index, it isn't supported
                    .unwrap_or(i32::MAX as u32)
            })
            .unwrap_or_default();
        let is_valid = idx.index_version <= max_supported_version as i32;
        if !is_valid {
            log::warn!(
                "Index {} has version {}, which is not supported (<={}), ignoring it",
                idx.name,
                idx.index_version,
                max_supported_version,
            );
        }
        is_valid
    })
}

/// A trait for internal dataset utilities
///
/// Internal use only. No API stability guarantees.
#[async_trait]
pub trait DatasetIndexInternalExt: DatasetIndexExt {
    /// Opens an index (scalar or vector) as a generic index
    async fn open_generic_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>>;
    /// Opens the requested scalar index
    async fn open_scalar_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>>;
    /// Opens the requested vector index
    async fn open_vector_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>>;

    /// Opens the fragment reuse index
    async fn open_frag_reuse_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<FragReuseIndex>>>;

    /// Opens the MemWAL index
    async fn open_mem_wal_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<MemWalIndex>>>;

    /// Gets the fragment reuse index UUID from the current manifest, if it exists
    async fn frag_reuse_index_uuid(&self) -> Option<Uuid>;

    /// Loads information about all the available scalar indices on the dataset
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo>;

    /// Return the fragments that are not covered by any of the deltas of the index.
    async fn unindexed_fragments(&self, idx_name: &str) -> Result<Vec<Fragment>>;

    /// Return the fragments that are covered by each of the deltas of the index.
    async fn indexed_fragments(&self, idx_name: &str) -> Result<Vec<Vec<Fragment>>>;

    /// Initialize a specific index on this dataset based on an index from a source dataset.
    async fn initialize_index(&mut self, source_dataset: &Dataset, index_name: &str) -> Result<()>;

    /// Initialize all indices on this dataset based on indices from a source dataset.
    /// This will call `initialize_index` for each non-system index in the source dataset.
    async fn initialize_indices(&mut self, source_dataset: &Dataset) -> Result<()>;
}

#[async_trait]
impl DatasetIndexInternalExt for Dataset {
    async fn open_generic_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn Index>> {
        // Checking for cache existence is cheap so we just check both scalar and vector caches
        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;
        let cache_key = ScalarIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self.index_cache.get_unsized_with_key(&cache_key).await {
            return Ok(index.as_index());
        }

        let vector_cache_key = VectorIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self
            .index_cache
            .get_unsized_with_key(&vector_cache_key)
            .await
        {
            return Ok(index.as_index());
        }

        let frag_reuse_cache_key = FragReuseIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self.index_cache.get_with_key(&frag_reuse_cache_key).await {
            return Ok(index.as_index());
        }

        // Sometimes we want to open an index and we don't care if it is a scalar or vector index.
        // For example, we might want to get statistics for an index, regardless of type.
        //
        // Currently, we solve this problem by checking for the existence of INDEX_FILE_NAME since
        // only vector indices have this file.  In the future, once we support multiple kinds of
        // scalar indices, we may start having this file with scalar indices too.  Once that happens
        // we can just read this file and look at the `implementation` or `index_type` fields to
        // determine what kind of index it is.
        let index_meta = self.load_index(uuid).await?.ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", uuid),
            location: location!(),
        })?;
        let index_dir = self.indice_files_dir(&index_meta)?;
        let index_file = index_dir.child(uuid).child(INDEX_FILE_NAME);
        if self.object_store.exists(&index_file).await? {
            let index = self.open_vector_index(column, uuid, metrics).await?;
            Ok(index.as_index())
        } else {
            let index = self.open_scalar_index(column, uuid, metrics).await?;
            Ok(index.as_index())
        }
    }

    async fn open_scalar_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;
        let cache_key = ScalarIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self.index_cache.get_unsized_with_key(&cache_key).await {
            return Ok(index);
        }

        let index_meta = self.load_index(uuid).await?.ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", uuid),
            location: location!(),
        })?;

        let index = scalar::open_scalar_index(self, column, &index_meta, metrics).await?;

        info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_SCALAR, index_type=index.index_type().to_string());
        metrics.record_index_load();

        self.index_cache
            .insert_unsized_with_key(&cache_key, index.clone())
            .await;
        Ok(index)
    }

    async fn open_vector_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;
        let cache_key = VectorIndexCacheKey::new(uuid, frag_reuse_uuid.as_ref());

        if let Some(index) = self.index_cache.get_unsized_with_key(&cache_key).await {
            log::debug!("Found vector index in cache uuid: {}", uuid);
            return Ok(index);
        }

        let frag_reuse_index = self.open_frag_reuse_index(metrics).await?;
        let index_meta = self.load_index(uuid).await?.ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", uuid),
            location: location!(),
        })?;
        let index_dir = self.indice_files_dir(&index_meta)?;
        let index_file = index_dir.child(uuid).child(INDEX_FILE_NAME);
        let reader: Arc<dyn Reader> = self.object_store.open(&index_file).await?.into();

        let tailing_bytes = read_last_block(reader.as_ref()).await?;
        let (major_version, minor_version) = read_version(&tailing_bytes)?;

        // Namespace the index cache by the UUID of the index.
        let index_cache = self.index_cache.with_key_prefix(&cache_key.key());

        // the index file is in lance format since version (0,2)
        // TODO: we need to change the legacy IVF_PQ to be in lance format
        let index = match (major_version, minor_version) {
            (0, 1) | (0, 0) => {
                info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.1", index_type="IVF_PQ");
                let proto = open_index_proto(reader.as_ref()).await?;
                match &proto.implementation {
                    Some(Implementation::VectorIndex(vector_index)) => {
                        let dataset = Arc::new(self.clone());
                        vector::open_vector_index(
                            dataset,
                            uuid,
                            vector_index,
                            reader,
                            frag_reuse_index,
                        )
                        .await
                    }
                    None => Err(Error::Internal {
                        message: "Index proto was missing implementation field".into(),
                        location: location!(),
                    }),
                }
            }

            (0, 2) => {
                info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.2", index_type="IVF_PQ");
                let reader = FileReader::try_new_self_described_from_reader(
                    reader.clone(),
                    Some(&self.metadata_cache.file_metadata_cache(&index_file)),
                )
                .await?;
                vector::open_vector_index_v2(
                    Arc::new(self.clone()),
                    column,
                    uuid,
                    reader,
                    frag_reuse_index,
                )
                .await
            }

            (0, 3) | (2, _) => {
                let scheduler = ScanScheduler::new(
                    self.object_store.clone(),
                    SchedulerConfig::max_bandwidth(&self.object_store),
                );
                let file = scheduler
                    .open_file(&index_file, &CachedFileSize::unknown())
                    .await?;
                let reader = v2::reader::FileReader::try_open(
                    file,
                    None,
                    Default::default(),
                    &self.metadata_cache.file_metadata_cache(&index_file),
                    FileReaderOptions::default(),
                )
                .await?;
                let index_metadata = reader
                    .schema()
                    .metadata
                    .get(INDEX_METADATA_SCHEMA_KEY)
                    .ok_or(Error::Index {
                        message: "Index Metadata not found".to_owned(),
                        location: location!(),
                    })?;
                let index_metadata: lance_index::IndexMetadata =
                    serde_json::from_str(index_metadata)?;
                let field = self.schema().field(column).ok_or_else(|| Error::Index {
                    message: format!("Column {} does not exist in the schema", column),
                    location: location!(),
                })?;

                let (_, element_type) = get_vector_type(self.schema(), column)?;

                info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.3", index_type=index_metadata.index_type);

                match index_metadata.index_type.as_str() {
                    "IVF_FLAT" => match element_type {
                        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                            let ivf = IVFIndex::<FlatIndex, FlatQuantizer>::try_new(
                                self.object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                frag_reuse_index,
                                self.metadata_cache.as_ref(),
                                index_cache,
                            )
                            .await?;
                            Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                        }
                        DataType::UInt8 => {
                            let ivf = IVFIndex::<FlatIndex, FlatBinQuantizer>::try_new(
                                self.object_store.clone(),
                                index_dir,
                                uuid.to_owned(),
                                frag_reuse_index,
                                self.metadata_cache.as_ref(),
                                index_cache,
                            )
                            .await?;
                            Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                        }
                        _ => Err(Error::Index {
                            message: format!(
                                "the field type {} is not supported for FLAT index",
                                field.data_type()
                            ),
                            location: location!(),
                        }),
                    },

                    "IVF_PQ" => {
                        let ivf = IVFIndex::<FlatIndex, ProductQuantizer>::try_new(
                            self.object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_SQ" => {
                        let ivf = IVFIndex::<FlatIndex, ScalarQuantizer>::try_new(
                            self.object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_RQ" => {
                        let ivf = IVFIndex::<FlatIndex, RabitQuantizer>::try_new(
                            self.object_store.clone(),
                            self.indices_dir(),
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_HNSW_FLAT" => {
                        let uri = index_dir.child(uuid).child("index.pb");
                        let file_metadata_cache =
                            self.session.metadata_cache.file_metadata_cache(&uri);
                        let ivf = IVFIndex::<HNSW, FlatQuantizer>::try_new(
                            self.object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            &file_metadata_cache,
                            index_cache,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_HNSW_SQ" => {
                        let ivf = IVFIndex::<HNSW, ScalarQuantizer>::try_new(
                            self.object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_HNSW_PQ" => {
                        let ivf = IVFIndex::<HNSW, ProductQuantizer>::try_new(
                            self.object_store.clone(),
                            index_dir,
                            uuid.to_owned(),
                            frag_reuse_index,
                            self.metadata_cache.as_ref(),
                            index_cache,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    _ => Err(Error::Index {
                        message: format!("Unsupported index type: {}", index_metadata.index_type),
                        location: location!(),
                    }),
                }
            }

            _ => Err(Error::Index {
                message: "unsupported index version (maybe need to upgrade your lance version)"
                    .to_owned(),
                location: location!(),
            }),
        };
        let index = index?;
        metrics.record_index_load();
        self.index_cache
            .insert_unsized_with_key(&cache_key, index.clone())
            .await;
        Ok(index)
    }

    async fn open_frag_reuse_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<FragReuseIndex>>> {
        if let Some(frag_reuse_index_meta) = self.load_index_by_name(FRAG_REUSE_INDEX_NAME).await? {
            let uuid = frag_reuse_index_meta.uuid.to_string();
            let frag_reuse_key = FragReuseIndexKey { uuid: &uuid };
            let uuid_clone = uuid.clone();

            let index = self
                .index_cache
                .get_or_insert_with_key(frag_reuse_key, || async move {
                    let index_meta = self.load_index(&uuid_clone).await?.ok_or_else(|| Error::Index {
                        message: format!("Index with id {} does not exist", uuid_clone),
                        location: location!(),
                    })?;
                    let index_details = load_frag_reuse_index_details(self, &index_meta).await?;
                    let index =
                        open_frag_reuse_index(frag_reuse_index_meta.uuid, index_details.as_ref()).await?;

                    info!(target: TRACE_IO_EVENTS, index_uuid=uuid_clone, r#type=IO_TYPE_OPEN_FRAG_REUSE);
                    metrics.record_index_load();

                    Ok(index)
                })
                .await?;

            Ok(Some(index))
        } else {
            Ok(None)
        }
    }

    async fn open_mem_wal_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<MemWalIndex>>> {
        let Some(mem_wal_meta) = self.load_index_by_name(MEM_WAL_INDEX_NAME).await? else {
            return Ok(None);
        };

        let frag_reuse_uuid = self.frag_reuse_index_uuid().await;
        let cache_key = MemWalCacheKey::new(&mem_wal_meta.uuid, frag_reuse_uuid.as_ref());
        if let Some(index) = self.index_cache.get_with_key(&cache_key).await {
            log::debug!("Found MemWAL index in cache uuid: {}", mem_wal_meta.uuid);
            return Ok(Some(index));
        }

        let uuid = mem_wal_meta.uuid.to_string();

        let index_meta = self.load_index(&uuid).await?.ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", uuid),
            location: location!(),
        })?;
        let index = open_mem_wal_index(index_meta)?;

        info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_MEM_WAL);
        metrics.record_index_load();

        self.index_cache
            .insert_with_key(&cache_key, index.clone())
            .await;
        Ok(Some(index))
    }

    async fn frag_reuse_index_uuid(&self) -> Option<Uuid> {
        if let Ok(indices) = self.load_indices().await {
            indices
                .iter()
                .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                .map(|idx| idx.uuid)
        } else {
            None
        }
    }

    #[instrument(level = "trace", skip_all)]
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo> {
        let indices = self.load_indices().await?;
        let schema = self.schema();
        let mut indexed_fields = Vec::new();
        for index in indices.iter().filter(|idx| {
            let idx_schema = schema.project_by_ids(idx.fields.as_slice(), true);
            let is_vector_index = idx_schema
                .fields
                .iter()
                .any(|f| is_vector_field(f.data_type()));

            // Check if this is an FTS index by looking at index details
            let is_fts_index = if let Some(details) = &idx.index_details {
                IndexDetails(details.clone()).supports_fts()
            } else {
                false
            };

            // Only include indices with non-empty fragment bitmaps, except for FTS indices
            // which need to be discoverable even when empty
            let has_non_empty_bitmap = idx.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                !bitmap.is_empty() && !(bitmap & self.fragment_bitmap.as_ref()).is_empty()
            });

            idx.fields.len() == 1 && !is_vector_index && (has_non_empty_bitmap || is_fts_index)
        }) {
            let field = index.fields[0];
            let field = schema.field_by_id(field).ok_or_else(|| Error::Internal {
                message: format!(
                    "Index referenced a field with id {field} which did not exist in the schema"
                ),
                location: location!(),
            })?;

            // Build the full field path for nested fields
            let field_path = if let Some(ancestors) = schema.field_ancestry_by_id(field.id) {
                let field_refs: Vec<&str> = ancestors.iter().map(|f| f.name.as_str()).collect();
                lance_core::datatypes::format_field_path(&field_refs)
            } else {
                field.name.clone()
            };

            let index_details = IndexDetails(fetch_index_details(self, &field_path, index).await?);
            if index_details.is_vector() {
                continue;
            }

            let plugin = index_details.get_plugin()?;
            let query_parser = plugin.new_query_parser(index.name.clone(), &index_details.0);

            if let Some(query_parser) = query_parser {
                indexed_fields.push((field_path, (field.data_type(), query_parser)));
            }
        }
        let mut index_info_map = HashMap::with_capacity(indexed_fields.len());
        for indexed_field in indexed_fields {
            // Need to wrap in an option here because we know that only one of and_modify and or_insert will be called
            // but the rust compiler does not.
            let mut parser = Some(indexed_field.1 .1);
            let parser = &mut parser;
            index_info_map
                .entry(indexed_field.0)
                .and_modify(|existing: &mut (DataType, Box<MultiQueryParser>)| {
                    // If there are two indices on the same column, they must have the same type
                    debug_assert_eq!(existing.0, indexed_field.1 .0);

                    existing.1.add(parser.take().unwrap());
                })
                .or_insert_with(|| {
                    (
                        indexed_field.1 .0,
                        Box::new(MultiQueryParser::single(parser.take().unwrap())),
                    )
                });
        }
        Ok(ScalarIndexInfo {
            indexed_columns: index_info_map,
        })
    }

    async fn unindexed_fragments(&self, name: &str) -> Result<Vec<Fragment>> {
        let indices = self.load_indices_by_name(name).await?;
        let mut total_fragment_bitmap = RoaringBitmap::new();
        for idx in indices.iter() {
            total_fragment_bitmap |= idx.fragment_bitmap.as_ref().ok_or(Error::Index {
                message: "Please upgrade lance to 0.8+ to use this function".to_string(),
                location: location!(),
            })?;
        }
        Ok(self
            .fragments()
            .iter()
            .filter(|f| !total_fragment_bitmap.contains(f.id as u32))
            .cloned()
            .collect())
    }

    async fn indexed_fragments(&self, name: &str) -> Result<Vec<Vec<Fragment>>> {
        let indices = self.load_indices_by_name(name).await?;
        indices
            .iter()
            .map(|index| {
                let fragment_bitmap = index.fragment_bitmap.as_ref().ok_or(Error::Index {
                    message: "Please upgrade lance to 0.8+ to use this function".to_string(),
                    location: location!(),
                })?;
                let mut indexed_frags = Vec::with_capacity(fragment_bitmap.len() as usize);
                for frag in self.fragments().iter() {
                    if fragment_bitmap.contains(frag.id as u32) {
                        indexed_frags.push(frag.clone());
                    }
                }
                Ok(indexed_frags)
            })
            .collect()
    }

    async fn initialize_index(&mut self, source_dataset: &Dataset, index_name: &str) -> Result<()> {
        let source_indices = source_dataset.load_indices_by_name(index_name).await?;

        if source_indices.is_empty() {
            return Err(Error::Index {
                message: format!("Index '{}' not found in source dataset", index_name),
                location: location!(),
            });
        }

        let source_index = source_indices
            .iter()
            .min_by_key(|idx| idx.created_at)
            .ok_or_else(|| Error::Index {
                message: format!("Could not determine oldest index for '{}'", index_name),
                location: location!(),
            })?;

        let mut field_names = Vec::new();
        for field_id in source_index.fields.iter() {
            let source_field = source_dataset
                .schema()
                .field_by_id(*field_id)
                .ok_or_else(|| Error::Index {
                    message: format!("Field with id {} not found in source dataset", field_id),
                    location: location!(),
                })?;

            let target_field =
                self.schema()
                    .field(&source_field.name)
                    .ok_or_else(|| Error::Index {
                        message: format!(
                            "Field '{}' required by index '{}' not found in target dataset",
                            source_field.name, index_name
                        ),
                        location: location!(),
                    })?;

            if source_field.data_type() != target_field.data_type() {
                return Err(Error::Index {
                    message: format!(
                        "Field '{}' has different types in source ({:?}) and target ({:?}) datasets",
                        source_field.name,
                        source_field.data_type(),
                        target_field.data_type()
                    ),
                    location: location!(),
                });
            }

            field_names.push(source_field.name.as_str());
        }

        if field_names.is_empty() {
            return Err(Error::Index {
                message: format!("Index '{}' has no fields", index_name),
                location: location!(),
            });
        }

        if let Some(index_details) = &source_index.index_details {
            let index_details_wrapper = IndexDetails(index_details.clone());

            if index_details_wrapper.is_vector() {
                vector::initialize_vector_index(self, source_dataset, source_index, &field_names)
                    .await?;
            } else {
                scalar::initialize_scalar_index(self, source_dataset, source_index, &field_names)
                    .await?;
            }
        } else {
            log::warn!(
                "Index '{}' has no index_details, skipping",
                source_index.name
            );
        }

        Ok(())
    }

    async fn initialize_indices(&mut self, source_dataset: &Dataset) -> Result<()> {
        let source_indices = source_dataset.load_indices().await?;
        let non_system_indices: Vec<_> = source_indices
            .iter()
            .filter(|idx| !lance_index::is_system_index(idx))
            .collect();

        if non_system_indices.is_empty() {
            return Ok(());
        }

        let mut unique_index_names = HashSet::new();
        for index in non_system_indices.iter() {
            unique_index_names.insert(index.name.clone());
        }

        for index_name in unique_index_names {
            self.initialize_index(source_dataset, &index_name).await?;
        }

        Ok(())
    }
}

fn is_vector_field(data_type: DataType) -> bool {
    match data_type {
        DataType::FixedSizeList(_, _) => true,
        DataType::List(inner) => {
            // If the inner type is a fixed size list, then it is a multivector field
            matches!(inner.data_type(), DataType::FixedSizeList(_, _))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{compact_files, CompactionOptions};
    use crate::dataset::{ReadParams, WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::session::Session;
    use crate::utils::test::{copy_test_data_to_tmp, DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::Int32Array;
    use lance_io::utils::tracking_store::IOTracker;
    use lance_io::{assert_io_eq, assert_io_lt};

    use super::*;

    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, Int32Type};
    use arrow_array::{
        FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{Field, Schema};
    use lance_arrow::*;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::gen_batch;
    use lance_datagen::{array, BatchCount, Dimension, RowCount};
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams};
    use lance_index::vector::{
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, sq::builder::SQBuildParams,
    };
    use lance_io::object_store::ObjectStoreParams;
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_testing::datagen::generate_random_array;
    use rstest::rstest;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_recreate_index() {
        const DIM: i32 = 8;
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "v",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
            Field::new(
                "o",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
                true,
            ),
        ]));
        let data = generate_random_array(2048 * DIM as usize);
        let batches: Vec<RecordBatch> = vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(FixedSizeListArray::try_new_from_values(data.clone(), DIM).unwrap()),
                Arc::new(FixedSizeListArray::try_new_from_values(data, DIM).unwrap()),
            ],
        )
        .unwrap()];

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset
            .create_index(&["o"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Create index again
        dataset
            .create_index(&["v"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Can not overwrite an index on different columns.
        assert!(dataset
            .create_index(
                &["v"],
                IndexType::Vector,
                Some("o_idx".to_string()),
                &params,
                true,
            )
            .await
            .is_err());
    }

    fn sample_vector_field() -> Field {
        let dimensions = 16;
        let column_name = "vec";
        Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        )
    }

    #[tokio::test]
    async fn test_drop_index() {
        let test_dir = TempStrDir::default();
        let schema = Schema::new(vec![
            sample_vector_field(),
            Field::new("ints", DataType::Int32, false),
        ]);
        let mut dataset = lance_datagen::rand(&schema)
            .into_dataset(
                &test_dir,
                FragmentCount::from(1),
                FragmentRowCount::from(256),
            )
            .await
            .unwrap();

        let idx_name = "name".to_string();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some(idx_name.clone()),
                &VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["ints"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        assert_eq!(dataset.load_indices().await.unwrap().len(), 2);

        dataset.drop_index(&idx_name).await.unwrap();

        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

        // Even though we didn't give the scalar index a name it still has an auto-generated one we can use
        let scalar_idx_name = &dataset.load_indices().await.unwrap()[0].name;
        dataset.drop_index(scalar_idx_name).await.unwrap();

        assert_eq!(dataset.load_indices().await.unwrap().len(), 0);

        // Make sure it returns an error if the index doesn't exist
        assert!(dataset.drop_index(scalar_idx_name).await.is_err());
    }

    #[tokio::test]
    async fn test_count_index_rows() {
        let test_dir = TempStrDir::default();
        let dimensions = 16;
        let column_name = "vec";
        let field = sample_vector_field();
        let schema = Arc::new(Schema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        // Make sure it returns None if there's no index with the passed identifier
        assert!(dataset.index_statistics("bad_id").await.is_err());
        // Create an index
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);

        // Now we'll append some rows which shouldn't be indexed and see the
        // count change
        let float_arr = generate_random_array(512 * dimensions as usize);
        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema);
        dataset.append(reader, None).await.unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
    }

    #[tokio::test]
    async fn test_optimize_delta_indices() {
        let dimensions = 16;
        let column_name = "vec";
        let vec_field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let other_column_name = "other_vec";
        let other_vec_field = Field::new(
            other_column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let schema = Arc::new(Schema::new(vec![vec_field, other_vec_field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors = Arc::new(
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap(),
        );

        let record_batch =
            RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()]).unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &[other_column_name],
                IndexType::Vector,
                Some("other_vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        async fn get_stats(dataset: &Dataset, name: &str) -> serde_json::Value {
            serde_json::from_str(&dataset.index_statistics(name).await.unwrap()).unwrap()
        }
        async fn get_meta(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .filter(|m| m.name == name)
                .cloned()
                .collect()
        }
        fn get_bitmap(meta: &IndexMetadata) -> Vec<u32> {
            meta.fragment_bitmap.as_ref().unwrap().iter().collect()
        }

        let stats = get_stats(&dataset, "vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        dataset.append(reader, None).await.unwrap();
        let stats = get_stats(&dataset, "vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        dataset
            .optimize_indices(&OptimizeOptions::append().index_names(vec![])) // Does nothing because no index name is passed
            .await
            .unwrap();
        let stats = get_stats(&dataset, "vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        // optimize the other index
        dataset
            .optimize_indices(
                &OptimizeOptions::append().index_names(vec!["other_vec_idx".to_owned()]),
            )
            .await
            .unwrap();
        let stats = get_stats(&dataset, "vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);

        let stats = get_stats(&dataset, "other_vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 2);
        let meta = get_meta(&dataset, "other_vec_idx").await;
        assert_eq!(meta.len(), 2);
        assert_eq!(get_bitmap(&meta[0]), vec![0]);
        assert_eq!(get_bitmap(&meta[1]), vec![1]);

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();

        let stats = get_stats(&dataset, "vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0, 1]);

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();
        let stats = get_stats(&dataset, "other_vec_idx").await;
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
        let meta = get_meta(&dataset, "other_vec_idx").await;
        assert_eq!(meta.len(), 1);
        assert_eq!(get_bitmap(&meta[0]), vec![0, 1]);
    }

    #[tokio::test]
    async fn test_optimize_ivf_hnsw_sq_delta_indices() {
        let test_dir = TempStrDir::default();
        let dimensions = 16;
        let column_name = "vec";
        let field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );
        let schema = Arc::new(Schema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let ivf_params = IvfBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let sq_params = SQBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::L2,
            ivf_params,
            hnsw_params,
            sq_params,
        );
        dataset
            .create_index(
                &[column_name],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
        dataset.append(reader, None).await.unwrap();
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 2);

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_optimize_fts(#[values(false, true)] with_position: bool) {
        let words = ["apple", "banana", "cherry", "date"];

        let dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let data = StringArray::from_iter_values(words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
        let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(batch_iterator, &dir, None).await.unwrap();

        let params = InvertedIndexParams::default()
            .lower_case(false)
            .with_position(with_position);
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        async fn assert_indexed_rows(dataset: &Dataset, expected_indexed_rows: usize) {
            let stats = dataset.index_statistics("text_idx").await.unwrap();
            let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
            let indexed_rows = stats["num_indexed_rows"].as_u64().unwrap() as usize;
            let unindexed_rows = stats["num_unindexed_rows"].as_u64().unwrap() as usize;
            let num_rows = dataset.count_all_rows().await.unwrap();
            assert_eq!(indexed_rows, expected_indexed_rows);
            assert_eq!(unindexed_rows, num_rows - expected_indexed_rows);
        }

        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        let new_words = ["elephant", "fig", "grape", "honeydew"];
        let new_data = StringArray::from_iter_values(new_words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(new_data)]).unwrap();
        let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        dataset.append(batch_iter, None).await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        for &word in words.iter().chain(new_words.iter()) {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 1);
            assert_eq!(texts[0], word);
        }

        let uppercase_words = ["Apple", "Banana", "Cherry", "Date"];
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 0);
        }
        let new_data = StringArray::from_iter_values(uppercase_words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(new_data)]).unwrap();
        let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        dataset.append(batch_iter, None).await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        // we should be able to query the new words
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
            assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);
        }

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_indexed_rows(&dataset, num_rows).await;

        // we should be able to query the new words after optimization
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();

            assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
            assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);

            // we should be able to query the new words after compaction
            compact_files(&mut dataset, CompactionOptions::default(), None)
                .await
                .unwrap();
            for &word in uppercase_words.iter() {
                let query_result = dataset
                    .scan()
                    .project(&["text"])
                    .unwrap()
                    .full_text_search(FullTextSearchQuery::new(word.to_string()))
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
                let texts = query_result["text"]
                    .as_string::<i32>()
                    .iter()
                    .map(|v| match v {
                        None => "".to_string(),
                        Some(v) => v.to_string(),
                    })
                    .collect::<Vec<String>>();
                assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
                assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);
            }
            assert_indexed_rows(&dataset, num_rows).await;
        }
    }

    #[tokio::test]
    async fn test_create_index_too_small_for_pq() {
        let test_dir = TempStrDir::default();
        let dimensions = 1536;

        let field = Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );

        let schema = Arc::new(Schema::new(vec![field]));
        let float_arr = generate_random_array(100 * dimensions as usize);

        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 96, DistanceType::L2, 1);
        let result = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await;

        assert!(matches!(result, Err(Error::Index { .. })));
        if let Error::Index { message, .. } = result.unwrap_err() {
            assert_eq!(
                message,
                "Not enough rows to train PQ. Requires 256 rows but only 100 available",
            )
        }
    }

    #[tokio::test]
    async fn test_create_bitmap_index() {
        let test_dir = TempStrDir::default();
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset
            .create_index(
                &["tag"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let index = dataset
            .open_generic_index("tag", &indices[0].uuid.to_string(), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(index.index_type(), IndexType::Bitmap);
    }

    // #[tokio::test]
    #[lance_test_macros::test(tokio::test)]
    async fn test_load_indices() {
        let session = Arc::new(Session::default());
        let io_tracker = Arc::new(IOTracker::default());
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(io_tracker.clone()),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        };

        let test_dir = TempStrDir::default();
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();
        dataset
            .create_index(
                &["tag"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        io_tracker.incremental_stats(); // Reset

        let indices = dataset.load_indices().await.unwrap();
        let stats = io_tracker.incremental_stats();
        // We should already have this cached since we just wrote it.
        assert_io_eq!(stats, read_iops, 0);
        assert_io_eq!(stats, read_bytes, 0);
        assert_eq!(indices.len(), 1);

        session.index_cache.clear().await; // Clear the cache

        let dataset2 = DatasetBuilder::from_uri(test_uri)
            .with_session(session.clone())
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_tracker.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        let stats = io_tracker.incremental_stats(); // Reset
        assert_io_lt!(stats, read_bytes, 64 * 1024);

        // Because the manifest is so small, we should have opportunistically
        // cached the indices in memory already.
        let indices2 = dataset2.load_indices().await.unwrap();
        let stats = io_tracker.incremental_stats();
        assert_io_eq!(stats, read_iops, 0);
        assert_io_eq!(stats, read_bytes, 0);
        assert_eq!(indices2.len(), 1);
    }

    #[tokio::test]
    async fn test_remap_empty() {
        let data = gen_batch()
            .col("int", array::step::<Int32Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(16)),
            )
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 1, DistanceType::L2, 1);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let remap_to_empty = (0..dataset.count_all_rows().await.unwrap())
            .map(|i| (i as u64, None))
            .collect::<HashMap<_, _>>();
        let new_uuid = remap_index(&dataset, &index_uuid, &remap_to_empty)
            .await
            .unwrap();
        assert_eq!(new_uuid, RemapResult::Keep(index_uuid));
    }

    #[tokio::test]
    async fn test_optimize_ivf_pq_up_to_date() {
        // https://github.com/lancedb/lance/issues/4016
        let nrows = 256;
        let dimensions = 16;
        let column_name = "vector";
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                column_name,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimensions,
                ),
                false,
            ),
        ]));

        let float_arr = generate_random_array(nrows * dimensions as usize);
        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from_iter_values(0..nrows as i32)),
                Arc::new(vectors),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(&[column_name], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query_vector = generate_random_array(dimensions as usize);

        let nearest = dataset
            .scan()
            .nearest(column_name, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let ids = nearest["id"].as_primitive::<Int32Type>();
        let mut seen = HashSet::new();
        for id in ids.values() {
            assert!(seen.insert(*id), "Duplicate id found: {}", id);
        }

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        dataset.validate().await.unwrap();

        let nearest_after = dataset
            .scan()
            .nearest(column_name, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let ids = nearest_after["id"].as_primitive::<Int32Type>();
        let mut seen = HashSet::new();
        for id in ids.values() {
            assert!(seen.insert(*id), "Duplicate id found: {}", id);
        }
    }

    #[tokio::test]
    async fn test_index_created_at_timestamp() {
        // Test that created_at is set when creating an index
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("values", DataType::Utf8, false),
        ]));

        let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(values),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Record time before creating index
        let before_index = chrono::Utc::now();

        // Create a scalar index
        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("test_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Record time after creating index
        let after_index = chrono::Utc::now();

        // Get index metadata
        let indices = dataset.load_indices().await.unwrap();
        let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();

        // Verify created_at is set and within reasonable bounds
        assert!(test_index.created_at.is_some());
        let created_at = test_index.created_at.unwrap();
        assert!(created_at >= before_index);
        assert!(created_at <= after_index);
    }

    #[tokio::test]
    async fn test_index_statistics_updated_at() {
        // Test that updated_at appears in index statistics
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("values", DataType::Utf8, false),
        ]));

        let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(values),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create a scalar index
        dataset
            .create_index(
                &["values"],
                IndexType::Scalar,
                Some("test_idx".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Get index statistics
        let stats_str = dataset.index_statistics("test_idx").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_str).unwrap();

        // Verify updated_at_timestamp_ms field exists in statistics
        assert!(stats["updated_at_timestamp_ms"].is_number());
        let updated_at = stats["updated_at_timestamp_ms"].as_u64().unwrap();

        // Get the index metadata to compare with created_at
        let indices = dataset.load_indices().await.unwrap();
        let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();
        let created_at = test_index.created_at.unwrap().timestamp_millis() as u64;

        // For a single index, updated_at should equal created_at
        assert_eq!(updated_at, created_at);
    }

    #[tokio::test]
    async fn test_index_statistics_updated_at_multiple_deltas() {
        // Test that updated_at reflects max(created_at) across multiple delta indices
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                false,
            ),
        ]));

        // Create initial dataset (need more rows for PQ training)
        let num_rows = 300;
        let float_arr = generate_random_array(4 * num_rows);
        let vectors = FixedSizeListArray::try_new_from_values(float_arr, 4).unwrap();
        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
                Arc::new(vectors),
            ],
        )
        .unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Create vector index
        let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_vec_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Get initial statistics
        let stats_str_1 = dataset.index_statistics("test_vec_idx").await.unwrap();
        let stats_1: serde_json::Value = serde_json::from_str(&stats_str_1).unwrap();
        let initial_updated_at = stats_1["updated_at_timestamp_ms"].as_u64().unwrap();

        // Add more data to create additional delta indices
        std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamp

        let num_rows_2 = 50;
        let float_arr_2 = generate_random_array(4 * num_rows_2);
        let vectors_2 = FixedSizeListArray::try_new_from_values(float_arr_2, 4).unwrap();
        let record_batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    num_rows as i32..(num_rows + num_rows_2) as i32,
                )),
                Arc::new(vectors_2),
            ],
        )
        .unwrap();

        let reader_2 =
            RecordBatchIterator::new(vec![record_batch_2].into_iter().map(Ok), schema.clone());

        dataset.append(reader_2, None).await.unwrap();

        // Update the index
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_vec_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Get updated statistics
        let stats_str_2 = dataset.index_statistics("test_vec_idx").await.unwrap();
        let stats_2: serde_json::Value = serde_json::from_str(&stats_str_2).unwrap();
        let final_updated_at = stats_2["updated_at_timestamp_ms"].as_u64().unwrap();

        // The updated_at should be newer than the initial one
        assert!(final_updated_at >= initial_updated_at);
    }

    #[tokio::test]
    async fn test_index_statistics_updated_at_none_when_no_created_at() {
        // Test backward compatibility: when indices were created before the created_at field,
        // updated_at_timestamp_ms should be null

        // Use test data created with Lance 0.29.0 (before created_at field was added)
        let test_dir =
            copy_test_data_to_tmp("v0.30.0_pre_created_at/index_without_created_at").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        let dataset = Dataset::open(test_uri).await.unwrap();

        // Get the index metadata to verify created_at is None
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty(), "Test dataset should have indices");

        // Verify that the index created with old version has no created_at
        for index in indices.iter() {
            assert!(
                index.created_at.is_none(),
                "Index from old version should have created_at = None"
            );
        }

        // Get index statistics - should work even with old indices
        let index_name = &indices[0].name;
        let stats_str = dataset.index_statistics(index_name).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_str).unwrap();

        // Verify updated_at_timestamp_ms field is null when no indices have created_at
        assert!(
            stats["updated_at_timestamp_ms"].is_null(),
            "updated_at_timestamp_ms should be null when no indices have created_at timestamps"
        );
    }
    #[rstest]
    #[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
    #[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
    #[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
    #[tokio::test]
    async fn test_create_empty_scalar_index(
        #[case] column_name: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
    ) {
        use lance_datagen::{array, BatchCount, ByteCount, RowCount};

        // Create dataset with scalar and text columns (no vector column needed)
        let reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

        // Create an empty index with train=false
        // Test using IntoFuture - can await directly without calling .execute()
        dataset
            .create_index_builder(&[column_name], index_type, params.as_ref())
            .name("index".to_string())
            .train(false)
            .await
            .unwrap();

        // Verify we can get index statistics
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Empty index should have zero indexed rows"
        );

        // Append new data using lance_datagen
        let append_reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        dataset.append(append_reader, None).await.unwrap();

        // Critical test: Verify the empty index is still present after append
        let indices_after_append = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_append.len(),
            1,
            "Index should be retained after append for index type {:?}",
            index_type
        );

        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Empty index should still have zero indexed rows after append"
        );

        // Test optimize_indices with empty index
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Verify the index still exists after optimization
        let indices_after_optimize = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_optimize.len(),
            1,
            "Index should still exist after optimization"
        );

        // Check index statistics after optimization
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_unindexed_rows"], 0,
            "Empty index should indexed all rows"
        );
    }

    /// Helper function to check if an index is being used in a query plan
    fn assert_index_usage(plan: &str, column_name: &str, should_use_index: bool, context: &str) {
        let index_used = if column_name == "text" {
            // For inverted index, look for MatchQuery which indicates FTS index usage
            plan.contains("MatchQuery")
        } else {
            // For btree/bitmap, look for MaterializeIndex which indicates scalar index usage
            plan.contains("ScalarIndexQuery")
        };

        if should_use_index {
            assert!(
                index_used,
                "Query plan should use index {}: {}",
                context, plan
            );
        } else {
            assert!(
                !index_used,
                "Query plan should NOT use index {}: {}",
                context, plan
            );
        }
    }

    /// Test that scalar indices are retained after deleting all data from a table.
    ///
    /// This test verifies that when we:
    /// 1. Create a table with data
    /// 2. Add a scalar index with train=true
    /// 3. Delete all data in the table
    /// The index remains available on the table.
    #[rstest]
    #[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
    #[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
    #[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
    #[tokio::test]
    async fn test_scalar_index_retained_after_delete_all(
        #[case] column_name: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
    ) {
        use lance_datagen::{array, BatchCount, ByteCount, RowCount};

        // Create dataset with initial data
        let reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

        // Create index with train=true (normal index with data)
        dataset
            .create_index_builder(&[column_name], index_type, params.as_ref())
            .name("index".to_string())
            .train(true)
            .await
            .unwrap();

        // Verify index was created and has indexed rows
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 100,
            "Index should have indexed all 100 rows"
        );

        // Verify index is being used in queries before delete
        let plan = if column_name == "text" {
            // Use full-text search for inverted index
            dataset
                .scan()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        } else {
            // Use equality filter for btree/bitmap indices
            dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        };
        // Verify index is being used before delete
        assert_index_usage(&plan, column_name, true, "before delete");

        let indexes = dataset.load_indices().await.unwrap();
        let original_index = indexes[0].clone();

        // Delete all rows from the table
        dataset.delete("true").await.unwrap();

        // Verify table is empty
        let row_count = dataset.count_rows(None).await.unwrap();
        assert_eq!(row_count, 0, "Table should be empty after delete all");

        // Critical test: Verify the index still exists after deleting all data
        let indices_after_delete = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_delete.len(),
            1,
            "Index should be retained after deleting all data"
        );
        assert_eq!(
            indices_after_delete[0].name, "index",
            "Index name should remain the same after delete"
        );

        // Critical test: Verify the fragment bitmap is empty after delete
        let index_after_delete = &indices_after_delete[0];
        let effective_bitmap = index_after_delete
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap();
        assert!(
            effective_bitmap.is_empty(),
            "Effective bitmap should be empty after deleting all data"
        );
        assert_eq!(
            index_after_delete.fragment_bitmap, original_index.fragment_bitmap,
            "Fragment bitmap should remain the same after delete"
        );

        // Verify we can still get index statistics
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Index should now report zero indexed rows after delete all"
        );
        assert_eq!(
            stats["num_unindexed_rows"], 0,
            "Index should report zero unindexed rows after delete all"
        );
        assert_eq!(
            stats["num_indexed_fragments"], 0,
            "Index should report zero indexed fragments after delete all"
        );
        assert_eq!(
            stats["num_unindexed_fragments"], 0,
            "Index should report zero unindexed fragments after delete all"
        );

        // Verify index is NOT being used in queries after delete (empty bitmap)
        if column_name == "text" {
            // Inverted indexes will still appear to be used in FTS queries.
            // TODO: once metrics are working on FTS queries, we can check the
            // analyze plan output instead for index usage.
            let _plan_after_delete = dataset
                .scan()
                .project(&[column_name])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            assert_index_usage(
                &_plan_after_delete,
                column_name,
                true,
                "after delete (empty bitmap)",
            );
        } else {
            // Use equality filter for btree/bitmap indices
            let _plan_after_delete = dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            // Verify index is NOT being used after delete (empty bitmap)
            assert_index_usage(
                &_plan_after_delete,
                column_name,
                false,
                "after delete (empty bitmap)",
            );
        }

        // Test that we can append new data and the index is still there
        let append_reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        dataset.append(append_reader, None).await.unwrap();

        // Verify index still exists after append
        let indices_after_append = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_append.len(),
            1,
            "Index should still exist after appending to empty table"
        );
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 0,
            "Index should now report zero indexed rows after data is added"
        );

        // Test optimize_indices after delete all
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Verify index still exists after optimization
        let indices_after_optimize = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_optimize.len(),
            1,
            "Index should still exist after optimization following delete all"
        );

        // Verify we can still get index statistics after optimization
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"],
            dataset.count_rows(None).await.unwrap(),
            "Index should now cover all newly added rows after optimization"
        );
    }

    /// Test that scalar indices are retained after updating rows in a table.
    ///
    /// This test verifies that when we:
    /// 1. Create a table with data
    /// 2. Add a scalar index with train=true
    /// 3. Update rows in the table
    /// The index remains available on the table.
    #[rstest]
    #[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
    #[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
    #[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
    #[tokio::test]
    async fn test_scalar_index_retained_after_update(
        #[case] column_name: &str,
        #[case] index_type: IndexType,
        #[case] params: Box<dyn IndexParams>,
    ) {
        use crate::dataset::UpdateBuilder;
        use lance_datagen::{array, BatchCount, ByteCount, RowCount};

        // Create dataset with initial data
        let reader = lance_datagen::gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

        // Create index with train=true (normal index with data)
        dataset
            .create_index_builder(&[column_name], index_type, params.as_ref())
            .name("index".to_string())
            .train(true)
            .await
            .unwrap();

        // Verify index was created and has indexed rows
        let stats = dataset.index_statistics("index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(
            stats["num_indexed_rows"], 100,
            "Index should have indexed all 100 rows"
        );

        // Verify index is being used in queries before update
        let plan = if column_name == "text" {
            // Use full-text search for inverted index
            dataset
                .scan()
                .project(&[column_name])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        } else {
            // Use equality filter for btree/bitmap indices
            dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap()
        };
        // Verify index is being used before update
        assert_index_usage(&plan, column_name, true, "before update");

        // Update some rows - update first 50 rows
        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .set("i", "i + 1000")
            .unwrap()
            .set("text", "'updated_' || text")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let mut dataset = update_result.new_dataset.as_ref().clone();

        // Verify row count remains the same
        let row_count = dataset.count_rows(None).await.unwrap();
        assert_eq!(row_count, 100, "Row count should remain 100 after update");

        // Critical test: Verify the index still exists after updating data
        let indices_after_update = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_update.len(),
            1,
            "Index should be retained after updating rows"
        );

        // Critical test: Verify the effective fragment bitmap is empty after update
        let indices = dataset.load_indices().await.unwrap();
        let index = &indices[0];
        let effective_bitmap = index
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap();
        assert!(
            effective_bitmap.is_empty(),
            "Effective fragment bitmap should be empty after updating all data"
        );

        // Verify we can still get index statistics
        let stats_after_update = dataset.index_statistics("index").await.unwrap();
        let stats_after_update: serde_json::Value =
            serde_json::from_str(&stats_after_update).unwrap();

        // The index should still be available
        assert_eq!(
            stats_after_update["num_indexed_rows"], 0,
            "Index statistics should be zero after update, as it is not re-trained"
        );

        // Verify index behavior in queries after update (empty bitmap)
        if column_name == "text" {
            // Inverted indexes will still appear to be used in FTS queries even with empty bitmaps.
            // This is because FTS queries require an index to exist, and the query execution
            // will handle the empty bitmap appropriately by falling back to scanning unindexed data.
            // TODO: once metrics are working on FTS queries, we can check the
            // analyze plan output instead for actual index usage statistics.
            let _plan_after_update = dataset
                .scan()
                .project(&[column_name])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new("test".to_string()))
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            assert_index_usage(
                &_plan_after_update,
                column_name,
                true, // FTS indices always appear in the plan, even with empty bitmaps
                "after update (empty bitmap)",
            );
        } else {
            // Use equality filter for btree/bitmap indices
            let _plan_after_update = dataset
                .scan()
                .filter(format!("{} = 50", column_name).as_str())
                .unwrap()
                .explain_plan(false)
                .await
                .unwrap();
            // With immutable bitmaps, index is still used even with empty effective bitmap
            // The prefilter will handle non-existent fragments
            assert_index_usage(
                &_plan_after_update,
                column_name,
                false,
                "after update (empty effective bitmap)",
            );
        }

        // Test that we can optimize indices after update
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Verify index still exists after optimization
        let indices_after_optimize = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_optimize.len(),
            1,
            "Index should still exist after optimization following update"
        );

        let stats_after_optimization = dataset.index_statistics("index").await.unwrap();
        let stats_after_optimization: serde_json::Value =
            serde_json::from_str(&stats_after_optimization).unwrap();

        // The index should still be available
        assert_eq!(
            stats_after_optimization["num_unindexed_rows"], 0,
            "Index should have zero unindexed rows after optimization"
        );
    }

    // Helper function to validate indices after each clone iteration
    async fn validate_indices_after_clone(
        dataset: &Dataset,
        round: usize,
        expected_scalar_rows: usize,
        dimensions: u32,
    ) {
        // Verify cloned dataset has indices
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices.len(),
            2,
            "Round {}: Cloned dataset should have 2 indices",
            round
        );
        let index_names: HashSet<String> = indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(
            index_names.contains("vector_idx"),
            "Round {}: Should contain vector_idx",
            round
        );
        assert!(
            index_names.contains("category_idx"),
            "Round {}: Should contain category_idx",
            round
        );

        // Test basic data access without using indices for now
        // This ensures the dataset is accessible and data is intact
        // In chain cloning, each round adds 50 rows to the previous dataset
        // Round 1: 300 (original), Round 2: 350 (300 + 50 from round 1), Round 3: 400 (350 + 50 from round 2)
        let expected_total_rows = 300 + (round - 1) * 50;
        let total_rows = dataset.count_rows(None).await.unwrap();
        assert_eq!(
            total_rows, expected_total_rows,
            "Round {}: Should have {} rows after clone (chain cloning accumulates data)",
            round, expected_total_rows
        );

        // Verify vector search
        let query_vector = generate_random_array(dimensions as usize);
        let search_results = dataset
            .scan()
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(
            search_results.num_rows() > 0,
            "Round {}: Vector search should return results immediately after clone",
            round
        );

        // Test basic scalar query to verify data integrity
        let scalar_results = dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            expected_scalar_rows,
            scalar_results.num_rows(),
            "Round {}: Scalar query should return {} results",
            round,
            expected_scalar_rows
        );
    }

    #[tokio::test]
    async fn test_shallow_clone_with_index() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create a schema with both vector and scalar columns
        let dimensions = 16u32;
        // Generate test data using lance_datagen (300 rows to satisfy PQ training requirements)
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("category", array::fill_utf8("category_0".to_string()))
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(dimensions)),
            )
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));

        // Create initial dataset
        let mut dataset = Dataset::write(data, test_uri, None).await.unwrap();
        // Create vector index (IVF_PQ)
        let vector_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .unwrap();

        // Create scalar index (BTree)
        dataset
            .create_index(
                &["category"],
                IndexType::BTree,
                Some("category_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Verify indices were created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2, "Should have 2 indices");
        let index_names: HashSet<String> = indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(index_names.contains("vector_idx"));
        assert!(index_names.contains("category_idx"));

        // Test scalar query on source dataset
        let scalar_results = dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let source_scalar_query_rows = scalar_results.num_rows();
        assert!(
            scalar_results.num_rows() > 0,
            "Scalar query should return results"
        );

        // Multiple shallow clone iterations test with chain cloning
        let clone_rounds = 3;
        let mut current_dataset = dataset;

        for round in 1..=clone_rounds {
            let round_clone_dir = format!("{}/clone_round_{}", test_dir, round);
            let round_cloned_uri = &round_clone_dir;
            let tag_name = format!("shallow_clone_test_{}", round);

            // Create tag for this round (use current dataset for chain cloning)
            let current_version = current_dataset.version().version;
            current_dataset
                .tags()
                .create(&tag_name, current_version)
                .await
                .unwrap();

            // Perform shallow clone for this round (chain cloning from current dataset)
            let mut round_cloned_dataset = current_dataset
                .shallow_clone(round_cloned_uri, tag_name.as_str(), None)
                .await
                .unwrap();

            // Immediately validate indices after each clone
            validate_indices_after_clone(
                &round_cloned_dataset,
                round,
                source_scalar_query_rows,
                dimensions,
            )
            .await;

            // Complete validation cycle after each clone: append data, optimize index, validate
            // Append new data to the cloned dataset
            let new_data = gen_batch()
                .col(
                    "id",
                    array::step_custom::<Int32Type>(300 + (round * 50) as i32, 1),
                )
                .col("category", array::fill_utf8(format!("category_{}", round)))
                .col(
                    "vector",
                    array::rand_vec::<Float32Type>(Dimension::from(dimensions)),
                )
                .into_reader_rows(RowCount::from(50), BatchCount::from(1));

            round_cloned_dataset = Dataset::write(
                new_data,
                round_cloned_uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            // Verify row count increased
            let expected_rows = 300 + round * 50;
            let total_rows = round_cloned_dataset.count_rows(None).await.unwrap();
            assert_eq!(
                total_rows, expected_rows,
                "Round {}: Should have {} rows after append",
                round, expected_rows
            );

            let indices_before_optimize = round_cloned_dataset.load_indices().await.unwrap();
            let vector_idx_before = indices_before_optimize
                .iter()
                .find(|idx| idx.name == "vector_idx")
                .unwrap();
            let category_idx_before = indices_before_optimize
                .iter()
                .find(|idx| idx.name == "category_idx")
                .unwrap();

            // Optimize indices
            round_cloned_dataset
                .optimize_indices(&OptimizeOptions::merge(indices_before_optimize.len()))
                .await
                .unwrap();

            // Verify index UUID has changed
            let optimized_indices = round_cloned_dataset.load_indices().await.unwrap();
            let new_vector_idx = optimized_indices
                .iter()
                .find(|idx| idx.name == "vector_idx")
                .unwrap();
            let new_category_idx = optimized_indices
                .iter()
                .find(|idx| idx.name == "category_idx")
                .unwrap();

            assert_ne!(
                new_vector_idx.uuid, vector_idx_before.uuid,
                "Round {}: Vector index should have a new UUID after optimization",
                round
            );
            assert_ne!(
                new_category_idx.uuid, category_idx_before.uuid,
                "Round {}: Category index should have a new UUID after optimization",
                round
            );

            // Verify the location of index files
            use std::path::PathBuf;
            let clone_indices_dir = PathBuf::from(round_cloned_uri).join("_indices");
            let vector_index_dir = clone_indices_dir.join(new_vector_idx.uuid.to_string());
            let category_index_dir = clone_indices_dir.join(new_category_idx.uuid.to_string());

            assert!(
                vector_index_dir.exists(),
                "Round {}: New vector index directory should exist in cloned dataset location: {:?}",
                round, vector_index_dir
            );
            assert!(
                category_index_dir.exists(),
                "Round {}: New category index directory should exist in cloned dataset location: {:?}",
                round, category_index_dir
            );

            // Verify base id
            assert!(
                new_vector_idx.base_id.is_none(),
                "Round {}: New vector index should not have base_id after optimization in cloned dataset",
                round
            );
            assert!(
                new_category_idx.base_id.is_none(),
                "Round {}: New category index should not have base_id after optimization in cloned dataset",
                round
            );

            // Verify the source location does NOT contain new data
            let original_indices_dir = PathBuf::from(current_dataset.uri()).join("_indices");
            let wrong_vector_dir = original_indices_dir.join(new_vector_idx.uuid.to_string());
            let wrong_category_dir = original_indices_dir.join(new_category_idx.uuid.to_string());

            assert!(
                !wrong_vector_dir.exists(),
                "Round {}: New vector index should NOT be in original dataset location: {:?}",
                round,
                wrong_vector_dir
            );
            assert!(
                !wrong_category_dir.exists(),
                "Round {}: New category index should NOT be in original dataset location: {:?}",
                round,
                wrong_category_dir
            );

            // Validate data integrity and index functionality after optimization
            let old_category_results = round_cloned_dataset
                .scan()
                .filter("category = 'category_0'")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            let new_category_results = round_cloned_dataset
                .scan()
                .filter(&format!("category = 'category_{}'", round))
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            assert_eq!(
                source_scalar_query_rows,
                old_category_results.num_rows(),
                "Round {}: Should find old category data with {} rows",
                round,
                source_scalar_query_rows
            );
            assert!(
                new_category_results.num_rows() > 0,
                "Round {}: Should find new category data",
                round
            );

            // Test vector search functionality
            let query_vector = generate_random_array(dimensions as usize);
            let search_results = round_cloned_dataset
                .scan()
                .nearest("vector", &query_vector, 10)
                .unwrap()
                .limit(Some(10), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();

            assert!(
                search_results.num_rows() > 0,
                "Round {}: Vector search should return results after optimization",
                round
            );

            // Verify statistic of indexes
            let vector_stats: serde_json::Value = serde_json::from_str(
                &round_cloned_dataset
                    .index_statistics("vector_idx")
                    .await
                    .unwrap(),
            )
            .unwrap();
            let category_stats: serde_json::Value = serde_json::from_str(
                &round_cloned_dataset
                    .index_statistics("category_idx")
                    .await
                    .unwrap(),
            )
            .unwrap();

            assert_eq!(
                vector_stats["num_indexed_rows"].as_u64().unwrap(),
                expected_rows as u64,
                "Round {}: Vector index should have {} indexed rows",
                round,
                expected_rows
            );
            assert_eq!(
                category_stats["num_indexed_rows"].as_u64().unwrap(),
                expected_rows as u64,
                "Round {}: Category index should have {} indexed rows",
                round,
                expected_rows
            );

            // Prepare for next round: use the cloned dataset as the source for next clone
            current_dataset = round_cloned_dataset;
        }

        // Use the final cloned dataset for any remaining tests
        let final_cloned_dataset = current_dataset;

        // Verify cloned dataset has indices
        let cloned_indices = final_cloned_dataset.load_indices().await.unwrap();
        assert_eq!(
            cloned_indices.len(),
            2,
            "Final cloned dataset should have 2 indices"
        );
        let cloned_index_names: HashSet<String> =
            cloned_indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(cloned_index_names.contains("vector_idx"));
        assert!(cloned_index_names.contains("category_idx"));

        // Test vector search on final cloned dataset
        let query_vector = generate_random_array(dimensions as usize);
        let search_results = final_cloned_dataset
            .scan()
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert!(
            search_results.num_rows() > 0,
            "Vector search should return results on final dataset"
        );

        // Test scalar query on final cloned dataset
        let scalar_results = final_cloned_dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            source_scalar_query_rows,
            scalar_results.num_rows(),
            "Scalar query should return results on final dataset"
        );
    }

    #[tokio::test]
    async fn test_initialize_indices() {
        use crate::dataset::Dataset;
        use arrow_array::types::Float32Type;
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{array, BatchCount, RowCount};
        use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
        use lance_linalg::distance::MetricType;
        use std::collections::HashSet;

        // Create source dataset with various index types
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        // Generate test data using lance_datagen (need at least 256 rows for PQ training)
        let source_reader = lance_datagen::gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .col(
                "text",
                array::cycle_utf8_literals(&["hello world", "foo bar", "test data"]),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));

        // Create source dataset
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create indices on source dataset
        // 1. Vector index
        let vector_params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 10);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &vector_params,
                false,
            )
            .await
            .unwrap();

        // 2. FTS index
        let fts_params = InvertedIndexParams::default();
        source_dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_idx".to_string()),
                &fts_params,
                false,
            )
            .await
            .unwrap();

        // 3. Scalar index
        let scalar_params = ScalarIndexParams::default();
        source_dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        // Reload source dataset to get updated index metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();

        // Verify source has 3 indices
        let source_indices = source_dataset.load_indices().await.unwrap();
        assert_eq!(
            source_indices.len(),
            3,
            "Source dataset should have 3 indices"
        );

        // Create target dataset with same schema but different data (need at least 256 rows for PQ)
        let target_reader = lance_datagen::gen_batch()
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .col(
                "text",
                array::cycle_utf8_literals(&["foo bar", "test data", "hello world"]),
            )
            .col("id", array::step_custom::<Int32Type>(100, 1))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize indices from source dataset
        target_dataset
            .initialize_indices(&source_dataset)
            .await
            .unwrap();

        // Verify target has same indices
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(
            target_indices.len(),
            3,
            "Target dataset should have 3 indices after initialization"
        );

        // Check index names match
        let source_names: HashSet<String> =
            source_indices.iter().map(|idx| idx.name.clone()).collect();
        let target_names: HashSet<String> =
            target_indices.iter().map(|idx| idx.name.clone()).collect();
        assert_eq!(
            source_names, target_names,
            "Index names should match between source and target"
        );

        // Verify indices are functional by running queries
        // 1. Test vector index
        let query_vector = generate_random_array(8);
        let search_results = target_dataset
            .scan()
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .limit(Some(5), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(
            search_results.num_rows() > 0,
            "Vector index should be functional"
        );

        // 2. Test scalar index
        let scalar_results = target_dataset
            .scan()
            .filter("id = 125")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            scalar_results.num_rows(),
            1,
            "Scalar index should find exact match"
        );
    }

    #[tokio::test]
    async fn test_initialize_indices_with_missing_field() {
        use crate::dataset::Dataset;
        use arrow_array::types::Int32Type;
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{array, BatchCount, RowCount};
        use lance_index::scalar::ScalarIndexParams;

        // Test that initialize_indices handles missing fields gracefully
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        // Create source dataset with extra field
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("extra", array::cycle_utf8_literals(&["test"]))
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create index on extra field in source
        source_dataset
            .create_index(
                &["extra"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Create target dataset without extra field
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step_custom::<Int32Type>(10, 1))
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize indices should skip the index on missing field with an error
        let result = target_dataset.initialize_indices(&source_dataset).await;

        // Should fail when field is missing
        assert!(result.is_err(), "Should error when field is missing");
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not found in target dataset"));
    }

    #[tokio::test]
    async fn test_initialize_single_index() {
        use crate::dataset::Dataset;
        use crate::index::vector::VectorIndexParams;
        use arrow_array::types::{Float32Type, Int32Type};
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{array, BatchCount, RowCount};
        use lance_index::scalar::ScalarIndexParams;
        use lance_linalg::distance::MetricType;

        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/{}", test_dir, "source");
        let target_uri = format!("{}/{}", test_dir, "target");

        // Create source dataset (need at least 256 rows for PQ training)
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("name", array::rand_utf8(4.into(), false))
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create multiple indices on source
        let scalar_params = ScalarIndexParams::default();
        source_dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_index".to_string()),
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        let vector_params = VectorIndexParams::ivf_pq(16, 8, 4, MetricType::L2, 50);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_index".to_string()),
                &vector_params,
                false,
            )
            .await
            .unwrap();

        // Reload source dataset to get updated index metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("name", array::rand_utf8(4.into(), false))
            .col("vector", array::rand_vec::<Float32Type>(8.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize only the vector index
        target_dataset
            .initialize_index(&source_dataset, "vector_index")
            .await
            .unwrap();

        // Verify only vector index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Should have only 1 index");
        assert_eq!(
            target_indices[0].name, "vector_index",
            "Should have the vector index"
        );

        // Initialize the scalar index
        target_dataset
            .initialize_index(&source_dataset, "id_index")
            .await
            .unwrap();

        // Verify both indices now exist
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 2, "Should have 2 indices");

        let index_names: HashSet<String> =
            target_indices.iter().map(|idx| idx.name.clone()).collect();
        assert!(
            index_names.contains("vector_index"),
            "Should have vector index"
        );
        assert!(index_names.contains("id_index"), "Should have id index");

        // Test error case - non-existent index
        let result = target_dataset
            .initialize_index(&source_dataset, "non_existent")
            .await;
        assert!(result.is_err(), "Should error for non-existent index");
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not found in source dataset"));
    }

    #[tokio::test]
    async fn test_vector_index_on_nested_field_with_dots() {
        let dimensions = 16;
        let num_rows = 256;

        // Create schema with nested field containing dots in the name
        let struct_field = Field::new(
            "embedding_data",
            DataType::Struct(
                vec![
                    Field::new(
                        "vector.v1", // Field name with dot
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dimensions,
                        ),
                        false,
                    ),
                    Field::new(
                        "vector.v2", // Another field name with dot
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dimensions,
                        ),
                        false,
                    ),
                    Field::new("metadata", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            struct_field,
        ]));

        // Generate test data
        let float_arr_v1 = generate_random_array(num_rows * dimensions as usize);
        let vectors_v1 = FixedSizeListArray::try_new_from_values(float_arr_v1, dimensions).unwrap();

        let float_arr_v2 = generate_random_array(num_rows * dimensions as usize);
        let vectors_v2 = FixedSizeListArray::try_new_from_values(float_arr_v2, dimensions).unwrap();

        let ids = Int32Array::from_iter_values(0..num_rows as i32);
        let metadata = StringArray::from_iter_values((0..num_rows).map(|i| format!("meta_{}", i)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "vector.v1",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                )),
                Arc::new(vectors_v1) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new(
                    "vector.v2",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                )),
                Arc::new(vectors_v2) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("metadata", DataType::Utf8, false)),
                Arc::new(metadata) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Test creating index on nested field with dots using quoted syntax
        let nested_column_path_v1 = "embedding_data.`vector.v1`";
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);

        dataset
            .create_index(
                &[nested_column_path_v1],
                IndexType::Vector,
                Some("vec_v1_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vec_v1_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "embedding_data.`vector.v1`");

        // Test creating index on the second vector field with dots
        let nested_column_path_v2 = "embedding_data.`vector.v2`";
        dataset
            .create_index(
                &[nested_column_path_v2],
                IndexType::Vector,
                Some("vec_v2_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify both indices exist
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);

        // Verify we can query using the indexed fields and check the plan
        let query_vector = generate_random_array(dimensions as usize);

        // Check the query plan for the first vector field
        let plan_v1 = dataset
            .scan()
            .nearest(nested_column_path_v1, &query_vector, 5)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the vector index is being used (should show ANNSubIndex or ANNIvfPartition)
        assert!(
            plan_v1.contains("ANNSubIndex") || plan_v1.contains("ANNIvfPartition"),
            "Query plan should use vector index for nested field with dots. Plan: {}",
            plan_v1
        );

        let search_results_v1 = dataset
            .scan()
            .nearest(nested_column_path_v1, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(search_results_v1.num_rows(), 5);

        // Check the query plan for the second vector field
        let plan_v2 = dataset
            .scan()
            .nearest(nested_column_path_v2, &query_vector, 5)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the vector index is being used
        assert!(
            plan_v2.contains("ANNSubIndex") || plan_v2.contains("ANNIvfPartition"),
            "Query plan should use vector index for second nested field with dots. Plan: {}",
            plan_v2
        );

        let search_results_v2 = dataset
            .scan()
            .nearest(nested_column_path_v2, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(search_results_v2.num_rows(), 5);
    }

    #[tokio::test]
    async fn test_vector_index_on_simple_nested_field() {
        // This test reproduces the Python test scenario from test_nested_field_vector_index
        // where the nested field path is simple (data.embedding) without dots in field names
        let dimensions = 16;
        let num_rows = 256;

        // Create schema with simple nested field (no dots in field names)
        let struct_field = Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new(
                        "embedding",
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::Float32, true)),
                            dimensions,
                        ),
                        false,
                    ),
                    Field::new("label", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            struct_field,
        ]));

        // Generate test data
        let float_arr = generate_random_array(num_rows * dimensions as usize);
        let vectors = FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let ids = Int32Array::from_iter_values(0..num_rows as i32);
        let labels = StringArray::from_iter_values((0..num_rows).map(|i| format!("label_{}", i)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new(
                    "embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                )),
                Arc::new(vectors) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("label", DataType::Utf8, false)),
                Arc::new(labels) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Test creating index on nested field
        let nested_column_path = "data.embedding";
        let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vec_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "data.embedding");

        // Test querying with the index
        let query_vector = generate_random_array(dimensions as usize);

        let plan = dataset
            .scan()
            .nearest(nested_column_path, &query_vector, 5)
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the vector index is being used
        assert!(
            plan.contains("ANNSubIndex") || plan.contains("ANNIvfPartition"),
            "Query plan should use vector index for nested field. Plan: {}",
            plan
        );

        let search_results = dataset
            .scan()
            .nearest(nested_column_path, &query_vector, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(search_results.num_rows(), 5);
    }

    #[tokio::test]
    async fn test_btree_index_on_nested_field_with_dots() {
        // Test creating BTree index on nested field with dots in the name
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create schema with nested field containing dots
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "data",
                DataType::Struct(
                    vec![
                        Field::new("value.v1", DataType::Int32, false),
                        Field::new("value.v2", DataType::Float32, false),
                        Field::new("text", DataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]));

        // Generate test data
        let num_rows = 1000;
        let ids = Int32Array::from_iter_values(0..num_rows);
        let values_v1 = Int32Array::from_iter_values((0..num_rows).map(|i| i % 100));
        let values_v2 = Float32Array::from_iter_values((0..num_rows).map(|i| (i as f32) * 0.1));
        let texts = StringArray::from_iter_values((0..num_rows).map(|i| format!("text_{}", i)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new("value.v1", DataType::Int32, false)),
                Arc::new(values_v1) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("value.v2", DataType::Float32, false)),
                Arc::new(values_v2) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("text", DataType::Utf8, false)),
                Arc::new(texts) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create BTree index on nested field with dots
        let nested_column_path = "data.`value.v1`";
        let params = ScalarIndexParams::default();

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::BTree,
                Some("btree_v1_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload dataset to ensure index is loaded
        dataset = Dataset::open(test_uri).await.unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "btree_v1_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "data.`value.v1`");

        // Test querying with the index and verify it's being used
        let plan = dataset
            .scan()
            .filter("data.`value.v1` = 42")
            .unwrap()
            .prefilter(true)
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the query plan (scalar indices on nested fields may use optimized filters)
        // The index may be used internally even if not shown as ScalarIndexQuery
        assert!(
            plan.contains("ScalarIndexQuery"),
            "Query plan should show optimized read. Plan: {}",
            plan
        );

        // Also test that the query returns results
        let results = dataset
            .scan()
            .filter("data.`value.v1` = 42")
            .unwrap()
            .prefilter(true)
            .try_into_batch()
            .await
            .unwrap();

        assert!(results.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_bitmap_index_on_nested_field_with_dots() {
        // Test creating Bitmap index on nested field with dots in the name
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create schema with nested field containing dots - using low cardinality for bitmap
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "metadata",
                DataType::Struct(
                    vec![
                        Field::new("status.code", DataType::Int32, false),
                        Field::new("category.name", DataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]));

        // Generate test data with low cardinality (good for bitmap index)
        let num_rows = 1000;
        let ids = Int32Array::from_iter_values(0..num_rows);
        // Only 10 unique status codes
        let status_codes = Int32Array::from_iter_values((0..num_rows).map(|i| i % 10));
        // Only 5 unique categories
        let categories =
            StringArray::from_iter_values((0..num_rows).map(|i| format!("category_{}", i % 5)));

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new("status.code", DataType::Int32, false)),
                Arc::new(status_codes) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("category.name", DataType::Utf8, false)),
                Arc::new(categories) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create Bitmap index on nested field with dots
        let nested_column_path = "metadata.`status.code`";
        let params = ScalarIndexParams::default();

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::Bitmap,
                Some("bitmap_status_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload dataset to ensure index is loaded
        dataset = Dataset::open(test_uri).await.unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "bitmap_status_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "metadata.`status.code`");

        // Test querying with the index and verify it's being used
        let plan = dataset
            .scan()
            .filter("metadata.`status.code` = 5")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the query plan (scalar indices on nested fields may use optimized filters)
        // The index may be used internally even if not shown as ScalarIndexQuery
        assert!(
            plan.contains("ScalarIndexQuery"),
            "Query plan should show optimized read. Plan: {}",
            plan
        );

        // Also test that the query returns results
        let results = dataset
            .scan()
            .filter("metadata.`status.code` = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        // Should have ~100 rows with status code 5
        assert!(results.num_rows() > 0);
        assert_eq!(results.num_rows(), 100);
    }

    #[tokio::test]
    async fn test_inverted_index_on_nested_field_with_dots() {
        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

        // Test creating Inverted index on nested text field with dots in the name
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create schema with nested text field containing dots
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "document",
                DataType::Struct(
                    vec![
                        Field::new("content.text", DataType::Utf8, false),
                        Field::new("content.summary", DataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]));

        // Generate test data with text content
        let num_rows = 100;
        let ids = Int32Array::from_iter_values(0..num_rows as i32);
        let content_texts = StringArray::from_iter_values((0..num_rows).map(|i| match i % 3 {
            0 => format!("The quick brown fox jumps over the lazy dog {}", i),
            1 => format!(
                "Machine learning and artificial intelligence document {}",
                i
            ),
            _ => format!("Data science and analytics content piece {}", i),
        }));
        let summaries = StringArray::from_iter_values(
            (0..num_rows).map(|i| format!("Summary of document {}", i)),
        );

        let struct_array = arrow_array::StructArray::from(vec![
            (
                Arc::new(Field::new("content.text", DataType::Utf8, false)),
                Arc::new(content_texts) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("content.summary", DataType::Utf8, false)),
                Arc::new(summaries) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)])
                .unwrap();

        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        // Create Inverted index on nested text field with dots
        let nested_column_path = "document.`content.text`";
        let params = InvertedIndexParams::default();

        dataset
            .create_index(
                &[nested_column_path],
                IndexType::Inverted,
                Some("inverted_content_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reload the dataset to ensure the index is loaded
        dataset = Dataset::open(test_uri).await.unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "inverted_content_idx");

        // Verify the correct field was indexed
        let field_id = indices[0].fields[0];
        let field_path = dataset.schema().field_path(field_id).unwrap();
        assert_eq!(field_path, "document.`content.text`");

        // Test full-text search on the nested field with dots
        // Use the field_path that the index reports
        let query = FullTextSearchQuery::new("machine learning".to_string())
            .with_column(field_path.clone())
            .unwrap();

        // Check the query plan uses the inverted index
        let plan = dataset
            .scan()
            .full_text_search(query.clone())
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // Verify the inverted index is being used
        assert!(
            plan.contains("MatchQuery") || plan.contains("PhraseQuery"),
            "Query plan should use inverted index for nested field with dots. Plan: {}",
            plan
        );

        let results = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        // Verify we get results from the full-text search
        assert!(
            !results.is_empty(),
            "Full-text search should return results"
        );

        // Check that we found documents containing "machine learning"
        let mut found_count = 0;
        for batch in results {
            found_count += batch.num_rows();
        }
        // We expect to find approximately 1/3 of documents (those with i % 3 == 1)
        assert!(
            found_count > 0,
            "Should find at least some documents with 'machine learning'"
        );
        assert!(found_count < num_rows, "Should not match all documents");
    }
}
