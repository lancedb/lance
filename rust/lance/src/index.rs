// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary Index
//!

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use arrow_schema::{DataType, Schema};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::utils::address::RowAddress;
use lance_core::utils::parse::str_is_truthy;
use lance_core::utils::tracing::{
    IO_TYPE_OPEN_FRAG_REUSE, IO_TYPE_OPEN_SCALAR, IO_TYPE_OPEN_VECTOR, TRACE_IO_EVENTS,
};
use lance_file::reader::FileReader;
use lance_file::v2;
use lance_file::v2::reader::FileReaderOptions;
use lance_index::frag_reuse::{FragReuseIndex, FRAG_REUSE_INDEX_NAME};
use lance_index::pb::index::Implementation;
use lance_index::scalar::expression::{
    FtsQueryParser, IndexInformationProvider, LabelListQueryParser, MultiQueryParser,
    SargableQueryParser, ScalarQueryParser, TextQueryParser,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{ScalarIndex, ScalarIndexType};
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::sq::ScalarQuantizer;
pub use lance_index::IndexParams;
use lance_index::{
    metrics::{MetricsCollector, NoOpMetricsCollector},
    scalar::inverted::tokenizer::InvertedIndexParams,
};
use lance_index::{optimize::OptimizeOptions, scalar::inverted::train_inverted_index};
use lance_index::{
    pb,
    scalar::{ScalarIndexParams, LANCE_SCALAR_INDEX},
    vector::VectorIndex,
    DatasetIndexExt, Index, IndexType, INDEX_FILE_NAME,
};
use lance_index::{ScalarIndexCriteria, INDEX_METADATA_SCHEMA_KEY};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Reader;
use lance_io::utils::{
    read_last_block, read_message, read_message_from_buf, read_metadata_offset, read_version,
    CachedFileSize,
};
use lance_table::format::Index as IndexMetadata;
use lance_table::format::{Fragment, SelfDescribingFileReader};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringBitmap;
use scalar::{
    build_inverted_index, detect_scalar_index_type, index_matches_criteria, infer_index_type,
    inverted_index_details, TrainingRequest,
};
use serde_json::json;
use snafu::location;
use tracing::{info, instrument};
use uuid::Uuid;
use vector::ivf::v2::IVFIndex;
use vector::utils::get_vector_type;

pub(crate) mod append;
pub(crate) mod cache;
pub mod frag_reuse;
pub mod prefilter;
pub mod scalar;
pub mod vector;

use crate::dataset::index::LanceIndexStoreExt;
pub use crate::index::prefilter::{FilterLoader, PreFilter};

use self::append::merge_indices;
use self::scalar::build_scalar_index;
use self::vector::{build_vector_index, VectorIndexParams, LANCE_VECTOR_INDEX};
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
use crate::index::vector::remap_vector_index;
use crate::{dataset::Dataset, Error, Result};

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
) -> Result<Uuid> {
    // Load indices from the disk.
    let indices = dataset.load_indices().await?;
    let matched = indices
        .iter()
        .find(|i| i.uuid == *index_id)
        .ok_or_else(|| Error::Index {
            message: format!("Index with id {index_id} does not exist"),
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
            return Ok(*index_id);
        }
    }

    let field = matched
        .fields
        .first()
        .expect("An index existed with no fields");

    let field = dataset.schema().field_by_id(*field).unwrap();

    let new_id = Uuid::new_v4();

    let generic = dataset
        .open_generic_index(&field.name, &index_id.to_string(), &NoOpMetricsCollector)
        .await?;

    match generic.index_type() {
        it if it.is_scalar() => {
            let new_store = LanceIndexStore::from_dataset(dataset, &new_id.to_string());

            let scalar_index = dataset
                .open_scalar_index(&field.name, &index_id.to_string(), &NoOpMetricsCollector)
                .await?;

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
                            field.name
                        );
                        let training_request = Box::new(TrainingRequest::new(
                            Arc::new(dataset.clone()),
                            field.name.clone(),
                        ));
                        train_inverted_index(
                            training_request,
                            &new_store,
                            inverted_index.params().clone(),
                        )
                        .await?;
                    } else {
                        scalar_index.remap(row_id_map, &new_store).await?;
                    }
                }
                _ => scalar_index.remap(row_id_map, &new_store).await?,
            };
        }
        it if it.is_vector() => {
            remap_vector_index(
                Arc::new(dataset.clone()),
                &field.name,
                index_id,
                &new_id,
                matched,
                row_id_map,
            )
            .await?;
        }
        _ => {
            return Err(Error::Index {
                message: format!("Index type {} is not supported", generic.index_type()),
                location: location!(),
            });
        }
    }

    Ok(new_id)
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
    #[instrument(skip_all)]
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<()> {
        if columns.len() != 1 {
            return Err(Error::Index {
                message: "Only support building index on 1 column at the moment".to_string(),
                location: location!(),
            });
        }
        let column = columns[0];
        let Some(field) = self.schema().field(column) else {
            return Err(Error::Index {
                message: format!("CreateIndex: column '{column}' does not exist"),
                location: location!(),
            });
        };

        // Load indices from the disk.
        let indices = self.load_indices().await?;
        let fri = self.open_frag_reuse_index(&NoOpMetricsCollector).await?;
        let index_name = name.unwrap_or(format!("{column}_idx"));
        if let Some(idx) = indices.iter().find(|i| i.name == index_name) {
            if idx.fields == [field.id] && !replace {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name} already exists, \
                        please specify a different name or use replace=True"
                    ),
                    location: location!(),
                });
            };
            if idx.fields != [field.id] {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name} already exists with different fields, \
                        please specify a different name"
                    ),
                    location: location!(),
                });
            }
        }

        let index_id = Uuid::new_v4();
        let index_details = match (index_type, params.index_name()) {
            (
                IndexType::Bitmap
                | IndexType::BTree
                | IndexType::Inverted
                | IndexType::NGram
                | IndexType::LabelList,
                LANCE_SCALAR_INDEX,
            ) => {
                let params = ScalarIndexParams::new(index_type.try_into()?);
                build_scalar_index(self, column, &index_id.to_string(), &params).await?
            }
            (IndexType::Scalar, LANCE_SCALAR_INDEX) => {
                // Guess the index type
                let params = params
                    .as_any()
                    .downcast_ref::<ScalarIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Scalar index type must take a ScalarIndexParams".to_string(),
                        location: location!(),
                    })?;
                build_scalar_index(self, column, &index_id.to_string(), params).await?
            }
            (IndexType::Inverted, _) => {
                // Inverted index params.
                let inverted_params = params
                    .as_any()
                    .downcast_ref::<InvertedIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Inverted index type must take a InvertedIndexParams".to_string(),
                        location: location!(),
                    })?;

                build_inverted_index(self, column, &index_id.to_string(), inverted_params).await?;
                inverted_index_details()
            }
            (IndexType::Vector, LANCE_VECTOR_INDEX) => {
                // Vector index params.
                let vec_params = params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| Error::Index {
                        message: "Vector index type must take a VectorIndexParams".to_string(),
                        location: location!(),
                    })?;

                // this is a large future so move it to heap
                Box::pin(build_vector_index(
                    self,
                    column,
                    &index_name,
                    &index_id.to_string(),
                    vec_params,
                    fri,
                ))
                .await?;
                vector_index_details()
            }
            // Can't use if let Some(...) here because it's not stable yet.
            // TODO: fix after https://github.com/rust-lang/rust/issues/51114
            (IndexType::Vector, name)
                if self
                    .session
                    .index_extensions
                    .contains_key(&(IndexType::Vector, name.to_string())) =>
            {
                let ext = self
                    .session
                    .index_extensions
                    .get(&(IndexType::Vector, name.to_string()))
                    .expect("already checked")
                    .clone()
                    .to_vector()
                    // this should never happen because we control the registration
                    // if this fails, the registration logic has a bug
                    .ok_or(Error::Internal {
                        message: "unable to cast index extension to vector".to_string(),
                        location: location!(),
                    })?;

                ext.create_index(self, column, &index_id.to_string(), params)
                    .await?;
                vector_index_details()
            }
            (IndexType::FragmentReuse, _) => {
                return Err(Error::Index {
                    message: "Fragment reuse index can only be created through compaction"
                        .to_string(),
                    location: location!(),
                })
            }
            (index_type, index_name) => {
                return Err(Error::Index {
                    message: format!(
                        "Index type {index_type} with name {index_name} is not supported"
                    ),
                    location: location!(),
                });
            }
        };

        let new_idx = IndexMetadata {
            uuid: index_id,
            name: index_name,
            fields: vec![field.id],
            dataset_version: self.manifest.version,
            fragment_bitmap: Some(self.get_fragments().iter().map(|f| f.id() as u32).collect()),
            index_details: Some(index_details),
            index_version: index_type.version(),
            created_at: Some(chrono::Utc::now()),
        };
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            /*blobs_op= */ None,
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn drop_index(&mut self, name: &str) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::IndexNotFound {
                identity: format!("name={name}"),
                location: location!(),
            });
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: indices.clone(),
            },
            /*blobs_op= */ None,
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
                identity: format!("name={name}"),
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
        let indices = match self
            .session
            .index_cache
            .get_metadata(self.base.as_ref(), self.version().version)
        {
            Some(indices) => indices,
            None => {
                let loaded_indices = read_manifest_indexes(
                    &self.object_store,
                    &self.manifest_location,
                    &self.manifest,
                )
                .await?;
                let loaded_indices = Arc::new(loaded_indices);
                self.session.index_cache.insert_metadata(
                    self.base.as_ref(),
                    self.version().version,
                    loaded_indices.clone(),
                );
                loaded_indices
            }
        };

        let mut indices = indices
            .iter()
            .filter(|idx| {
                let max_valid_version = infer_index_type(idx)
                    .map(|t| t.version())
                    .unwrap_or_default();
                let is_valid = idx.index_version <= max_valid_version;
                if !is_valid {
                    log::warn!(
                        "Index {} has version {}, which is not supported (<={}), ignoring it",
                        idx.name,
                        idx.index_version,
                        max_valid_version,
                    );
                }
                is_valid
            })
            .cloned()
            .collect::<Vec<_>>();

        if let Some(fri_meta) = indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME) {
            let uuid = fri_meta.uuid.to_string();
            let fri = if let Some(index) = self.session.index_cache.get_frag_reuse(&uuid) {
                log::debug!("Found fragment reuse index in cache uuid: {uuid}");
                index
            } else {
                let index_details = load_frag_reuse_index_details(self, fri_meta).await?;
                let index = open_frag_reuse_index(index_details.as_ref()).await?;
                self.session
                    .index_cache
                    .insert_frag_reuse(&uuid, index.clone());
                index
            };
            indices.iter_mut().for_each(|idx| {
                if let Some(bitmap) = idx.fragment_bitmap.as_mut() {
                    fri.remap_fragment_bitmap(bitmap).unwrap();
                }
            });
        }

        Ok(Arc::new(indices))
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
        };

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            /*blobs_op= */ None,
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
                    if index_matches_criteria(idx, &criteria, field, has_multiple)? {
                        return Ok(Some(idx.clone()));
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
                    && idx.name != FRAG_REUSE_INDEX_NAME
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
                index_details: last_idx.index_details.clone(),
                index_version: res.new_index_version,
                created_at: Some(chrono::Utc::now()),
            };
            removed_indices.extend(res.removed_indices.iter().map(|&idx| idx.clone()));
            if deltas.len() > removed_indices.len() {
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
            /*blobs_op= */ None,
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
                identity: format!("name={index_name}"),
                location: location!(),
            });
        }

        let column = self
            .schema()
            .field_by_id(metadatas[0].fields[0])
            .map(|f| f.name.as_str())
            .ok_or(Error::IndexNotFound {
                identity: index_name.to_string(),
                location: location!(),
            })?;

        // Open all delta indices
        let indices = stream::iter(metadatas.iter())
            .then(|m| async move {
                self.open_generic_index(column, &m.uuid.to_string(), &NoOpMetricsCollector)
                    .await
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
                            To disable migration, set LANCE_AUTO_MIGRATION=false. Original error: {err}"),
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
            message: format!("Failed to serialize index statistics: {e}"),
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
                identity: format!("name={index_name}"),
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

    /// Loads information about all the available scalar indices on the dataset
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo>;

    /// Return the fragments that are not covered by any of the deltas of the index.
    async fn unindexed_fragments(&self, idx_name: &str) -> Result<Vec<Fragment>>;

    /// Return the fragments that are covered by each of the deltas of the index.
    async fn indexed_fragments(&self, idx_name: &str) -> Result<Vec<Vec<Fragment>>>;
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
        if let Some(index) = self.session.index_cache.get_scalar(uuid) {
            return Ok(index.as_index());
        }
        if let Some(index) = self.session.index_cache.get_vector(uuid) {
            return Ok(index.as_index());
        }
        if let Some(index) = self.session.index_cache.get_frag_reuse(uuid) {
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
        let index_dir = self.indices_dir().child(uuid);
        let index_file = index_dir.child(INDEX_FILE_NAME);
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
        let cache_key = index_cache_key(self, uuid).await.unwrap();
        if let Some(index) = self.session.index_cache.get_scalar(cache_key.as_ref()) {
            return Ok(index);
        }

        let index_meta = self.load_index(uuid).await?.ok_or_else(|| Error::Index {
            message: format!("Index with id {uuid} does not exist"),
            location: location!(),
        })?;

        let index = scalar::open_scalar_index(self, column, &index_meta, metrics).await?;

        info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_SCALAR, index_type=index.index_type().to_string());
        metrics.record_index_load();

        self.session.index_cache.insert_scalar(uuid, index.clone());
        Ok(index)
    }

    async fn open_vector_index(
        &self,
        column: &str,
        uuid: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndex>> {
        let cache_key = index_cache_key(self, uuid).await.unwrap();
        if let Some(index) = self.session.index_cache.get_vector(cache_key.as_ref()) {
            log::debug!("Found vector index in cache uuid: {uuid}");
            return Ok(index);
        }

        let fri = self.open_frag_reuse_index(metrics).await?;
        let index_dir = self.indices_dir().child(uuid);
        let index_file = index_dir.child(INDEX_FILE_NAME);
        let reader: Arc<dyn Reader> = self.object_store.open(&index_file).await?.into();

        let tailing_bytes = read_last_block(reader.as_ref()).await?;
        let (major_version, minor_version) = read_version(&tailing_bytes)?;

        // the index file is in lance format since version (0,2)
        // TODO: we need to change the legacy IVF_PQ to be in lance format
        let index = match (major_version, minor_version) {
            (0, 1) | (0, 0) => {
                info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.1", index_type="IVF_PQ");
                let proto = open_index_proto(reader.as_ref()).await?;
                match &proto.implementation {
                    Some(Implementation::VectorIndex(vector_index)) => {
                        let dataset = Arc::new(self.clone());
                        vector::open_vector_index(dataset, uuid, vector_index, reader, fri).await
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
                    Some(&self.metadata_cache),
                )
                .await?;
                vector::open_vector_index_v2(Arc::new(self.clone()), column, uuid, reader, fri)
                    .await
            }

            (0, 3) => {
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
                    &self.metadata_cache,
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
                    message: format!("Column {column} does not exist in the schema"),
                    location: location!(),
                })?;

                let (_, element_type) = get_vector_type(self.schema(), column)?;

                info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_VECTOR, version="0.3", index_type=index_metadata.index_type);

                match index_metadata.index_type.as_str() {
                    "IVF_FLAT" => match element_type {
                        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                            let ivf = IVFIndex::<FlatIndex, FlatQuantizer>::try_new(
                                self.object_store.clone(),
                                self.indices_dir(),
                                uuid.to_owned(),
                                Arc::downgrade(&self.session),
                                fri,
                            )
                            .await?;
                            Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                        }
                        DataType::UInt8 => {
                            let ivf = IVFIndex::<FlatIndex, FlatBinQuantizer>::try_new(
                                self.object_store.clone(),
                                self.indices_dir(),
                                uuid.to_owned(),
                                Arc::downgrade(&self.session),
                                fri,
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
                            self.indices_dir(),
                            uuid.to_owned(),
                            Arc::downgrade(&self.session),
                            fri,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_HNSW_FLAT" => {
                        let ivf = IVFIndex::<HNSW, FlatQuantizer>::try_new(
                            self.object_store.clone(),
                            self.indices_dir(),
                            uuid.to_owned(),
                            Arc::downgrade(&self.session),
                            fri,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_HNSW_SQ" => {
                        let ivf = IVFIndex::<HNSW, ScalarQuantizer>::try_new(
                            self.object_store.clone(),
                            self.indices_dir(),
                            uuid.to_owned(),
                            Arc::downgrade(&self.session),
                            fri,
                        )
                        .await?;
                        Ok(Arc::new(ivf) as Arc<dyn VectorIndex>)
                    }

                    "IVF_HNSW_PQ" => {
                        let ivf = IVFIndex::<HNSW, ProductQuantizer>::try_new(
                            self.object_store.clone(),
                            self.indices_dir(),
                            uuid.to_owned(),
                            Arc::downgrade(&self.session),
                            fri,
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
        self.session.index_cache.insert_vector(uuid, index.clone());
        Ok(index)
    }

    async fn open_frag_reuse_index(
        &self,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<Arc<FragReuseIndex>>> {
        if let Some(fri_meta) = self.load_index_by_name(FRAG_REUSE_INDEX_NAME).await? {
            let uuid = fri_meta.uuid.to_string();
            if let Some(index) = self.session.index_cache.get_frag_reuse(&uuid) {
                log::debug!("Found vector index in cache uuid: {uuid}");
                return Ok(Some(index));
            }

            let index_meta = self.load_index(&uuid).await?.ok_or_else(|| Error::Index {
                message: format!("Index with id {uuid} does not exist"),
                location: location!(),
            })?;
            let index_details = load_frag_reuse_index_details(self, &index_meta).await?;
            let index = open_frag_reuse_index(index_details.as_ref()).await?;

            info!(target: TRACE_IO_EVENTS, index_uuid=uuid, r#type=IO_TYPE_OPEN_FRAG_REUSE);
            metrics.record_index_load();

            self.session
                .index_cache
                .insert_frag_reuse(&uuid, index.clone());
            Ok(Some(index))
        } else {
            Ok(None)
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

            idx.fields.len() == 1 && !is_vector_index
        }) {
            let field = index.fields[0];
            let field = schema.field_by_id(field).ok_or_else(|| Error::Internal {
                message: format!(
                    "Index referenced a field with id {field} which did not exist in the schema"
                ),
                location: location!(),
            })?;

            let query_parser = match field.data_type() {
                DataType::List(_) => Box::new(LabelListQueryParser::new(index.name.clone()))
                    as Box<dyn ScalarQueryParser>,
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let index_type =
                        detect_scalar_index_type(self, index, &field.name, self.session.as_ref())
                            .await?;
                    match index_type {
                        ScalarIndexType::BTree | ScalarIndexType::Bitmap => {
                            Box::new(SargableQueryParser::new(index.name.clone()))
                                as Box<dyn ScalarQueryParser>
                        }
                        ScalarIndexType::NGram => {
                            Box::new(TextQueryParser::new(index.name.clone()))
                                as Box<dyn ScalarQueryParser>
                        }
                        ScalarIndexType::Inverted => {
                            Box::new(FtsQueryParser::new(index.name.clone()))
                                as Box<dyn ScalarQueryParser>
                        }
                        _ => continue,
                    }
                }
                _ => Box::new(SargableQueryParser::new(index.name.clone()))
                    as Box<dyn ScalarQueryParser>,
            };

            indexed_fields.push((field.name.clone(), (field.data_type(), query_parser)));
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

/// Index cache key should be the ID of the index plus the ID of the FRI.
/// If FRI has changed, the index would have changed further
async fn index_cache_key<'a>(dataset: &Dataset, index_id: &'a str) -> Result<Cow<'a, str>> {
    if let Some(fri) = dataset.load_index_by_name(FRAG_REUSE_INDEX_NAME).await? {
        Ok(Cow::Owned(format!("{}-{}", index_id, fri.uuid)))
    } else {
        Ok(Cow::Borrowed(index_id))
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{compact_files, CompactionOptions};
    use crate::dataset::{ReadParams, WriteParams};
    use crate::session::Session;
    use crate::utils::test::{
        copy_test_data_to_tmp, DatagenExt, FragmentCount, FragmentRowCount, StatsHolder,
    };
    use arrow_array::Int32Array;

    use super::*;

    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, Int32Type};
    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{Field, Schema};
    use lance_arrow::*;
    use lance_datagen::r#gen;
    use lance_datagen::{array, BatchCount, Dimension, RowCount};
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::vector::{
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, sq::builder::SQBuildParams,
    };
    use lance_io::object_store::ObjectStoreParams;
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_testing::datagen::generate_random_array;
    use rstest::rstest;
    use std::collections::HashSet;
    use tempfile::tempdir;

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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
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
        let test_dir = tempdir().unwrap();
        let schema = Schema::new(vec![
            sample_vector_field(),
            Field::new("ints", DataType::Int32, false),
        ]);
        let mut dataset = lance_datagen::rand(&schema)
            .into_dataset(
                test_dir.path().to_str().unwrap(),
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
        let test_dir = tempdir().unwrap();
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
        let test_uri = test_dir.path().to_str().unwrap();
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
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 1, // merge the index with new data
                ..Default::default()
            })
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
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 2,
                ..Default::default()
            })
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
        let test_dir = tempdir().unwrap();
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

        let test_uri = test_dir.path().to_str().unwrap();
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
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 0, // Just create index for delta
                ..Default::default()
            })
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
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 2,
                ..Default::default()
            })
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

        let dir = tempdir().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let data = StringArray::from_iter_values(words.iter().map(|s| s.to_string()));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
        let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(batch_iterator, dir.path().to_str().unwrap(), None)
            .await
            .unwrap();

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

            assert_eq!(texts.len(), 1, "query: {word}, texts: {texts:?}");
            assert_eq!(texts[0], word, "query: {word}, texts: {texts:?}");
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

            assert_eq!(texts.len(), 1, "query: {word}, texts: {texts:?}");
            assert_eq!(texts[0], word, "query: {word}, texts: {texts:?}");

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
                assert_eq!(texts.len(), 1, "query: {word}, texts: {texts:?}");
                assert_eq!(texts[0], word, "query: {word}, texts: {texts:?}");
            }
            assert_indexed_rows(&dataset, num_rows).await;
        }
    }

    #[tokio::test]
    async fn test_create_index_too_small_for_pq() {
        let test_dir = tempdir().unwrap();
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

        let test_uri = test_dir.path().to_str().unwrap();
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
        let test_dir = tempdir().unwrap();
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = test_dir.path().to_str().unwrap();
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

    #[tokio::test]
    async fn test_load_indices() {
        let session = Arc::new(Session::default());
        let io_stats = Arc::new(StatsHolder::default());
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(io_stats.clone()),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        };

        let test_dir = tempdir().unwrap();
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![field]));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let reader = RecordBatchIterator::new(
            vec![record_batch.clone()].into_iter().map(Ok),
            schema.clone(),
        );

        let test_uri = test_dir.path().to_str().unwrap();
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
        io_stats.incremental_stats(); // Reset

        let indices = dataset.load_indices().await.unwrap();
        let stats = io_stats.incremental_stats();
        // We should already have this cached since we just wrote it.
        assert_eq!(stats.read_iops, 0);
        assert_eq!(stats.read_bytes, 0);
        assert_eq!(indices.len(), 1);

        session.index_cache.clear(); // Clear the cache

        let dataset2 = DatasetBuilder::from_uri(test_uri)
            .with_session(session.clone())
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_stats.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        let stats = io_stats.incremental_stats(); // Reset
        assert!(stats.read_bytes < 64 * 1024);

        // Because the manifest is so small, we should have opportunistically
        // cached the indices in memory already.
        let indices2 = dataset2.load_indices().await.unwrap();
        let stats = io_stats.incremental_stats();
        assert_eq!(stats.read_iops, 0);
        assert_eq!(stats.read_bytes, 0);
        assert_eq!(indices2.len(), 1);
    }

    #[tokio::test]
    async fn test_remap_empty() {
        let data = gen()
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
        assert_eq!(new_uuid, index_uuid);
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
            assert!(seen.insert(*id), "Duplicate id found: {id}");
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
            assert!(seen.insert(*id), "Duplicate id found: {id}");
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
        let test_uri = test_dir.path().to_str().unwrap();

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
}
