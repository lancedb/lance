// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Secondary Index
//!

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_file::reader::FileReader;
use lance_file::v2;
use lance_file::v2::reader::FileReaderOptions;
use lance_index::optimize::OptimizeOptions;
use lance_index::pb::index::Implementation;
use lance_index::scalar::expression::{
    IndexInformationProvider, LabelListQueryParser, SargableQueryParser, ScalarQueryParser,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{InvertedIndexParams, ScalarIndex, ScalarIndexType};
use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::sq::ScalarQuantizer;
pub use lance_index::IndexParams;
use lance_index::INDEX_METADATA_SCHEMA_KEY;
use lance_index::{
    pb,
    scalar::{ScalarIndexParams, LANCE_SCALAR_INDEX},
    vector::VectorIndex,
    DatasetIndexExt, Index, IndexType, INDEX_FILE_NAME,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Reader;
use lance_io::utils::{
    read_last_block, read_message, read_message_from_buf, read_metadata_offset, read_version,
};
use lance_table::format::Index as IndexMetadata;
use lance_table::format::{Fragment, SelfDescribingFileReader};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringBitmap;
use scalar::{build_inverted_index, detect_scalar_index_type};
use serde_json::json;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;
use vector::ivf::v2::IVFIndex;

pub(crate) mod append;
pub(crate) mod cache;
pub mod prefilter;
pub mod scalar;
pub mod vector;

use crate::dataset::index::LanceIndexStoreExt;
pub use crate::index::prefilter::{FilterLoader, PreFilter};

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::vector::remap_vector_index;
use crate::io::commit::commit_transaction;
use crate::{dataset::Dataset, Error, Result};

use self::append::merge_indices;
use self::scalar::build_scalar_index;
use self::vector::{build_vector_index, VectorIndexParams, LANCE_VECTOR_INDEX};

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
            message: format!("Index with id {} does not exist", index_id),
            location: location!(),
        })?;

    if matched.fields.len() > 1 {
        return Err(Error::Index {
            message: "Remapping indices with multiple fields is not supported".to_string(),
            location: location!(),
        });
    }
    let field = matched
        .fields
        .first()
        .expect("An index existed with no fields");

    let field = dataset.schema().field_by_id(*field).unwrap();

    let new_id = Uuid::new_v4();

    let generic = dataset
        .open_generic_index(&field.name, &index_id.to_string())
        .await?;

    match generic.index_type() {
        it if it.is_scalar() => {
            let new_store = match it {
                IndexType::Scalar | IndexType::BTree => {
                    LanceIndexStore::from_dataset(dataset, &new_id.to_string())
                        .with_legacy_format(true)
                }
                _ => LanceIndexStore::from_dataset(dataset, &new_id.to_string()),
            };

            let scalar_index = dataset
                .open_scalar_index(&field.name, &index_id.to_string())
                .await?;
            scalar_index.remap(row_id_map, &new_store).await?;
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
    indexed_columns: HashMap<String, (DataType, Box<dyn ScalarQueryParser>)>,
}

impl IndexInformationProvider for ScalarIndexInfo {
    fn get_index(&self, col: &str) -> Option<(&DataType, &dyn ScalarQueryParser)> {
        self.indexed_columns
            .get(col)
            .map(|(ty, parser)| (ty, parser.as_ref()))
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
        match (index_type, params.index_name()) {
            (
                IndexType::Bitmap | IndexType::BTree | IndexType::Inverted | IndexType::LabelList,
                LANCE_SCALAR_INDEX,
            ) => {
                let params = ScalarIndexParams::new(index_type.try_into()?);
                build_scalar_index(self, column, &index_id.to_string(), &params).await?;
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
                build_scalar_index(self, column, &index_id.to_string(), params).await?;
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

                build_vector_index(self, column, &index_name, &index_id.to_string(), vec_params)
                    .await?;
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
                    // this should never happen beause we control the registration
                    // if this fails, the registration logic has a bug
                    .ok_or(Error::Internal {
                        message: "unable to cast index extension to vector".to_string(),
                        location: location!(),
                    })?;

                ext.create_index(self, column, &index_id.to_string(), params)
                    .await?;
            }
            (index_type, index_name) => {
                return Err(Error::Index {
                    message: format!(
                        "Index type {index_type} with name {index_name} is not supported"
                    ),
                    location: location!(),
                });
            }
        }

        let new_idx = IndexMetadata {
            uuid: index_id,
            name: index_name,
            fields: vec![field.id],
            dataset_version: self.manifest.version,
            fragment_bitmap: Some(self.get_fragments().iter().map(|f| f.id() as u32).collect()),
        };
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            None,
        );

        let new_manifest = commit_transaction(
            self,
            self.object_store(),
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(new_manifest);

        Ok(())
    }

    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let dataset_dir = self.base.to_string();
        if let Some(indices) = self
            .session
            .index_cache
            .get_metadata(&dataset_dir, self.version().version)
        {
            return Ok(indices);
        }

        let manifest_file = self.manifest_file(self.version().version).await?;
        let loaded_indices: Arc<Vec<IndexMetadata>> =
            read_manifest_indexes(&self.object_store, &manifest_file, &self.manifest)
                .await?
                .into();

        self.session.index_cache.insert_metadata(
            &dataset_dir,
            self.version().version,
            loaded_indices.clone(),
        );
        Ok(loaded_indices)
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

        let new_idx = IndexMetadata {
            uuid: index_id,
            name: index_name.to_string(),
            fields: vec![field.id],
            dataset_version: self.manifest.version,
            fragment_bitmap: Some(self.get_fragments().iter().map(|f| f.id() as u32).collect()),
        };

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            None,
        );

        let new_manifest = commit_transaction(
            self,
            self.object_store(),
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(new_manifest);

        Ok(())
    }

    async fn load_scalar_index_for_column(&self, col: &str) -> Result<Option<IndexMetadata>> {
        Ok(self
            .load_indices()
            .await?
            .iter()
            .filter(|idx| idx.fields.len() == 1)
            .find(|idx| {
                let field = self.schema().field_by_id(idx.fields[0]);
                if let Some(field) = field {
                    field.name == col
                } else {
                    false
                }
            })
            .cloned())
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
                    .map_or(true, |names| names.contains(&idx.name))
            })
            .map(|idx| (idx.name.clone(), idx))
            .into_group_map();

        let mut new_indices = vec![];
        let mut removed_indices = vec![];
        for deltas in name_to_indices.values() {
            let Some((new_id, removed, mut new_frag_ids)) =
                merge_indices(dataset.clone(), deltas.as_slice(), options).await?
            else {
                continue;
            };
            for removed_idx in removed.iter() {
                new_frag_ids |= removed_idx.fragment_bitmap.as_ref().unwrap();
            }

            let last_idx = deltas.last().expect("Delte indices should not be empty");
            let new_idx = IndexMetadata {
                uuid: new_id,
                name: last_idx.name.clone(), // Keep the same name
                fields: last_idx.fields.clone(),
                dataset_version: self.manifest.version,
                fragment_bitmap: Some(new_frag_ids),
            };
            removed_indices.extend(removed.iter().map(|&idx| idx.clone()));
            if deltas.len() > removed.len() {
                new_indices.extend(
                    deltas[0..(deltas.len() - removed.len())]
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

        let new_manifest = commit_transaction(
            self,
            self.object_store(),
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(new_manifest);
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
            .then(|m| async move { self.open_generic_index(column, &m.uuid.to_string()).await })
            .try_collect::<Vec<_>>()
            .await?;

        // Stastistics for each delta index.
        let indices_stats = indices
            .iter()
            .map(|idx| idx.statistics())
            .collect::<Result<Vec<_>>>()?;

        let index_type = indices[0].index_type().to_string();

        let indexed_fragments_per_delta = self.indexed_fragments(index_name).await?;
        let num_indexed_rows_per_delta = self.indexed_fragments(index_name).await?
        .iter()
        .map(|frags| {
            frags.iter().map(|f| f.num_rows().expect("Fragment should have row counts, please upgrade lance and trigger a single right to fix this")).sum::<usize>()
        })
        .collect::<Vec<_>>();

        let num_indexed_fragments = indexed_fragments_per_delta
            .clone()
            .into_iter()
            .flatten()
            .map(|f| f.id)
            .collect::<HashSet<_>>()
            .len();

        let num_unindexed_fragments = self.fragments().len() - num_indexed_fragments;
        let num_indexed_rows = num_indexed_rows_per_delta.iter().last().unwrap();
        let num_unindexed_rows = self.count_rows(None).await? - num_indexed_rows;

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
        });

        serde_json::to_string(&stats).map_err(|e| Error::Index {
            message: format!("Failed to serialize index statistics: {}", e),
            location: location!(),
        })
    }
}

/// A trait for internal dataset utilities
///
/// Internal use only. No API stability guarantees.
#[async_trait]
pub trait DatasetIndexInternalExt: DatasetIndexExt {
    /// Opens an index (scalar or vector) as a generic index
    async fn open_generic_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn Index>>;
    /// Opens the requested scalar index
    async fn open_scalar_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn ScalarIndex>>;
    /// Opens the requested vector index
    async fn open_vector_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn VectorIndex>>;
    /// Loads information about all the available scalar indices on the dataset
    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo>;

    /// Return the fragments that are not covered by any of the deltas of the index.
    async fn unindexed_fragments(&self, idx_name: &str) -> Result<Vec<Fragment>>;

    /// Return the fragments that are covered by each of the deltas of the index.
    async fn indexed_fragments(&self, idx_name: &str) -> Result<Vec<Vec<Fragment>>>;
}

#[async_trait]
impl DatasetIndexInternalExt for Dataset {
    async fn open_generic_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn Index>> {
        // Checking for cache existence is cheap so we just check both scalar and vector caches
        if let Some(index) = self.session.index_cache.get_scalar(uuid) {
            return Ok(index.as_index());
        }
        if let Some(index) = self.session.index_cache.get_vector(uuid) {
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
            let index = self.open_vector_index(column, uuid).await?;
            Ok(index.as_index())
        } else {
            let index = self.open_scalar_index(column, uuid).await?;
            Ok(index.as_index())
        }
    }

    async fn open_scalar_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn ScalarIndex>> {
        if let Some(index) = self.session.index_cache.get_scalar(uuid) {
            return Ok(index);
        }

        let index = crate::index::scalar::open_scalar_index(self, column, uuid).await?;
        self.session.index_cache.insert_scalar(uuid, index.clone());
        Ok(index)
    }

    async fn open_vector_index(&self, column: &str, uuid: &str) -> Result<Arc<dyn VectorIndex>> {
        if let Some(index) = self.session.index_cache.get_vector(uuid) {
            log::debug!("Found vector index in cache uuid: {}", uuid);
            return Ok(index);
        }

        let index_dir = self.indices_dir().child(uuid);
        let index_file = index_dir.child(INDEX_FILE_NAME);
        let reader: Arc<dyn Reader> = self.object_store.open(&index_file).await?.into();

        let tailing_bytes = read_last_block(reader.as_ref()).await?;
        let (major_version, minor_version) = read_version(&tailing_bytes)?;

        // the index file is in lance format since version (0,2)
        // TODO: we need to change the legacy IVF_PQ to be in lance format
        let index = match (major_version, minor_version) {
            (0, 1) | (0, 0) => {
                let proto = open_index_proto(reader.as_ref()).await?;
                match &proto.implementation {
                    Some(Implementation::VectorIndex(vector_index)) => {
                        let dataset = Arc::new(self.clone());
                        crate::index::vector::open_vector_index(
                            dataset,
                            column,
                            uuid,
                            vector_index,
                            reader,
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
                let reader = FileReader::try_new_self_described_from_reader(
                    reader.clone(),
                    Some(&self.session.file_metadata_cache),
                )
                .await?;
                crate::index::vector::open_vector_index_v2(
                    Arc::new(self.clone()),
                    column,
                    uuid,
                    reader,
                )
                .await
            }

            (0, 3) => {
                let scheduler = ScanScheduler::new(
                    self.object_store.clone(),
                    SchedulerConfig::max_bandwidth(&self.object_store),
                );
                let file = scheduler.open_file(&index_file).await?;
                let reader = v2::reader::FileReader::try_open(
                    file,
                    None,
                    Default::default(),
                    &self.session.file_metadata_cache,
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

                let value_type = if let DataType::FixedSizeList(df, _) = field.data_type() {
                    Result::Ok(df.data_type().to_owned())
                } else {
                    return Err(Error::Index {
                        message: format!("Column {} is not a vector column", column),
                        location: location!(),
                    });
                }?;
                match index_metadata.index_type.as_str() {
                    "IVF_FLAT" => match value_type {
                        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                            let ivf = IVFIndex::<FlatIndex, FlatQuantizer>::try_new(
                                self.object_store.clone(),
                                self.indices_dir(),
                                uuid.to_owned(),
                                Arc::downgrade(&self.session),
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
        self.session.index_cache.insert_vector(uuid, index.clone());
        Ok(index)
    }

    async fn scalar_index_info(&self) -> Result<ScalarIndexInfo> {
        let indices = self.load_indices().await?;
        let schema = self.schema();
        let mut indexed_fields = Vec::new();
        for index in indices.iter().filter(|idx| {
            let idx_schema = schema.project_by_ids(idx.fields.as_slice());
            let is_vector_index = idx_schema
                .fields
                .iter()
                .any(|f| matches!(f.data_type(), DataType::FixedSizeList(_, _)));
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
                DataType::List(_) => {
                    Box::<LabelListQueryParser>::default() as Box<dyn ScalarQueryParser>
                }
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let uuid = index.uuid.to_string();
                    let index_type = detect_scalar_index_type(self, &field.name, &uuid).await?;
                    // Inverted index can't be used for filtering
                    if matches!(index_type, ScalarIndexType::Inverted) {
                        continue;
                    }
                    Box::<SargableQueryParser>::default() as Box<dyn ScalarQueryParser>
                }
                _ => Box::<SargableQueryParser>::default() as Box<dyn ScalarQueryParser>,
            };

            indexed_fields.push((field.name.clone(), (field.data_type(), query_parser)));
        }
        let index_info_map = HashMap::from_iter(indexed_fields);
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
        let mut res = vec![];
        for idx in indices.iter() {
            let mut total_fragment_bitmap = RoaringBitmap::new();
            total_fragment_bitmap |= idx.fragment_bitmap.as_ref().ok_or(Error::Index {
                message: "Please upgrade lance to 0.8+ to use this function".to_string(),
                location: location!(),
            })?;

            res.push(
                self.fragments()
                    .iter()
                    .filter(|f| total_fragment_bitmap.contains(f.id as u32))
                    .cloned()
                    .collect(),
            );
        }

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::builder::DatasetBuilder;

    use super::*;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{Field, Schema};
    use lance_arrow::*;
    use lance_index::vector::{
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, sq::builder::SQBuildParams,
    };
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_testing::datagen::generate_random_array;
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

    #[tokio::test]
    async fn test_count_index_rows() {
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
        let test_dir = tempdir().unwrap();
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

        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
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
                num_indices_to_merge: 0,   // Just create index for delta
                index_names: Some(vec![]), // Optimize nothing
            })
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        // optimize the other index
        dataset
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 0, // Just create index for delta
                index_names: Some(vec!["other_vec_idx".to_string()]),
            })
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_unindexed_rows"], 512);
        assert_eq!(stats["num_indexed_rows"], 512);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);
        assert_eq!(stats["num_indices"], 1);

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("other_vec_idx").await.unwrap())
                .unwrap();
        assert_eq!(stats["num_unindexed_rows"], 0);
        assert_eq!(stats["num_indexed_rows"], 1024);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        assert_eq!(stats["num_indices"], 2);

        dataset
            .optimize_indices(&OptimizeOptions {
                num_indices_to_merge: 0, // Just create index for delta
                ..Default::default()
            })
            .await
            .unwrap();
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

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
            .open_generic_index("tag", &indices[0].uuid.to_string())
            .await
            .unwrap();
        assert_eq!(index.index_type(), IndexType::Bitmap);
    }
}
