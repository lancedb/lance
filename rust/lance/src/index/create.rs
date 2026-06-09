// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    Error, Result,
    dataset::{
        Dataset,
        transaction::{Operation, TransactionBuilder},
    },
    index::{
        DatasetIndexExt, DatasetIndexInternalExt,
        scalar::{build_bitmap_index_segment, build_scalar_index},
        vector::{
            LANCE_VECTOR_INDEX, VectorIndexParams, build_distributed_vector_index,
            build_empty_vector_index, build_vector_index,
        },
        vector_index_details, vector_index_details_default,
    },
};
use futures::future::BoxFuture;
use lance_core::datatypes::format_field_path;
use lance_index::progress::{IndexBuildProgress, NoopIndexBuildProgress};
use lance_index::{IndexParams, IndexType, scalar::CreatedIndex};
use lance_index::{
    metrics::NoOpMetricsCollector,
    scalar::{LANCE_SCALAR_INDEX, ScalarIndexParams, inverted::tokenizer::InvertedIndexParams},
};
use lance_table::format::{IndexMetadata, list_index_files_with_sizes};
use std::{collections::HashMap, future::IntoFuture, sync::Arc};
use tracing::instrument;
use uuid::Uuid;

use arrow_array::RecordBatchReader;
/// Generate default index name from field path.
///
/// Joins field names with `.` to create the base index name.
/// For example: `["meta-data", "user-id"]` -> `"meta-data.user-id"`
fn default_index_name(fields: &[&str]) -> String {
    if fields.iter().any(|f| f.contains('.')) {
        format_field_path(fields)
    } else {
        fields.join(".")
    }
}

pub struct CreateIndexBuilder<'a> {
    dataset: &'a mut Dataset,
    columns: Vec<String>,
    index_type: IndexType,
    params: &'a dyn IndexParams,
    name: Option<String>,
    replace: bool,
    train: bool,
    fragments: Option<Vec<u32>>,
    index_uuid: Option<Uuid>,
    preprocessed_data: Option<Box<dyn RecordBatchReader + Send + 'static>>,
    progress: Arc<dyn IndexBuildProgress>,
    /// Transaction properties to store with this commit.
    transaction_properties: Option<Arc<HashMap<String, String>>>,
}

impl<'a> CreateIndexBuilder<'a> {
    pub fn new(
        dataset: &'a mut Dataset,
        columns: &[&str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> Self {
        Self {
            dataset,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            index_type,
            params,
            name: None,
            replace: false,
            train: true,
            fragments: None,
            index_uuid: None,
            preprocessed_data: None,
            progress: Arc::new(NoopIndexBuildProgress),
            transaction_properties: None,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }

    pub fn train(mut self, train: bool) -> Self {
        self.train = train;
        self
    }

    pub fn fragments(mut self, fragment_ids: Vec<u32>) -> Self {
        self.fragments = Some(fragment_ids);
        self
    }

    pub fn index_uuid(mut self, uuid: Uuid) -> Self {
        self.index_uuid = Some(uuid);
        self
    }

    pub fn preprocessed_data(
        mut self,
        stream: Box<dyn RecordBatchReader + Send + 'static>,
    ) -> Self {
        self.preprocessed_data = Some(stream);
        self
    }

    pub fn progress(mut self, p: Arc<dyn IndexBuildProgress>) -> Self {
        self.progress = p;
        self
    }

    /// Set transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed index jobs).
    pub fn transaction_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.transaction_properties = Some(Arc::new(properties));
        self
    }

    #[instrument(skip_all)]
    pub async fn execute_uncommitted(&mut self) -> Result<IndexMetadata> {
        if self.columns.len() != 1 {
            return Err(Error::index(
                "Only support building index on 1 column at the moment".to_string(),
            ));
        }
        let column_input = &self.columns[0];
        // Use case-insensitive lookup for both simple and nested paths.
        // resolve_case_insensitive tries exact match first, then falls back to case-insensitive.
        let Some(field_path) = self.dataset.schema().resolve_case_insensitive(column_input) else {
            return Err(Error::index(format!(
                "CreateIndex: column '{column_input}' does not exist"
            )));
        };
        let field = *field_path.last().unwrap();
        // Reconstruct the column path with correct case from schema
        // Use quoted format for SQL parsing (special chars are quoted)
        let names: Vec<&str> = field_path.iter().map(|f| f.name.as_str()).collect();
        let quoted_column: String = format_field_path(&names);
        let column = quoted_column.as_str();

        // If train is true but dataset is empty, automatically set train to false
        let train = if self.train {
            self.dataset.count_rows(None).await? > 0
        } else {
            false
        };

        // Load indices from the disk.
        let indices = self.dataset.load_indices().await?;
        let fri = self
            .dataset
            .open_frag_reuse_index(&NoOpMetricsCollector)
            .await?;
        let index_name = if let Some(name) = self.name.take() {
            name
        } else {
            // Generate default name with collision handling
            let column_path = default_index_name(&names);
            let base_name = format!("{column_path}_idx");
            let mut candidate = base_name.clone();
            let mut counter = 2; // Start with no suffix, then use _2, _3, ...
            // Find unique name by appending numeric suffix if needed
            while indices
                .iter()
                .any(|idx| idx.name == candidate && idx.fields != [field.id])
            {
                candidate = format!("{base_name}_{counter}");
                counter += 1;
            }
            candidate
        };
        let existing_named_indices = indices
            .iter()
            .filter(|idx| idx.name == index_name)
            .collect::<Vec<_>>();
        if existing_named_indices
            .iter()
            .any(|idx| idx.fields != [field.id])
        {
            return Err(Error::index(format!(
                "Index name '{index_name}' already exists with different fields, \
                please specify a different name"
            )));
        }
        if !existing_named_indices.is_empty() && !self.replace {
            return Err(Error::index(format!(
                "Index name '{index_name}' already exists, \
                please specify a different name or use replace=True"
            )));
        }
        ensure_index_uuid_allowed(
            self.index_type,
            self.params,
            self.fragments.as_ref(),
            self.index_uuid.as_ref(),
        )?;

        let index_id = self.index_uuid.unwrap_or_else(Uuid::new_v4);
        let mut output_index_uuid = index_id;
        let created_index = match (self.index_type, self.params.index_name()) {
            (
                IndexType::Bitmap
                | IndexType::BTree
                | IndexType::Inverted
                | IndexType::NGram
                | IndexType::FMIndex
                | IndexType::ZoneMap
                | IndexType::BloomFilter
                | IndexType::LabelList
                | IndexType::RTree,
                LANCE_SCALAR_INDEX,
            ) => {
                assert!(
                    self.preprocessed_data.is_none() || self.index_type.eq(&IndexType::BTree),
                    "Preprocessed data stream can only be provided for B-Tree index type at the moment."
                );
                let base_params = ScalarIndexParams::for_builtin(self.index_type.try_into()?);

                // If custom params were provided, extract the params JSON and apply it
                let params = if let Some(provided_params) =
                    self.params.as_any().downcast_ref::<ScalarIndexParams>()
                {
                    if let Some(params_json) = &provided_params.params {
                        // Parse and apply the custom parameters
                        if let Ok(json_value) =
                            serde_json::from_str::<serde_json::Value>(params_json)
                        {
                            base_params.with_params(&json_value)
                        } else {
                            base_params
                        }
                    } else {
                        base_params
                    }
                } else {
                    base_params
                };

                let preprocesssed_data = self
                    .preprocessed_data
                    .take()
                    .map(|reader| lance_datafusion::utils::reader_to_stream(Box::new(reader)));
                if self.index_type == IndexType::Bitmap && self.fragments.is_some() {
                    if !train {
                        return Err(Error::invalid_input(
                            "canonical bitmap segment build requires train=true".to_string(),
                        ));
                    }
                    if preprocesssed_data.is_some() {
                        return Err(Error::invalid_input(
                            "canonical bitmap segment build does not accept preprocessed data"
                                .to_string(),
                        ));
                    }
                    let fragments = self.fragments.clone().ok_or_else(|| {
                        Error::invalid_input(
                            "canonical bitmap segment build requires fragment ids".to_string(),
                        )
                    })?;
                    build_bitmap_index_segment(
                        self.dataset,
                        column,
                        index_id,
                        fragments,
                        self.progress.clone(),
                    )
                    .await?
                } else {
                    build_scalar_index(
                        self.dataset,
                        column,
                        index_id,
                        &params,
                        train,
                        self.fragments.clone(),
                        preprocesssed_data,
                        self.progress.clone(),
                    )
                    .await?
                }
            }
            (IndexType::Scalar, LANCE_SCALAR_INDEX) => {
                // Guess the index type
                let params = self
                    .params
                    .as_any()
                    .downcast_ref::<ScalarIndexParams>()
                    .ok_or_else(|| {
                        Error::index("Scalar index type must take a ScalarIndexParams".to_string())
                    })?;
                build_scalar_index(
                    self.dataset,
                    column,
                    index_id,
                    params,
                    train,
                    self.fragments.clone(),
                    None,
                    self.progress.clone(),
                )
                .await?
            }
            (IndexType::Inverted, _) => {
                // Inverted index params.
                let inverted_params = self
                    .params
                    .as_any()
                    .downcast_ref::<InvertedIndexParams>()
                    .ok_or_else(|| {
                        Error::index(
                            "Inverted index type must take a InvertedIndexParams".to_string(),
                        )
                    })?;

                let params = ScalarIndexParams::new("inverted".to_string())
                    .with_params(&inverted_params.to_training_json()?);
                build_scalar_index(
                    self.dataset,
                    column,
                    index_id,
                    &params,
                    train,
                    self.fragments.clone(),
                    None,
                    self.progress.clone(),
                )
                .await?
            }
            (
                IndexType::Vector
                | IndexType::IvfPq
                | IndexType::IvfSq
                | IndexType::IvfFlat
                | IndexType::IvfRq
                | IndexType::IvfHnswFlat
                | IndexType::IvfHnswPq
                | IndexType::IvfHnswSq,
                LANCE_VECTOR_INDEX,
            ) => {
                // Vector index params.
                let vec_params = self
                    .params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| {
                        Error::index("Vector index type must take a VectorIndexParams".to_string())
                    })?;
                let index_version = vec_params.index_type().version() as u32;

                let files = if train {
                    // Check if this is distributed indexing (fragment-level)
                    if let Some(fragments) = &self.fragments {
                        // For distributed indexing, build only on specified fragments
                        // This creates temporary index metadata without committing
                        let (segment_uuid, files) = Box::pin(build_distributed_vector_index(
                            self.dataset,
                            column,
                            &index_name,
                            index_id,
                            vec_params,
                            fri,
                            fragments,
                            self.progress.clone(),
                        ))
                        .await?;
                        output_index_uuid = segment_uuid;
                        files
                    } else {
                        // Standard full dataset indexing
                        Box::pin(build_vector_index(
                            self.dataset,
                            column,
                            &index_name,
                            index_id,
                            vec_params,
                            fri,
                            self.progress.clone(),
                        ))
                        .await?
                    }
                } else {
                    // Create empty vector index
                    build_empty_vector_index(
                        self.dataset,
                        column,
                        &index_name,
                        index_id,
                        vec_params,
                    )
                    .await?
                };
                CreatedIndex {
                    index_details: vector_index_details(vec_params),
                    index_version,
                    files,
                }
            }
            // Can't use if let Some(...) here because it's not stable yet.
            // TODO: fix after https://github.com/rust-lang/rust/issues/51114
            (IndexType::Vector, name)
                if self
                    .dataset
                    .session
                    .index_extensions
                    .contains_key(&(IndexType::Vector, name.to_string())) =>
            {
                let ext = self
                    .dataset
                    .session
                    .index_extensions
                    .get(&(IndexType::Vector, name.to_string()))
                    .expect("already checked")
                    .clone()
                    .to_vector()
                    // this should never happen because we control the registration
                    // if this fails, the registration logic has a bug
                    .ok_or(Error::internal(
                        "unable to cast index extension to vector".to_string(),
                    ))?;

                if train {
                    ext.create_index(self.dataset, column, &index_id, self.params)
                        .await?;
                } else {
                    todo!("create empty vector index when train=false");
                }
                // Capture file sizes after vector index creation
                let index_dir = self.dataset.indices_dir().join(index_id.to_string());
                let files =
                    list_index_files_with_sizes(&self.dataset.object_store, &index_dir).await?;
                CreatedIndex {
                    index_details: vector_index_details_default(),
                    index_version: self.index_type.version() as u32,
                    files,
                }
            }
            (IndexType::FragmentReuse, _) => {
                return Err(Error::index(
                    "Fragment reuse index can only be created through compaction".to_string(),
                ));
            }
            (index_type, index_name) => {
                return Err(Error::index(format!(
                    "Index type {index_type} with name {index_name} is not supported"
                )));
            }
        };

        Ok(IndexMetadata {
            uuid: output_index_uuid,
            name: index_name,
            fields: vec![field.id],
            dataset_version: self.dataset.manifest.version,
            fragment_bitmap: if train {
                match &self.fragments {
                    Some(fragment_ids) => Some(fragment_ids.iter().collect()),
                    None => Some(self.dataset.fragment_bitmap.as_ref().clone()),
                }
            } else {
                // Empty bitmap for untrained indices
                Some(roaring::RoaringBitmap::new())
            },
            index_details: Some(Arc::new(created_index.index_details)),
            index_version: created_index.index_version as i32,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(created_index.files),
        })
    }

    #[instrument(skip_all)]
    async fn execute(mut self) -> Result<IndexMetadata> {
        let new_idx = self.execute_uncommitted().await?;
        let index_uuid = new_idx.uuid;
        let removed_indices = if self.replace {
            self.dataset
                .load_indices()
                .await?
                .iter()
                .filter(|idx| idx.name == new_idx.name)
                .cloned()
                .collect()
        } else {
            vec![]
        };
        let transaction = if uses_segment_commit_path(self.index_type, self.params) {
            let dataset_version = new_idx.dataset_version;
            TransactionBuilder::new(
                dataset_version,
                Operation::CreateIndex {
                    new_indices: vec![new_idx],
                    removed_indices,
                },
            )
            .transaction_properties(self.transaction_properties.clone())
            .build()
        } else {
            TransactionBuilder::new(
                new_idx.dataset_version,
                Operation::CreateIndex {
                    new_indices: vec![new_idx],
                    removed_indices,
                },
            )
            .transaction_properties(self.transaction_properties.clone())
            .build()
        };

        self.dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        // Fetch the committed index metadata from the dataset.
        // This ensures we return the version that may have been modified by the commit.
        let indices = self.dataset.load_indices().await?;
        indices
            .iter()
            .find(|idx| idx.uuid == index_uuid)
            .cloned()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Index with UUID {} not found after commit",
                    index_uuid
                ))
            })
    }
}

fn is_btree_scalar_params(params: &dyn IndexParams) -> bool {
    params
        .as_any()
        .downcast_ref::<ScalarIndexParams>()
        .is_some_and(|p| p.index_type.eq_ignore_ascii_case("btree"))
}

/// Validate that a user-supplied `index_uuid` is permitted for this build.
fn ensure_index_uuid_allowed(
    index_type: IndexType,
    params: &dyn IndexParams,
    fragments: Option<&Vec<u32>>,
    index_uuid: Option<&Uuid>,
) -> Result<()> {
    let is_btree = index_type == IndexType::BTree
        || params
            .as_any()
            .downcast_ref::<ScalarIndexParams>()
            .map(|params| params.index_type.eq_ignore_ascii_case("btree"))
            .unwrap_or(false);

    if index_uuid.is_some() && fragments.is_some_and(|fragments| !fragments.is_empty()) && is_btree
    {
        return Err(Error::invalid_input(
            "index_uuid is no longer accepted for BTree distributed index builds; segment UUIDs \
             are generated by Lance and returned in the index metadata."
                .to_string(),
        ));
    }

    Ok(())
}

fn uses_segment_commit_path(index_type: IndexType, params: &dyn IndexParams) -> bool {
    let params_family = params.index_name();

    if params_family == LANCE_VECTOR_INDEX
        && matches!(
            index_type,
            IndexType::Vector
                | IndexType::IvfPq
                | IndexType::IvfSq
                | IndexType::IvfFlat
                | IndexType::IvfRq
                | IndexType::IvfHnswFlat
                | IndexType::IvfHnswPq
                | IndexType::IvfHnswSq
        )
        && params.as_any().is::<VectorIndexParams>()
    {
        return true;
    }

    if params_family == LANCE_SCALAR_INDEX {
        match index_type {
            IndexType::BTree => return true,
            IndexType::Scalar if is_btree_scalar_params(params) => return true,
            _ => {}
        }
    }

    false
}

impl<'a> IntoFuture for CreateIndexBuilder<'a> {
    type Output = Result<IndexMetadata>;
    type IntoFuture = BoxFuture<'a, Result<IndexMetadata>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::{DatasetIndexExt, IndexSegment};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::datatypes::{Float32Type, Int32Type};
    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatchIterator};
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use datafusion::common::ScalarValue;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::{address::RowAddress, tempfile::TempStrDir};
    use lance_datagen::{self, gen_batch};
    use lance_index::optimize::OptimizeOptions;
    use lance_index::progress::IndexBuildProgress;
    use lance_index::scalar::{
        FullTextSearchQuery, SargableQuery, SearchResult, inverted::tokenizer::InvertedIndexParams,
    };
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
    use lance_linalg::distance::{DistanceType, MetricType};
    use std::{collections::BTreeSet, ops::Bound, sync::Arc};
    use uuid::Uuid;

    lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

    #[test]
    fn test_inverted_training_params_include_build_only_fields() {
        let params = InvertedIndexParams::default()
            .memory_limit_mb(4096)
            .num_workers(7);
        let scalar_params = ScalarIndexParams::new("inverted".to_string())
            .with_params(&params.to_training_json().unwrap());
        let json: serde_json::Value =
            serde_json::from_str(scalar_params.params.as_ref().unwrap()).unwrap();
        assert_eq!(
            json.get("memory_limit"),
            Some(&serde_json::Value::from(4096))
        );
        assert_eq!(json.get("num_workers"), Some(&serde_json::Value::from(7)));
    }

    #[test]
    fn test_default_index_name() {
        // Single field - preserved as-is
        assert_eq!(default_index_name(&["user-id"]), "user-id");
        assert_eq!(default_index_name(&["user:id"]), "user:id");
        assert_eq!(default_index_name(&["userId"]), "userId");

        // Nested paths - joined with dot
        assert_eq!(
            default_index_name(&["meta-data", "user-id"]),
            "meta-data.user-id"
        );
        assert_eq!(
            default_index_name(&["MetaData", "userId"]),
            "MetaData.userId"
        );

        // Path with dots in field names - escape
        assert_eq!(
            default_index_name(&["meta.data", "user.id"]),
            "`meta.data`.`user.id`"
        );

        // Empty input
        assert_eq!(default_index_name(&[]), "");
    }

    #[tokio::test]
    async fn test_default_index_name_with_special_chars() {
        // Verify default index names preserve special characters in column names.
        let mut dataset = gen_batch()
            .col("user-id", lance_datagen::array::step::<Int32Type>())
            .col("user:id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);

        // Create index on column with hyphen
        let idx1 = CreateIndexBuilder::new(&mut dataset, &["user-id"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx1.name, "user-id_idx");

        // Create index on column with colon
        let idx2 = CreateIndexBuilder::new(&mut dataset, &["user:id"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx2.name, "user:id_idx");

        // Verify both indices exist
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
    }

    #[tokio::test]
    async fn test_index_name_collision_with_explicit_name() {
        // Test collision handling when explicit name conflicts with default name.
        let mut dataset = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .col("b", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);

        // (a) Explicit name on first index, default on second that would collide
        // Create index on "a" with explicit name "b_idx"
        let idx1 = CreateIndexBuilder::new(&mut dataset, &["a"], IndexType::BTree, &params)
            .name("b_idx".to_string())
            .execute()
            .await
            .unwrap();
        assert_eq!(idx1.name, "b_idx");

        // Create index on "b" with default name - would be "b_idx" but that's taken
        // so it should get "b_idx_2"
        let idx2 = CreateIndexBuilder::new(&mut dataset, &["b"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx2.name, "b_idx_2");

        // Verify both indices exist
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
    }

    #[tokio::test]
    async fn test_index_name_collision_explicit_errors() {
        // Test that explicit name collision with existing index errors.
        let mut dataset = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .col("b", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);

        // (b) Default name on first, explicit same name on second should error
        // Create index on "a" with default name "a_idx"
        let idx1 = CreateIndexBuilder::new(&mut dataset, &["a"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx1.name, "a_idx");

        // Try to create index on "b" with explicit name "a_idx" - should error
        let result = CreateIndexBuilder::new(&mut dataset, &["b"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn test_concurrent_create_index_same_name_returns_retryable_conflict() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());
        let reader = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(100),
                lance_datagen::BatchCount::from(1),
            );
        let dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let read_version = dataset.manifest.version;
        let mut reader1 = dataset.checkout_version(read_version).await.unwrap();
        let mut reader2 = dataset.checkout_version(read_version).await.unwrap();

        let first = CreateIndexBuilder::new(&mut reader1, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await;
        assert!(
            first.is_ok(),
            "first create_index should succeed: {first:?}"
        );

        let second = CreateIndexBuilder::new(&mut reader2, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await;
        assert!(
            matches!(second, Err(Error::RetryableCommitConflict { .. })),
            "second concurrent create_index should be retryable, got {second:?}"
        );

        let latest_indices = reader1.load_indices_by_name("a_idx").await.unwrap();
        assert_eq!(latest_indices.len(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_replace_index_same_name_returns_retryable_conflict() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());
        let reader = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(100),
                lance_datagen::BatchCount::from(1),
            );
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let original = CreateIndexBuilder::new(&mut dataset, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await
            .unwrap();

        let read_version = dataset.manifest.version;
        let mut reader1 = dataset.checkout_version(read_version).await.unwrap();
        let mut reader2 = dataset.checkout_version(read_version).await.unwrap();

        let replacement = CreateIndexBuilder::new(&mut reader1, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .replace(true)
            .execute()
            .await
            .unwrap();
        assert_ne!(replacement.uuid, original.uuid);

        let second = CreateIndexBuilder::new(&mut reader2, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .replace(true)
            .execute()
            .await;
        assert!(
            matches!(second, Err(Error::RetryableCommitConflict { .. })),
            "second concurrent replace should be retryable, got {second:?}"
        );

        let latest_indices = reader1.load_indices_by_name("a_idx").await.unwrap();
        assert_eq!(latest_indices.len(), 1);
        assert_eq!(latest_indices[0].uuid, replacement.uuid);
        assert_ne!(latest_indices[0].uuid, original.uuid);
    }

    // Helper function to create test data with text field suitable for inverted index
    fn create_text_batch(start: i32, end: i32) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Utf8, false),
        ]));
        let texts = (start..end)
            .map(|i| match i % 3 {
                0 => format!("document {} with some text content", i),
                1 => format!("another document {} containing different words", i),
                _ => format!("text sample {} for testing inverted index", i),
            })
            .collect::<Vec<_>>();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(start..end)),
                Arc::new(StringArray::from_iter_values(texts)),
            ],
        )
        .unwrap()
    }

    async fn prepare_vector_ivf(dataset: &Dataset, vector_column: &str) -> IvfBuildParams {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<Float32Type>();

        let kmeans = train_kmeans::<Float32Type>(
            values,
            KMeansParams::new(None, 10, 1, DistanceType::L2),
            dim,
            4,
            3,
        )
        .unwrap();
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                kmeans.centroids.as_primitive::<Float32Type>().clone(),
                dim as i32,
            )
            .unwrap(),
        );
        IvfBuildParams::try_with_centroids(4, centroids).unwrap()
    }

    #[tokio::test]
    async fn test_execute_uncommitted() {
        // Test the complete workflow that covers the user's specified code pattern:
        // 1. Create dataset with multiple fragments
        // 2. Get fragment IDs from dataset using dataset.get_fragments()
        // 3. Create CreateIndexBuilder with fragments() method
        // 4. Call execute_uncommitted() to get IndexMetadata
        // 5. Verify IndexMetadata contains correct fragment_bitmap

        // Create temporary directory for dataset
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Create test data with multiple fragments
        let batch1 = create_text_batch(0, 10);
        let batch2 = create_text_batch(10, 20);
        let batch3 = create_text_batch(20, 30);

        let write_params = WriteParams {
            max_rows_per_file: 10, // Force multiple fragments
            max_rows_per_group: 5,
            ..Default::default()
        };

        // Write dataset with multiple batches to create multiple fragments
        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params))
            .await
            .unwrap();

        let params = InvertedIndexParams::default();

        // Get fragment IDs from the dataset
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        assert!(
            fragment_ids.len() >= 2,
            "Should have multiple fragments for testing"
        );

        // Test fragments() method with specific fragment IDs and ensure duplicate/out-of-order fragments are handled properly
        let selected_fragments = vec![
            fragment_ids[1],
            fragment_ids[0],
            fragment_ids[1],
            fragment_ids[2],
        ];
        let selected_fragments_expected = vec![fragment_ids[0], fragment_ids[1], fragment_ids[2]];

        let mut builder =
            CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                .name("fragment_index".to_string())
                .fragments(selected_fragments.clone());

        // Execute uncommitted to get index metadata
        let index_metadata = builder.execute_uncommitted().await.unwrap();

        // Verify the IndexMetadata contains the correct fragment_bitmap
        let fragment_bitmap = index_metadata.fragment_bitmap.unwrap();
        let indexed_fragments: Vec<u32> = fragment_bitmap.iter().collect();
        assert_eq!(
            indexed_fragments, selected_fragments_expected,
            "Index should only cover the selected fragments"
        );

        // Verify other metadata fields
        assert_eq!(index_metadata.name, "fragment_index");
        assert!(!index_metadata.uuid.is_nil());
        assert!(index_metadata.created_at.is_some());
    }

    #[tokio::test]
    async fn test_merge_index_metadata_inverted_reports_progress() {
        // This exercises the public distributed inverted-index workflow end to end:
        // 1. build one uncommitted shard per fragment with CreateIndexBuilder.progress(...)
        // 2. merge those shards with Dataset::merge_index_metadata(...)
        //
        // Expected outcomes:
        // - the build callback should surface public build stages such as load_data,
        //   tokenize_docs, copy_partitions, and write_metadata
        // - the merge callback should surface public merge stages such as
        //   read_partition_metadata, remap_partition_files, and write_merged_metadata
        // - merge stages should be reported in execution order
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batch1 = create_text_batch(0, 15);
        let batch2 = create_text_batch(15, 30);
        let batch3 = create_text_batch(30, 45);

        let write_params = WriteParams {
            max_rows_per_file: 15,
            max_rows_per_group: 5,
            ..Default::default()
        };

        // Write dataset with multiple batches to create multiple fragments
        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params))
            .await
            .unwrap();

        let params = InvertedIndexParams::default();
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        let shared_uuid = Uuid::new_v4();
        let build_progress = Arc::new(RecordingProgress::default());

        for &fragment_id in &fragment_ids {
            let mut builder =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("distributed_index".to_string())
                    .fragments(vec![fragment_id])
                    .index_uuid(shared_uuid)
                    .progress(build_progress.clone());

            let index_metadata = builder.execute_uncommitted().await.unwrap();
            assert_eq!(index_metadata.uuid, shared_uuid);
            assert_eq!(index_metadata.name, "distributed_index");

            let fragment_bitmap = index_metadata.fragment_bitmap.as_ref().unwrap();
            let indexed_fragments: Vec<u32> = fragment_bitmap.iter().collect();
            assert_eq!(indexed_fragments, vec![fragment_id]);
        }

        let merge_progress = Arc::new(RecordingProgress::default());
        dataset
            .merge_index_metadata(
                &shared_uuid,
                IndexType::Inverted,
                None,
                merge_progress.clone(),
            )
            .await
            .unwrap();

        let build_tags = build_progress
            .recorded_events()
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        assert!(
            build_tags.iter().any(|e| e == "start:load_data"),
            "expected load_data progress during public distributed build"
        );
        assert!(
            build_tags.iter().any(|e| e == "start:tokenize_docs"),
            "expected tokenize_docs progress during public distributed build"
        );
        assert!(
            build_tags.iter().any(|e| e == "start:copy_partitions"),
            "expected copy_partitions progress during public distributed build"
        );
        assert!(
            build_tags.iter().any(|e| e == "start:write_metadata"),
            "expected write_metadata progress during public distributed build"
        );

        let merge_events = merge_progress.recorded_events();
        let merge_tags = merge_events
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        let read_start = merge_tags
            .iter()
            .position(|e| e == "start:read_partition_metadata")
            .expect("missing read_partition_metadata start");
        let read_complete = merge_tags
            .iter()
            .position(|e| e == "complete:read_partition_metadata")
            .expect("missing read_partition_metadata complete");
        let remap_start = merge_tags
            .iter()
            .position(|e| e == "start:remap_partition_files")
            .expect("missing remap_partition_files start");
        let remap_complete = merge_tags
            .iter()
            .position(|e| e == "complete:remap_partition_files")
            .expect("missing remap_partition_files complete");
        let metadata_start = merge_tags
            .iter()
            .position(|e| e == "start:write_merged_metadata")
            .expect("missing write_merged_metadata start");
        let metadata_complete = merge_tags
            .iter()
            .position(|e| e == "complete:write_merged_metadata")
            .expect("missing write_merged_metadata complete");
        assert!(read_start < read_complete);
        assert!(read_complete < remap_start);
        assert!(remap_start < remap_complete);
        assert!(remap_complete < metadata_start);
        assert!(metadata_start < metadata_complete);
        assert!(
            merge_tags
                .iter()
                .any(|e| e == "progress:read_partition_metadata"),
            "expected read_partition_metadata progress during public merge"
        );
        assert!(
            merge_tags
                .iter()
                .any(|e| e == "progress:remap_partition_files"),
            "expected remap_partition_files progress during public merge"
        );
        assert!(
            merge_tags
                .iter()
                .any(|e| e == "progress:write_merged_metadata"),
            "expected write_merged_metadata progress during public merge"
        );
    }

    #[tokio::test]
    async fn test_merge_index_metadata_btree_soft_break() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());
        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(8),
                lance_datagen::BatchCount::from(1),
            );
        let dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let err = dataset
            .merge_index_metadata(
                &Uuid::new_v4(),
                IndexType::BTree,
                None,
                Arc::new(NoopIndexBuildProgress),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("no longer supports merge_index_metadata"),
            "expected BTree merge_index_metadata soft-break error, got: {err}"
        );
    }

    /// Assert a committed segment directory holds exactly one canonical BTree
    /// payload — one `page_data.lance` + one `page_lookup.lance` — and no `part_*`
    /// shard files. Locks the "every segment has exactly one lookup" invariant.
    async fn assert_canonical_btree_segment(dataset: &Dataset, uuid: &Uuid) {
        let index_dir = dataset.indices_dir().join(uuid.to_string());
        let files = list_index_files_with_sizes(&dataset.object_store, &index_dir)
            .await
            .unwrap();
        let names: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            names.iter().filter(|n| **n == "page_lookup.lance").count(),
            1,
            "segment must have exactly one canonical page_lookup.lance, got {names:?}"
        );
        assert_eq!(
            names.iter().filter(|n| **n == "page_data.lance").count(),
            1,
            "segment must have exactly one canonical page_data.lance, got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("part_")),
            "segment must have no part_* shard files, got {names:?}"
        );
    }

    #[tokio::test]
    async fn test_segmented_btree_multi_fragment_commit_and_search() {
        let test_dir = TempStrDir::default();
        let dataset = gen_batch()
            .col("value", lance_datagen::array::step::<Int32Type>())
            .into_dataset(
                test_dir.as_str(),
                FragmentCount::from(4),
                FragmentRowCount::from(16),
            )
            .await
            .unwrap();
        let mut dataset = dataset;
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 4);

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let mut segments = Vec::new();
        for fragment in &fragments {
            segments.push(
                CreateIndexBuilder::new(&mut dataset, &["value"], IndexType::BTree, &params)
                    .name("value_btree_segments".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        let segment_uuids = segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();

        dataset
            .commit_existing_index_segments("value_btree_segments", "value", segments)
            .await
            .unwrap();

        let committed = dataset
            .load_indices_by_name("value_btree_segments")
            .await
            .unwrap();
        assert_eq!(committed.len(), fragments.len());
        for segment_uuid in &segment_uuids {
            assert_canonical_btree_segment(&dataset, segment_uuid).await;
        }

        let logical = crate::index::scalar_logical::open_named_scalar_index(
            &dataset,
            "value",
            "value_btree_segments",
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();

        let point_query = SargableQuery::Equals(ScalarValue::Int32(Some(33)));
        let point_matches = match logical
            .search(&point_query, &NoOpMetricsCollector)
            .await
            .unwrap()
        {
            SearchResult::Exact(row_addrs) => row_addrs.true_rows().row_addrs().unwrap().count(),
            other => panic!("expected exact point result, got {other:?}"),
        };
        assert_eq!(point_matches, 1);

        let range_query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(14))),
            Bound::Excluded(ScalarValue::Int32(Some(35))),
        );
        let range_row_addrs = match logical
            .search(&range_query, &NoOpMetricsCollector)
            .await
            .unwrap()
        {
            SearchResult::Exact(row_addrs) => row_addrs,
            other => panic!("expected exact range result, got {other:?}"),
        };
        let searched_fragments = range_row_addrs
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(|row_addr| RowAddress::from(u64::from(row_addr)).fragment_id())
            .collect::<Vec<_>>();
        assert_eq!(searched_fragments.len(), 21);
        assert_eq!(
            searched_fragments.into_iter().collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1, 2])
        );
    }

    #[tokio::test]
    async fn test_btree_distributed_index_uuid_rejected() {
        let test_dir = TempStrDir::default();
        let dataset = gen_batch()
            .col("value", lance_datagen::array::step::<Int32Type>())
            .into_dataset(
                test_dir.as_str(),
                FragmentCount::from(2),
                FragmentRowCount::from(16),
            )
            .await
            .unwrap();
        let mut dataset = dataset;
        let fragment_id = dataset.get_fragments()[0].id() as u32;

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        for index_type in [IndexType::BTree, IndexType::Scalar] {
            let err = CreateIndexBuilder::new(&mut dataset, &["value"], index_type, &params)
                .name("value_btree_segments".to_string())
                .fragments(vec![fragment_id])
                .index_uuid(Uuid::new_v4())
                .execute_uncommitted()
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::InvalidInput { .. }),
                "expected invalid input error, got: {err}"
            );
            assert!(
                err.to_string().contains(
                    "index_uuid is no longer accepted for BTree distributed index builds"
                ),
                "unexpected error: {err}"
            );
        }
    }

    #[tokio::test]
    async fn test_range_based_btree_index_create() {
        use crate::dataset::scanner::ColumnOrdering;
        use futures::TryStreamExt;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Write the dataset with deliberately unsorted ids so the sort step is real.
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let ids: Vec<i32> = (0..256).rev().collect();
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(ids))]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        // The worker scans the dataset's `(id, _rowid)` rows and sorts them by value,
        // producing the BTree training stream `(value, _rowid)` externally — the "scan,
        // sort, then hand a pre-sorted reader to the builder" path; no `range_id`.
        let sorted_batches: Vec<RecordBatch> = {
            let mut scan = dataset.scan();
            scan.order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                "id".to_string(),
            )]))
            .unwrap();
            scan.with_row_id();
            scan.project_with_transform(&[("value", "id")]).unwrap();
            scan.try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap()
        };
        let train_schema = sorted_batches[0].schema();
        let sorted_reader =
            RecordBatchIterator::new(sorted_batches.into_iter().map(Ok), train_schema);

        // Build one self-contained segment directly from the sorted reader.
        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let segment = CreateIndexBuilder::new(&mut dataset, &["id"], IndexType::BTree, &params)
            .name("id_btree".to_string())
            .preprocessed_data(Box::new(sorted_reader))
            .execute_uncommitted()
            .await
            .unwrap();
        let segment_uuid = segment.uuid;

        // Commit the segment via the segmented-index API — no merge step.
        dataset
            .commit_existing_index_segments("id_btree", "id", vec![segment])
            .await
            .unwrap();

        let committed = dataset.load_indices_by_name("id_btree").await.unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].uuid, segment_uuid);

        // Exactly one canonical data + one lookup, no shards.
        assert_canonical_btree_segment(&dataset, &segment_uuid).await;

        // The committed index answers a range query correctly.
        let logical = crate::index::scalar_logical::open_named_scalar_index(
            &dataset,
            "id",
            "id_btree",
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(0))),
            Bound::Excluded(ScalarValue::Int32(Some(64))),
        );
        let matched = match logical.search(&query, &NoOpMetricsCollector).await.unwrap() {
            SearchResult::Exact(row_addrs) => row_addrs.true_rows().row_addrs().unwrap().count(),
            other => panic!("expected exact result, got {other:?}"),
        };
        assert_eq!(matched, 64);
    }

    #[tokio::test]
    async fn test_bitmap_execute_uncommitted_writes_canonical_segment() {
        use lance_index::scalar::bitmap::BITMAP_LOOKUP_NAME;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "category",
            DataType::Int32,
            false,
        )]));
        let batches = (0..4)
            .map(
                |fragment_id| -> std::result::Result<_, arrow_schema::ArrowError> {
                    let values = vec![fragment_id, fragment_id, fragment_id + 10, fragment_id + 10];
                    Ok(RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(values))],
                    )
                    .unwrap())
                },
            )
            .collect::<Vec<_>>();
        let reader = RecordBatchIterator::new(batches.into_iter(), schema);

        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let base_params =
            ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::Bitmap);
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        let selected_fragments = fragment_ids[..2].to_vec();
        let index =
            CreateIndexBuilder::new(&mut dataset, &["category"], IndexType::Bitmap, &base_params)
                .name("bitmap_segment".to_string())
                .fragments(selected_fragments.clone())
                .execute_uncommitted()
                .await
                .unwrap();

        assert_eq!(
            index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            selected_fragments
        );

        let files = index.files.as_ref().unwrap();
        assert!(files.iter().any(|file| file.path == BITMAP_LOOKUP_NAME));
        assert!(
            files.iter().all(|file| !file.path.starts_with("part_")),
            "staged bitmap segment should only reference canonical files"
        );
    }

    #[tokio::test]
    async fn test_vector_execute_uncommitted_segments_commit_without_staging() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );
        let mut input_segments = Vec::new();

        for fragment in &fragments {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                    .name("vector_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            let segment_index = dataset
                .indices_dir()
                .clone()
                .join(segment.uuid.to_string())
                .join(crate::index::INDEX_FILE_NAME);
            assert!(
                dataset
                    .object_store
                    .as_ref()
                    .exists(&segment_index)
                    .await
                    .unwrap()
            );
            input_segments.push(segment);
        }

        let segments = input_segments.clone();
        assert_eq!(segments.len(), fragments.len());
        let mut built_segment_ids = segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        built_segment_ids.sort();
        let mut input_segment_ids = input_segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        input_segment_ids.sort();
        assert_eq!(built_segment_ids, input_segment_ids);

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), fragments.len());

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_vector_commits_multi_segment_logical_index() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );
        let mut input_segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                    .name("vector_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            input_segments.push(segment);
        }

        let segments = input_segments;
        assert_eq!(segments.len(), 2);

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 2);
        let mut committed_fragment_sets = indices
            .iter()
            .map(|metadata| {
                metadata
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        committed_fragment_sets.sort();
        assert_eq!(committed_fragment_sets, vec![vec![0], vec![1]]);

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_rejects_vector_segments_without_index_details() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );
        let mut input_segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let mut segment =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                    .name("vector_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            segment.index_details = None;
            input_segments.push(segment);
        }

        let err = dataset
            .commit_existing_index_segments("vector_idx", "vector", input_segments)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("missing index details"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_segments_finalizes_fts_segments() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batch1 = create_text_batch(0, 10);
        let batch2 = create_text_batch(10, 20);
        let batch3 = create_text_batch(20, 30);

        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 5,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let mut input_segments = Vec::new();
        for fragment in dataset.get_fragments() {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("text_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            input_segments.push(segment);
        }

        let segments = input_segments.clone();
        assert_eq!(segments.len(), input_segments.len());

        dataset
            .commit_existing_index_segments("text_idx", "text", segments)
            .await
            .unwrap();

        for segment in &input_segments {
            let metadata_path = dataset
                .indices_dir()
                .clone()
                .join(segment.uuid.to_string())
                .join(lance_index::scalar::inverted::METADATA_FILE);
            assert!(
                dataset
                    .object_store
                    .as_ref()
                    .exists(&metadata_path)
                    .await
                    .unwrap()
            );
        }

        let indices = dataset.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(indices.len(), input_segments.len());
    }

    #[tokio::test]
    async fn test_merge_existing_index_segments_supports_fts_segments() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batches = RecordBatchIterator::new(
            vec![
                Ok(create_text_batch(0, 10)),
                Ok(create_text_batch(10, 20)),
                Ok(create_text_batch(20, 30)),
            ],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 5,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let mut input_segments = Vec::new();
        let mut expected_fragments = roaring::RoaringBitmap::new();
        for fragment in dataset.get_fragments() {
            expected_fragments.insert(fragment.id() as u32);
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("text_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            input_segments.push(segment);
        }

        let merged = dataset
            .merge_existing_index_segments(input_segments)
            .await
            .unwrap();
        assert_eq!(
            merged
                .fragment_bitmap
                .as_ref()
                .expect("merged FTS segment should have fragment coverage"),
            &expected_fragments
        );
        assert!(
            merged
                .index_details
                .as_ref()
                .expect("merged FTS segment should have index details")
                .type_url
                .ends_with("InvertedIndexDetails")
        );

        dataset
            .commit_existing_index_segments("text_idx", "text", vec![merged])
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(indices.len(), 1);

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("document".to_string()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 20);
    }

    #[tokio::test]
    async fn test_btree_merge_existing_index_segments() {
        use datafusion::common::ScalarValue;
        use lance_index::scalar::{SargableQuery, SearchResult};
        use std::ops::Bound;

        // Open `segment` and count rows whose `id` falls in `[lo, hi)`.
        async fn count_in_range(
            dataset: &Dataset,
            segment: &IndexMetadata,
            lo: i32,
            hi: i32,
        ) -> usize {
            let field_path = dataset.schema().field_path(segment.fields[0]).unwrap();
            let index = crate::index::scalar::open_scalar_index(
                dataset,
                &field_path,
                segment,
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
            let query = SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(lo))),
                Bound::Excluded(ScalarValue::Int32(Some(hi))),
            );
            match index.search(&query, &NoOpMetricsCollector).await.unwrap() {
                SearchResult::Exact(row_addrs) => {
                    row_addrs.true_rows().row_addrs().unwrap().count()
                }
                other => panic!("expected exact result, got {other:?}"),
            }
        }

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // 128 rows across two 64-row fragments. Stable row ids so the
        // retired-fragment filter below exercises the exact row-id allow-list.
        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(64),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // One staged BTree segment per fragment, committed as a multi-segment
        // logical index.
        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let mut staged = Vec::new();
        for fragment in dataset.get_fragments() {
            staged.push(
                CreateIndexBuilder::new(&mut dataset, &["id"], IndexType::BTree, &params)
                    .name("id_btree".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments("id_btree", "id", staged)
            .await
            .unwrap();

        // Phase 1 — healthy merge: the two per-fragment segments consolidate
        // into a single canonical segment covering both fragments, and a range
        // spanning both (ids 50..100) returns every matching row.
        let merged = dataset
            .merge_existing_index_segments(dataset.load_indices_by_name("id_btree").await.unwrap())
            .await
            .unwrap();
        assert_eq!(
            merged.fragment_bitmap.as_ref().unwrap(),
            &roaring::RoaringBitmap::from_iter([0u32, 1])
        );
        assert!(
            merged
                .index_details
                .as_ref()
                .unwrap()
                .type_url
                .ends_with("BTreeIndexDetails")
        );
        assert_eq!(count_in_range(&dataset, &merged, 50, 100).await, 50);

        // Phase 2 — retire fragment 0: delete >10% of its rows so compaction
        // rewrites only frag 0 (frag 1 has no deletions and is at target size).
        // The committed per-fragment segment now claims a fragment the dataset
        // no longer has.
        dataset.delete("id < 16").await.unwrap();
        crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 64,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let live_frags: roaring::RoaringBitmap = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id() as u32)
            .collect();
        assert!(!live_frags.contains(0), "compaction should retire frag 0");

        // Filtered merge: coverage drops the retired fragment but keeps the
        // live one, and the merged page data does not leak the retired row ids
        // (ids < 16 lived only in frag 0, so the range now returns nothing).
        let merged = dataset
            .merge_existing_index_segments(dataset.load_indices_by_name("id_btree").await.unwrap())
            .await
            .unwrap();
        let coverage = merged.fragment_bitmap.as_ref().unwrap();
        assert!(!coverage.contains(0), "must drop retired frag 0");
        assert!(coverage.contains(1), "must keep live frag 1");
        assert_eq!(
            count_in_range(&dataset, &merged, 0, 16).await,
            0,
            "must filter retired-fragment row ids"
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_supports_local_hnsw_segments() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(128),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let uuid = Uuid::new_v4();
        let params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
            HnswBuildParams::default(),
        );

        CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .index_uuid(uuid)
            .execute_uncommitted()
            .await
            .unwrap();

        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![IndexSegment::new(
                    uuid,
                    dataset.fragment_bitmap.as_ref().clone(),
                    Arc::new(vector_index_details(&params)),
                    IndexType::IvfHnswFlat.version(),
                )],
            )
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, uuid);
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
    }

    #[tokio::test]
    async fn test_create_index_vector_commits_with_segment_metadata() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(128),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );

        let committed = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        assert!(
            committed
                .files
                .as_ref()
                .is_some_and(|files| !files.is_empty()),
            "single-machine vector create_index should preserve committed file info"
        );

        let loaded = dataset.load_indices_by_name(&committed.name).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].uuid, committed.uuid);
        assert!(
            loaded[0]
                .files
                .as_ref()
                .is_some_and(|files| !files.is_empty()),
            "committed metadata loaded from the manifest should include file info"
        );
    }

    #[tokio::test]
    async fn test_create_index_ivf_rq_preserves_index_version_on_segment_commit_path() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(128),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_rq(4, 1, DistanceType::L2);

        let committed = dataset
            .create_index(&["vector"], IndexType::IvfRq, None, &params, false)
            .await
            .unwrap();

        assert_eq!(committed.index_version, IndexType::IvfRq.version());

        let loaded = dataset.load_indices_by_name(&committed.name).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].index_version, IndexType::IvfRq.version());
    }

    #[tokio::test]
    async fn test_optimize_should_not_removes_delta_indices() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let num_rows = 256;
        let reader = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(num_rows),
                lance_datagen::BatchCount::from(1),
            );

        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let vector_params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None, // Will auto-generate name "vector_idx"
                &vector_params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "Should have 1 index");
        assert_eq!(indices[0].name, "vector_idx");
        assert_eq!(indices[0].fragment_bitmap.as_ref().unwrap().len(), 1);
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().contains(0));

        // create again with replace=false
        let res = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None, // Will auto-generate name "vector_idx"
                &vector_params,
                false,
            )
            .await;
        assert!(res.is_err());

        // create again with replace=true
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None, // Will auto-generate name "vector_idx"
                &vector_params,
                true,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "Should have 1 index");
        assert_eq!(indices[0].name, "vector_idx");
        assert_eq!(indices[0].fragment_bitmap.as_ref().unwrap().len(), 1);
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().contains(0));

        let scalar_params =
            ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None, // Will auto-generate name "id_idx"
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2, "Should have 2 indices");

        let num_new_rows = 32;
        let new_reader = lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::step_custom::<Int32Type>(num_rows as i32, 1),
            )
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(num_new_rows),
                lance_datagen::BatchCount::from(1),
            );

        dataset = Dataset::write(
            new_reader,
            &dataset_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Load indices before optimization
        let indices_before = dataset.load_indices().await.unwrap();
        assert_eq!(indices_before.len(), 2, "Should still have 2 indices");

        // Optimize with num_indices_to_merge=0
        let optimize_options = OptimizeOptions::append();
        dataset.optimize_indices(&optimize_options).await.unwrap();

        // Load indices after optimization
        let indices_after = dataset.load_indices().await.unwrap();

        // After unifying scalar optimize, `OptimizeOptions::append()` honors
        // `Some(0)` for BTree the same way it does for vector: keep the old
        // segment, add a delta for the unindexed fragment. So we now expect:
        //   1. id_idx old segment, bitmap [0]
        //   2. id_idx delta segment, bitmap [1]
        //   3. vector_idx old segment, bitmap [0]
        //   4. vector_idx delta segment, bitmap [1]
        // Previously BTree silently merged into 1 segment because legacy
        // scalar ignored `num_indices_to_merge`.
        assert_eq!(indices_after.len(), 4, "{:?}", indices_after);
        let id_indices = indices_after
            .iter()
            .filter(|idx| idx.name == "id_idx")
            .collect::<Vec<_>>();
        let vector_indices = indices_after
            .iter()
            .filter(|idx| idx.name == "vector_idx")
            .collect::<Vec<_>>();
        for indices in [&id_indices, &vector_indices] {
            assert_eq!(indices.len(), 2);
            assert!(
                indices
                    .iter()
                    .any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(0)
                        && idx.fragment_bitmap.as_ref().unwrap().len() == 1)
            );
            assert!(
                indices
                    .iter()
                    .any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(1)
                        && idx.fragment_bitmap.as_ref().unwrap().len() == 1)
            );
        }
    }
}
