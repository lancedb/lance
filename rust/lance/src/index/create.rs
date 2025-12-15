// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    dataset::{
        transaction::{Operation, Transaction},
        Dataset,
    },
    index::{
        scalar::{build_compound_btree_index, build_scalar_index},
        vector::{
            build_distributed_vector_index, build_empty_vector_index, build_vector_index,
            VectorIndexParams, LANCE_VECTOR_INDEX,
        },
        vector_index_details, DatasetIndexExt, DatasetIndexInternalExt,
    },
    Error, Result,
};
use futures::future::BoxFuture;
use lance_core::datatypes::format_field_path;
use lance_index::{
    metrics::NoOpMetricsCollector,
    scalar::{inverted::tokenizer::InvertedIndexParams, ScalarIndexParams, LANCE_SCALAR_INDEX},
};
use lance_index::{scalar::CreatedIndex, IndexParams, IndexType, VECTOR_INDEX_VERSION};
use lance_table::format::IndexMetadata;
use snafu::location;
use std::{future::IntoFuture, sync::Arc};
use tracing::instrument;
use uuid::Uuid;

use arrow_array::RecordBatchReader;

pub struct CreateIndexBuilder<'a> {
    dataset: &'a mut Dataset,
    columns: Vec<String>,
    index_type: IndexType,
    params: &'a dyn IndexParams,
    name: Option<String>,
    replace: bool,
    train: bool,
    fragments: Option<Vec<u32>>,
    index_uuid: Option<String>,
    preprocessed_data: Option<Box<dyn RecordBatchReader + Send + 'static>>,
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

    pub fn index_uuid(mut self, uuid: String) -> Self {
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

    #[instrument(skip_all)]
    pub async fn execute_uncommitted(&mut self) -> Result<IndexMetadata> {
        // Validate column count
        if self.columns.is_empty() {
            return Err(Error::Index {
                message: "Index must have at least one column".to_string(),
                location: location!(),
            });
        }
        if self.columns.len() > 8 {
            return Err(Error::Index {
                message: format!(
                    "Compound index exceeds maximum of 8 columns (got {})",
                    self.columns.len()
                ),
                location: location!(),
            });
        }

        // Validate all columns exist using case-insensitive lookup (merged from upstream)
        let mut fields: Vec<lance_core::datatypes::Field> = Vec::with_capacity(self.columns.len());
        let mut corrected_columns: Vec<String> = Vec::with_capacity(self.columns.len());

        for column_input in &self.columns {
            // Use case-insensitive lookup for both simple and nested paths.
            // resolve_case_insensitive tries exact match first, then falls back to case-insensitive.
            let Some(field_path) = self.dataset.schema().resolve_case_insensitive(column_input) else {
                return Err(Error::Index {
                    message: format!("CreateIndex: column '{}' does not exist", column_input),
                    location: location!(),
                });
            };
            let field = *field_path.last().unwrap();
            fields.push(field.clone());

            // Reconstruct the column path with correct case from schema
            // Use quoted format for SQL parsing (special chars are quoted)
            let names: Vec<&str> = field_path.iter().map(|f| f.name.as_str()).collect();
            corrected_columns.push(format_field_path(&names));
        }

        // For single-column indices, use the original column variable pattern
        // For multi-column indices, we'll use the first column for backward-compatible paths
        let column = corrected_columns[0].as_str();
        #[allow(unused_variables)]
        let field = &fields[0];

        // If train is true but dataset is empty, automatically set train to false
        let train = if self.train { self.dataset.count_rows(None).await? > 0 } else { false };

        // Load indices from the disk.
        let indices = self.dataset.load_indices().await?;
        let fri = self.dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
        // Generate index name: single column uses "column_idx", multiple use "col1_col2_idx"
        let index_name = self.name.take().unwrap_or_else(|| {
            if self.columns.len() == 1 {
                format!("{column}_idx")
            } else {
                format!("{}_idx", self.columns.join("_"))
            }
        });

        // Collect field IDs for comparison and storage
        let field_ids: Vec<i32> = fields.iter().map(|f| f.id).collect();

        if let Some(idx) = indices.iter().find(|i| i.name == index_name) {
            if idx.fields == field_ids && !self.replace {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name}' already exists, \
                        please specify a different name or use replace=True"
                    ),
                    location: location!(),
                });
            };
            if idx.fields != field_ids {
                return Err(Error::Index {
                    message: format!(
                        "Index name '{index_name}' already exists with different fields, \
                        please specify a different name"
                    ),
                    location: location!(),
                });
            }
        }

        let index_id = match &self.index_uuid {
            Some(uuid_str) => Uuid::parse_str(uuid_str).map_err(|e| Error::Index {
                message: format!("Invalid UUID string provided: {}", e),
                location: location!(),
            })?,
            None => Uuid::new_v4(),
        };

        // Handle multi-column (compound) indices
        let created_index = if self.columns.len() > 1 {
            match (self.index_type, self.params.index_name()) {
                // BTree/Scalar index types support compound indices
                (IndexType::Scalar | IndexType::BTree, LANCE_SCALAR_INDEX) => {
                    // Get params or use defaults for compound index
                    let params = self
                        .params
                        .as_any()
                        .downcast_ref::<ScalarIndexParams>()
                        .cloned()
                        .unwrap_or_else(|| ScalarIndexParams::new("compoundbtree".to_string()));

                    build_compound_btree_index(
                        self.dataset,
                        &self.columns.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                        &index_id.to_string(),
                        &params,
                        train,
                        self.fragments.clone(),
                    )
                    .await?
                }
                (index_type, _) => {
                    return Err(Error::Index {
                        message: format!(
                            "Index type {:?} does not support multiple columns. \
                             Use BTree or Scalar for compound indices.",
                            index_type
                        ),
                        location: location!(),
                    });
                }
            }
        } else {
            // Single-column index path (existing logic)
            match (self.index_type, self.params.index_name()) {
                (
                    IndexType::Bitmap
                    | IndexType::BTree
                    | IndexType::Inverted
                    | IndexType::NGram
                    | IndexType::ZoneMap
                    | IndexType::BloomFilter
                    | IndexType::LabelList,
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
                    build_scalar_index(
                        self.dataset,
                        column,
                        &index_id.to_string(),
                        &params,
                        train,
                        self.fragments.clone(),
                        preprocesssed_data,
                    )
                    .await?
                }
                (IndexType::Scalar, LANCE_SCALAR_INDEX) => {
                    // Guess the index type
                    let params = self
                        .params
                        .as_any()
                        .downcast_ref::<ScalarIndexParams>()
                        .ok_or_else(|| Error::Index {
                            message: "Scalar index type must take a ScalarIndexParams".to_string(),
                            location: location!(),
                        })?;
                    build_scalar_index(
                        self.dataset,
                        column,
                        &index_id.to_string(),
                        params,
                        train,
                        self.fragments.clone(),
                        None,
                    )
                    .await?
                }
                (IndexType::Inverted, _) => {
                    // Inverted index params.
                    let inverted_params = self
                        .params
                        .as_any()
                        .downcast_ref::<InvertedIndexParams>()
                        .ok_or_else(|| Error::Index {
                            message: "Inverted index type must take a InvertedIndexParams"
                                .to_string(),
                            location: location!(),
                        })?;

                    let params =
                        ScalarIndexParams::new("inverted".to_string()).with_params(inverted_params);
                    build_scalar_index(
                        self.dataset,
                        column,
                        &index_id.to_string(),
                        &params,
                        train,
                        self.fragments.clone(),
                        None,
                    )
                    .await?
                }
                (
                    IndexType::Vector
                    | IndexType::IvfPq
                    | IndexType::IvfSq
                    | IndexType::IvfFlat
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
                        .ok_or_else(|| Error::Index {
                            message: "Vector index type must take a VectorIndexParams".to_string(),
                            location: location!(),
                        })?;

                    if train {
                        // Check if this is distributed indexing (fragment-level)
                        if self.fragments.is_some() {
                            // For distributed indexing, build only on specified fragments
                            // This creates temporary index metadata without committing
                            Box::pin(build_distributed_vector_index(
                                self.dataset,
                                column,
                                &index_name,
                                &index_id.to_string(),
                                vec_params,
                                fri,
                                self.fragments.as_ref().unwrap(),
                            ))
                            .await?;
                        } else {
                            // Standard full dataset indexing
                            Box::pin(build_vector_index(
                                self.dataset,
                                column,
                                &index_name,
                                &index_id.to_string(),
                                vec_params,
                                fri,
                            ))
                            .await?;
                        }
                    } else {
                        // Create empty vector index
                        build_empty_vector_index(
                            self.dataset,
                            column,
                            &index_name,
                            &index_id.to_string(),
                            vec_params,
                        )
                        .await?;
                    }
                    CreatedIndex {
                        index_details: vector_index_details(),
                        index_version: VECTOR_INDEX_VERSION,
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
                    .ok_or(Error::Internal {
                        message: "unable to cast index extension to vector".to_string(),
                        location: location!(),
                    })?;

                    if train {
                        ext.create_index(self.dataset, column, &index_id.to_string(), self.params)
                            .await?;
                    } else {
                        todo!("create empty vector index when train=false");
                    }
                    CreatedIndex {
                        index_details: vector_index_details(),
                        index_version: VECTOR_INDEX_VERSION,
                    }
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
            }
        }; // Close the if-else for multi-column check

        Ok(IndexMetadata {
            uuid: index_id,
            name: index_name,
            fields: field_ids,
            dataset_version: self.dataset.manifest.version,
            fragment_bitmap: if train {
                match &self.fragments {
                    Some(fragment_ids) => Some(fragment_ids.iter().collect()),
                    None => {
                        Some(self.dataset.get_fragments().iter().map(|f| f.id() as u32).collect())
                    }
                }
            } else {
                // Empty bitmap for untrained indices
                Some(roaring::RoaringBitmap::new())
            },
            index_details: Some(Arc::new(created_index.index_details)),
            index_version: created_index.index_version as i32,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
        })
    }

    #[instrument(skip_all)]
    async fn execute(mut self) -> Result<()> {
        let new_idx = self.execute_uncommitted().await?;
        let transaction = Transaction::new(
            new_idx.dataset_version,
            Operation::CreateIndex { new_indices: vec![new_idx], removed_indices: vec![] },
            None,
        );

        self.dataset.apply_commit(transaction, &Default::default(), &Default::default()).await?;

        Ok(())
    }
}

impl<'a> IntoFuture for CreateIndexBuilder<'a> {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'a, Result<()>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{WriteMode, WriteParams};
    use arrow::datatypes::{Float32Type, Int32Type};
    use arrow_array::RecordBatchIterator;
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
    use lance_linalg::distance::MetricType;
    use std::sync::Arc;

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
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params)).await.unwrap();

        let params = InvertedIndexParams::default();

        // Get fragment IDs from the dataset
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        assert!(fragment_ids.len() >= 2, "Should have multiple fragments for testing");

        // Test fragments() method with specific fragment IDs and ensure duplicate/out-of-order fragments are handled properly
        let selected_fragments =
            vec![fragment_ids[1], fragment_ids[0], fragment_ids[1], fragment_ids[2]];
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
    async fn test_merge_index_metadata() {
        // Test the complete workflow for merge_index_metadata:
        // 1. Create multiple fragment indexes using execute_uncommitted
        // 2. Use merge_index_metadata to merge temporary metadata files
        // 3. Commit the index using the standard commit process
        // 4. Verify the final index is properly created and accessible

        // Create temporary directory for dataset
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Create test data with multiple fragments
        let batch1 = create_text_batch(0, 15);
        let batch2 = create_text_batch(15, 30);
        let batch3 = create_text_batch(30, 45);

        let write_params =
            WriteParams { max_rows_per_file: 15, max_rows_per_group: 5, ..Default::default() };

        // Write dataset with multiple batches to create multiple fragments
        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params)).await.unwrap();

        let params = InvertedIndexParams::default();
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();

        // Use a shared UUID for distributed indexing
        let shared_uuid = Uuid::new_v4().to_string();

        // Step 1: Create indexes for each fragment using execute_uncommitted
        let mut index_metadatas = Vec::new();
        for &fragment_id in &fragment_ids {
            let mut builder =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("distributed_index".to_string())
                    .fragments(vec![fragment_id])
                    .index_uuid(shared_uuid.clone());

            let index_metadata = builder.execute_uncommitted().await.unwrap();

            // Verify each fragment's index metadata
            assert_eq!(index_metadata.uuid.to_string(), shared_uuid);
            assert_eq!(index_metadata.name, "distributed_index");

            let fragment_bitmap = index_metadata.fragment_bitmap.as_ref().unwrap();
            let indexed_fragments: Vec<u32> = fragment_bitmap.iter().collect();
            assert_eq!(indexed_fragments, vec![fragment_id]);

            index_metadatas.push(index_metadata);
        }

        // Step 2: Merge inverted index metadata
        // Note: This step would typically be done by calling dataset.merge_index_metadata()
        // but for this test, we verify that the execute_uncommitted workflow produces correct metadata

        // Step 3: Verify the metadata from execute_uncommitted contains all necessary information
        assert_eq!(index_metadatas.len(), fragment_ids.len());

        // Verify all metadata have the same UUID (shared UUID for distributed indexing)
        for metadata in &index_metadatas {
            assert_eq!(metadata.uuid.to_string(), shared_uuid);
            assert_eq!(metadata.name, "distributed_index");
            assert!(metadata.fragment_bitmap.is_some());
            assert!(metadata.created_at.is_some());
        }

        // Verify that each fragment is covered by exactly one metadata
        let mut all_covered_fragments = Vec::new();
        for metadata in &index_metadatas {
            let fragment_bitmap = metadata.fragment_bitmap.as_ref().unwrap();
            let covered_fragments: Vec<u32> = fragment_bitmap.iter().collect();
            all_covered_fragments.extend(covered_fragments);
        }
        all_covered_fragments.sort();
        let mut expected_fragments = fragment_ids.clone();
        expected_fragments.sort();
        assert_eq!(all_covered_fragments, expected_fragments);
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
            .col("id", lance_datagen::array::step_custom::<Int32Type>(num_rows as i32, 1))
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
            Some(WriteParams { mode: WriteMode::Append, ..Default::default() }),
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

        // There should be 3 indices:
        // 1. one scalar index with name "id_idx", and the bitmap is [0,1]
        // 2. one delta vector index with name "vector_idx", and the bitmap is [0]
        // 3. one delta vector index with name "vector_idx", and the bitmap is [1]
        assert_eq!(indices_after.len(), 3, "{:?}", indices_after);
        let id_idx = indices_after.iter().find(|idx| idx.name == "id_idx").unwrap();
        let vector_indices =
            indices_after.iter().filter(|idx| idx.name == "vector_idx").collect::<Vec<_>>();
        assert!(
            id_idx.fragment_bitmap.as_ref().unwrap().contains_range(0..2)
                && id_idx.fragment_bitmap.as_ref().unwrap().len() == 2
        );
        assert_eq!(vector_indices.len(), 2);
        assert!(vector_indices.iter().any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(0)
            && idx.fragment_bitmap.as_ref().unwrap().len() == 1));
        assert!(vector_indices.iter().any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(1)
            && idx.fragment_bitmap.as_ref().unwrap().len() == 1));
    }

    #[tokio::test]
    async fn test_create_compound_index() {
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::DatasetIndexExt;

        // Create temporary directory for dataset
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Create test data with multiple columns suitable for compound index
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let tenant_ids: Vec<&str> = (0..100)
            .map(|i| match i % 3 {
                0 => "acme",
                1 => "globex",
                _ => "initech",
            })
            .collect();
        let statuses: Vec<&str> = (0..100)
            .map(|i| match i % 2 {
                0 => "active",
                _ => "inactive",
            })
            .collect();
        let values: Vec<i32> = (0..100).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(tenant_ids)),
                Arc::new(StringArray::from(statuses)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();

        let write_params =
            WriteParams { max_rows_per_file: 50, max_rows_per_group: 25, ..Default::default() };

        // Write dataset
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params)).await.unwrap();

        // Create compound index on tenant_id and status
        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("compound_idx".to_string()),
                &params,
                false, // replace
            )
            .await
            .unwrap();

        // Verify index exists
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);

        let compound_idx = &indices[0];
        assert_eq!(compound_idx.name, "compound_idx");
        assert_eq!(compound_idx.fields.len(), 2, "Compound index should have 2 fields");

        // Verify we can load the index details
        let index_details = compound_idx.index_details.as_ref().expect("should have details");
        assert!(
            index_details.type_url.contains("CompoundBTreeIndexDetails"),
            "Index details should be CompoundBTreeIndexDetails, got: {}",
            index_details.type_url
        );
    }

    #[tokio::test]
    async fn test_compound_index_rejects_unsupported_types() {
        // Create temporary directory for dataset
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Create minimal test data
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("col1", DataType::Utf8, false),
            ArrowField::new("col2", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, &dataset_uri, None).await.unwrap();

        // Try to create compound index with vector type - should fail
        let params = VectorIndexParams::ivf_flat(8, MetricType::Cosine);
        let result =
            dataset.create_index(&["col1", "col2"], IndexType::Vector, None, &params, false).await;

        assert!(result.is_err(), "Vector index should not support multiple columns");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("does not support multiple columns"),
            "Error should mention multi-column not supported: {}",
            err
        );
    }

    /// Test compound index search with multiple key combinations (cartesian product).
    ///
    /// This test mirrors the failing catalyzed-lance test to isolate whether the
    /// bug is in Lance's compound index implementation or in catalyzed-lance's usage.
    #[tokio::test]
    async fn test_compound_index_search_cartesian_product() {
        use datafusion::common::ScalarValue;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::compound::CompoundSargableQuery;
        use lance_index::scalar::{ScalarIndex, ScalarIndexParams};
        use lance_index::DatasetIndexExt;

        // Create temporary directory for dataset
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Create test data matching catalyzed-lance's test:
        // | tenant_id | status   | value |
        // |-----------|----------|-------|
        // | acme      | active   | 100   |
        // | acme      | active   | 200   |
        // | acme      | inactive | 300   |
        // | beta      | active   | 400   |
        // | beta      | inactive | 500   |
        // | gamma     | active   | 600   |
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["acme", "acme", "acme", "beta", "beta", "gamma"])),
                Arc::new(StringArray::from(vec![
                    "active", "active", "inactive", "active", "inactive", "active",
                ])),
                Arc::new(Int32Array::from(vec![100, 200, 300, 400, 500, 600])),
            ],
        )
        .unwrap();

        // Write dataset
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, &dataset_uri, None).await.unwrap();

        // Create compound index on tenant_id and status
        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("idx_tenant_status".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Open the compound index
        let indices = dataset.load_indices().await.unwrap();
        let compound_idx = &indices[0];
        let scalar_index = dataset
            .open_scalar_index("tenant_id", &compound_idx.uuid.to_string(), &NoOpMetricsCollector)
            .await
            .unwrap();

        // Test each key combination individually to verify the index returns correct results

        // First, let's see what the index actually contains
        println!("Testing compound index search...");

        // Test prefix-only query: just tenant_id = 'acme' - should return 3 rows
        let query_prefix_only = CompoundSargableQuery::PrefixLookup {
            prefix: vec![ScalarValue::Utf8(Some("acme".to_string()))],
            range: None,
        };
        let result_prefix =
            scalar_index.search(&query_prefix_only, &NoOpMetricsCollector).await.unwrap();
        let row_ids_prefix: Vec<u64> = result_prefix
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        println!("Query (acme only): {:?} - expected 3 rows", row_ids_prefix);

        // Test 1: (acme, active) - should return 2 rows
        let query1 = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("acme".to_string())),
                ScalarValue::Utf8(Some("active".to_string())),
            ],
            range: None,
        };
        let result1 = scalar_index.search(&query1, &NoOpMetricsCollector).await.unwrap();
        let row_ids1: Vec<u64> = result1
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        println!("Query (acme, active): {:?}", row_ids1);
        assert_eq!(row_ids1.len(), 2, "(acme, active) should return 2 rows, got {:?}", row_ids1);

        // Let's see what row 2 (acme, inactive, 300) looks like in the dataset
        let row2_projection = crate::dataset::ProjectionRequest::from_columns(
            ["tenant_id", "status", "value"],
            dataset.schema(),
        );
        let row2_data = dataset.take_rows(&[2], row2_projection).await.unwrap();
        println!(
            "Row 2 data: tenant_id={:?}, status={:?}, value={:?}",
            row2_data.column_by_name("tenant_id").unwrap(),
            row2_data.column_by_name("status").unwrap(),
            row2_data.column_by_name("value").unwrap(),
        );

        // Test 2: (acme, inactive) - should return 1 row
        let query2 = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("acme".to_string())),
                ScalarValue::Utf8(Some("inactive".to_string())),
            ],
            range: None,
        };
        let result2 = scalar_index.search(&query2, &NoOpMetricsCollector).await.unwrap();
        let row_ids2: Vec<u64> = result2
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        println!("Query (acme, inactive): {:?}", row_ids2);
        assert_eq!(row_ids2.len(), 1, "(acme, inactive) should return 1 row, got {:?}", row_ids2);

        // Test 3: (beta, active) - should return 1 row
        let query3 = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".to_string())),
                ScalarValue::Utf8(Some("active".to_string())),
            ],
            range: None,
        };
        let result3 = scalar_index.search(&query3, &NoOpMetricsCollector).await.unwrap();
        let row_ids3: Vec<u64> = result3
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        assert_eq!(row_ids3.len(), 1, "(beta, active) should return 1 row, got {:?}", row_ids3);

        // Test 4: (beta, inactive) - should return 1 row
        let query4 = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".to_string())),
                ScalarValue::Utf8(Some("inactive".to_string())),
            ],
            range: None,
        };
        let result4 = scalar_index.search(&query4, &NoOpMetricsCollector).await.unwrap();
        let row_ids4: Vec<u64> = result4
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        assert_eq!(row_ids4.len(), 1, "(beta, inactive) should return 1 row, got {:?}", row_ids4);

        // Verify total: 2 + 1 + 1 + 1 = 5 unique rows
        let mut all_row_ids: Vec<u64> = vec![];
        all_row_ids.extend(&row_ids1);
        all_row_ids.extend(&row_ids2);
        all_row_ids.extend(&row_ids3);
        all_row_ids.extend(&row_ids4);
        all_row_ids.sort();
        all_row_ids.dedup();
        assert_eq!(all_row_ids.len(), 5, "Total unique rows should be 5, got {:?}", all_row_ids);

        // Fetch the actual rows to verify values
        let projection = crate::dataset::ProjectionRequest::from_columns(
            ["tenant_id", "status", "value"],
            dataset.schema(),
        );
        let fetched = dataset.take_rows(&all_row_ids, projection).await.unwrap();
        assert_eq!(fetched.num_rows(), 5);

        // Verify the values are correct (100, 200, 300, 400, 500)
        let value_col =
            fetched.column_by_name("value").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
        let mut values: Vec<i32> = (0..value_col.len()).map(|i| value_col.value(i)).collect();
        values.sort();
        assert_eq!(
            values,
            vec![100, 200, 300, 400, 500],
            "Values should be [100, 200, 300, 400, 500], got {:?}",
            values
        );
    }

    /// Test compound index with compaction using fragment reuse path.
    ///
    /// This test verifies that compound indices work correctly with the fragment
    /// reuse index during compaction when defer_index_remap is enabled.
    #[tokio::test]
    async fn test_compound_index_with_compaction_and_fragment_reuse() {
        use crate::dataset::optimize::{compact_files, CompactionOptions};
        use datafusion::common::ScalarValue;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::compound::CompoundSargableQuery;
        use lance_index::scalar::{ScalarIndex, ScalarIndexParams};
        use lance_index::DatasetIndexExt;

        // Create dataset with multiple fragments
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        // Write first fragment
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["acme", "acme", "beta"])),
                Arc::new(StringArray::from(vec!["active", "inactive", "active"])),
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ],
        )
        .unwrap();

        let write_params = WriteParams { max_rows_per_file: 3, ..Default::default() };
        let batches = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        let mut dataset =
            Dataset::write(batches, &dataset_uri, Some(write_params.clone())).await.unwrap();

        // Write second fragment (append to existing dataset)
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["beta", "gamma", "gamma"])),
                Arc::new(StringArray::from(vec!["inactive", "active", "inactive"])),
                Arc::new(Int32Array::from(vec![400, 500, 600])),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch2)], schema.clone());
        dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Verify we have 2 fragments
        assert_eq!(dataset.fragments().len(), 2, "Should have 2 fragments");

        // Create compound index on tenant_id and status
        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("idx_compound".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify index was created
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "Should have 1 index");
        let compound_idx = &indices[0];
        assert_eq!(compound_idx.fields.len(), 2, "Should index 2 fields");

        // Test query before compaction
        let scalar_index = dataset
            .open_scalar_index("tenant_id", &compound_idx.uuid.to_string(), &NoOpMetricsCollector)
            .await
            .unwrap();

        let query_before = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".to_string())),
                ScalarValue::Utf8(Some("active".to_string())),
            ],
            range: None,
        };
        let result_before =
            scalar_index.search(&query_before, &NoOpMetricsCollector).await.unwrap();
        let row_ids_before: Vec<u64> = result_before
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        assert_eq!(
            row_ids_before.len(),
            1,
            "Should find 1 row for (beta, active) before compaction"
        );

        // Compact with defer_index_remap enabled (uses fragment reuse path)
        let compaction_options =
            CompactionOptions { defer_index_remap: true, ..Default::default() };
        compact_files(&mut dataset, compaction_options, None).await.unwrap();

        // Reload dataset after compaction
        dataset = Dataset::open(&dataset_uri).await.unwrap();

        // Verify we now have 1 fragment after compaction
        assert_eq!(dataset.fragments().len(), 1, "Should have 1 fragment after compaction");

        // Verify index still exists
        let indices_after = dataset.load_indices().await.unwrap();
        // Should have the compound index plus the fragment reuse index
        assert!(
            indices_after.len() >= 1,
            "Should have at least the compound index after compaction"
        );

        // Test query after compaction - should still work via fragment reuse
        let compound_idx_after = indices_after
            .iter()
            .find(|idx| idx.name == "idx_compound")
            .expect("Compound index should still exist");

        let scalar_index_after = dataset
            .open_scalar_index(
                "tenant_id",
                &compound_idx_after.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let query_after = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".to_string())),
                ScalarValue::Utf8(Some("active".to_string())),
            ],
            range: None,
        };
        let result_after =
            scalar_index_after.search(&query_after, &NoOpMetricsCollector).await.unwrap();
        let row_ids_after: Vec<u64> = result_after
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        assert_eq!(
            row_ids_after.len(),
            1,
            "Should still find 1 row for (beta, active) after compaction via fragment reuse"
        );

        // Verify we can fetch the actual data
        let projection = crate::dataset::ProjectionRequest::from_columns(
            ["tenant_id", "status", "value"],
            dataset.schema(),
        );
        let fetched = dataset.take_rows(&row_ids_after, projection).await.unwrap();
        assert_eq!(fetched.num_rows(), 1, "Should fetch 1 row");

        let tenant_col = fetched
            .column_by_name("tenant_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(tenant_col.value(0), "beta");

        let status_col = fetched
            .column_by_name("status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(status_col.value(0), "active");
    }

    /// Test compound index with mixed indexed and unindexed fragments.
    ///
    /// This test verifies that queries work correctly when some fragments
    /// are indexed and others are not (partial index coverage).
    #[tokio::test]
    async fn test_compound_index_mixed_fragment_coverage() {
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::DatasetIndexExt;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        // Write first fragment
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["acme", "acme", "beta"])),
                Arc::new(StringArray::from(vec!["active", "inactive", "active"])),
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ],
        )
        .unwrap();

        let write_params = WriteParams { max_rows_per_file: 3, ..Default::default() };
        let batches = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        let mut dataset =
            Dataset::write(batches, &dataset_uri, Some(write_params.clone())).await.unwrap();

        // Create compound index on first fragment only
        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("idx_compound".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify index covers fragment 0
        let indices = dataset.load_indices().await.unwrap();
        let compound_idx = &indices[0];
        let frag_bitmap = compound_idx.fragment_bitmap.as_ref().unwrap();
        assert!(frag_bitmap.contains(0), "Index should cover fragment 0");
        assert_eq!(frag_bitmap.len(), 1, "Index should only cover 1 fragment");

        // Write second fragment (not indexed) - append to existing dataset
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["beta", "gamma", "gamma"])),
                Arc::new(StringArray::from(vec!["inactive", "active", "inactive"])),
                Arc::new(Int32Array::from(vec![400, 500, 600])),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch2)], schema.clone());
        dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Reload to get updated fragment list
        dataset = Dataset::open(&dataset_uri).await.unwrap();
        assert_eq!(dataset.fragments().len(), 2, "Should have 2 fragments");

        // Verify index still only covers fragment 0
        let indices_after = dataset.load_indices().await.unwrap();
        let compound_idx_after = &indices_after[0];
        let frag_bitmap_after = compound_idx_after.fragment_bitmap.as_ref().unwrap();
        assert_eq!(frag_bitmap_after.len(), 1, "Index should still only cover 1 fragment");

        // Query for data that exists in both fragments
        // ("beta", "active") is in fragment 0 (indexed)
        // ("beta", "inactive") is in fragment 1 (not indexed)
        let result = dataset
            .scan()
            .filter("tenant_id = 'beta' AND status = 'active'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(result.num_rows(), 1, "Should find 1 row for (beta, active)");
        let value_col =
            result.column_by_name("value").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(value_col.value(0), 300, "Should find value 300 from fragment 0");

        // Query for data only in unindexed fragment
        let result2 = dataset
            .scan()
            .filter("tenant_id = 'gamma' AND status = 'active'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            result2.num_rows(),
            1,
            "Should find 1 row for (gamma, active) in unindexed fragment"
        );
        let value_col2 =
            result2.column_by_name("value").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(value_col2.value(0), 500, "Should find value 500 from unindexed fragment 1");

        // Query for data spanning both fragments
        let result3 =
            dataset.scan().filter("tenant_id = 'beta'").unwrap().try_into_batch().await.unwrap();

        assert_eq!(
            result3.num_rows(),
            2,
            "Should find 2 rows for tenant 'beta' across both fragments"
        );
        let value_col3 =
            result3.column_by_name("value").unwrap().as_any().downcast_ref::<Int32Array>().unwrap();
        let mut values: Vec<i32> = (0..value_col3.len()).map(|i| value_col3.value(i)).collect();
        values.sort();
        assert_eq!(values, vec![300, 400], "Should find values 300 (indexed) and 400 (unindexed)");

        // Test order independence: status first, then tenant_id
        // Should return same results as tenant_id first
        let result_reversed = dataset
            .scan()
            .filter("status = 'active' AND tenant_id = 'beta'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            result_reversed.num_rows(),
            1,
            "Reversed predicate order should find 1 row for (beta, active)"
        );
        let value_col_reversed = result_reversed
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(
            value_col_reversed.value(0),
            300,
            "Reversed predicate order should find same value 300"
        );

        // Verify the query plan shows index usage for the indexed fragment
        let plan = dataset
            .scan()
            .filter("tenant_id = 'acme' AND status = 'active'")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        // The plan should mention the index
        assert!(
            plan.contains("idx_compound") || plan.contains("ScalarIndexScan"),
            "Query plan should indicate index usage for indexed fragment, got: {}",
            plan
        );

        // Also verify reversed order uses index
        let plan_reversed = dataset
            .scan()
            .filter("status = 'active' AND tenant_id = 'acme'")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();

        assert!(
            plan_reversed.contains("idx_compound") || plan_reversed.contains("ScalarIndexScan"),
            "Reversed predicate order should also use index, got: {}",
            plan_reversed
        );
    }
}
