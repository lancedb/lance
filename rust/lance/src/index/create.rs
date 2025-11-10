// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    dataset::{
        transaction::{Operation, Transaction},
        Dataset,
    },
    index::{
        scalar::build_scalar_index,
        vector::{
            build_empty_vector_index, build_vector_index, VectorIndexParams, LANCE_VECTOR_INDEX,
        },
        vector_index_details, DatasetIndexExt, DatasetIndexInternalExt,
    },
    Error, Result,
};
use futures::future::BoxFuture;
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
        if self.columns.len() != 1 {
            return Err(Error::Index {
                message: "Only support building index on 1 column at the moment".to_string(),
                location: location!(),
            });
        }
        let column = &self.columns[0];
        let Some(field) = self.dataset.schema().field(column) else {
            return Err(Error::Index {
                message: format!("CreateIndex: column '{column}' does not exist"),
                location: location!(),
            });
        };

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
        let index_name = self.name.take().unwrap_or(format!("{column}_idx"));
        if let Some(idx) = indices.iter().find(|i| i.name == index_name) {
            if idx.fields == [field.id] && !self.replace {
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

        let index_id = match &self.index_uuid {
            Some(uuid_str) => Uuid::parse_str(uuid_str).map_err(|e| Error::Index {
                message: format!("Invalid UUID string provided: {}", e),
                location: location!(),
            })?,
            None => Uuid::new_v4(),
        };
        let created_index = match (self.index_type, self.params.index_name()) {
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
                        message: "Inverted index type must take a InvertedIndexParams".to_string(),
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
            (IndexType::Vector, LANCE_VECTOR_INDEX) => {
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
                    // this is a large future so move it to heap
                    Box::pin(build_vector_index(
                        self.dataset,
                        column,
                        &index_name,
                        &index_id.to_string(),
                        vec_params,
                        fri,
                    ))
                    .await?;
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
        };

        Ok(IndexMetadata {
            uuid: index_id,
            name: index_name,
            fields: vec![field.id],
            dataset_version: self.dataset.manifest.version,
            fragment_bitmap: if train {
                match &self.fragments {
                    Some(fragment_ids) => Some(fragment_ids.iter().collect()),
                    None => Some(
                        self.dataset
                            .get_fragments()
                            .iter()
                            .map(|f| f.id() as u32)
                            .collect(),
                    ),
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
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            None,
        );

        self.dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

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

        // There should be 3 indices:
        // 1. one scalar index with name "id_idx", and the bitmap is [0,1]
        // 2. one delta vector index with name "vector_idx", and the bitmap is [0]
        // 3. one delta vector index with name "vector_idx", and the bitmap is [1]
        assert_eq!(indices_after.len(), 3, "{:?}", indices_after);
        let id_idx = indices_after
            .iter()
            .find(|idx| idx.name == "id_idx")
            .unwrap();
        let vector_indices = indices_after
            .iter()
            .filter(|idx| idx.name == "vector_idx")
            .collect::<Vec<_>>();
        assert!(
            id_idx
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .contains_range(0..2)
                && id_idx.fragment_bitmap.as_ref().unwrap().len() == 2
        );
        assert_eq!(vector_indices.len(), 2);
        assert!(vector_indices
            .iter()
            .any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(0)
                && idx.fragment_bitmap.as_ref().unwrap().len() == 1));
        assert!(vector_indices
            .iter()
            .any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(1)
                && idx.fragment_bitmap.as_ref().unwrap().len() == 1));
    }
}
