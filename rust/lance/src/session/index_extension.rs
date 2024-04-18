// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::Result;
use lance_file::reader::FileReader;

use crate::{index::vector::VectorIndex, Dataset};

#[async_trait::async_trait]
pub trait VectorIndexExtension: Send + Sync {
    /// TODO: add create_index and optimize_index methods

    /// Load a vector index from a file.
    async fn load_index(
        &self,
        dataset: Arc<Dataset>,
        column: &str,
        uuid: &str,
        reader: FileReader,
    ) -> Result<Arc<dyn VectorIndex>>;
}

#[cfg(test)]
mod test {
    use crate::{
        dataset::{
            builder::DatasetBuilder,
            scanner::test_dataset::TestVectorDataset,
            transaction::{Operation, Transaction},
        },
        index::{DatasetIndexInternalExt, PreFilter},
        io::commit::commit_transaction,
        session::Session,
    };

    use super::*;

    use std::{any::Any, collections::HashMap, sync::Arc};

    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
    use lance_file::writer::{FileWriter, FileWriterOptions};
    use lance_index::{
        vector::{hnsw::VECTOR_ID_FIELD, Query},
        DatasetIndexExt, Index, IndexMetadata, IndexType, INDEX_FILE_NAME,
        INDEX_METADATA_SCHEMA_KEY,
    };
    use lance_io::traits::Reader;
    use lance_linalg::distance::MetricType;
    use lance_table::io::manifest::ManifestDescribing;
    use roaring::RoaringBitmap;
    use serde_json::json;
    use uuid::Uuid;

    #[derive(Debug)]
    struct MockIndex;

    #[async_trait::async_trait]
    impl Index for MockIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(json!(()))
        }

        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::new())
        }
    }

    #[async_trait::async_trait]
    impl VectorIndex for MockIndex {
        async fn search(&self, _: &Query, _: Arc<PreFilter>) -> Result<RecordBatch> {
            todo!("panic")
        }

        fn is_loadable(&self) -> bool {
            true
        }

        fn use_residual(&self) -> bool {
            true
        }

        fn check_can_remap(&self) -> Result<()> {
            Ok(())
        }

        async fn load(
            &self,
            _: Arc<dyn Reader>,
            _: usize,
            _: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            todo!("panic")
        }

        fn remap(&mut self, _: &HashMap<u64, Option<u64>>) -> Result<()> {
            Ok(())
        }

        fn metric_type(&self) -> MetricType {
            MetricType::L2
        }
    }

    struct MockIndexExtension {}

    #[async_trait::async_trait]
    impl VectorIndexExtension for MockIndexExtension {
        async fn load_index(
            &self,
            _dataset: Arc<Dataset>,
            _column: &str,
            _uuid: &str,
            _reader: FileReader,
        ) -> Result<Arc<dyn VectorIndex>> {
            Ok(Arc::new(MockIndex))
        }
    }

    async fn make_empty_index(
        dataset: &mut Dataset,
        index_type: &str,
        index_uuid: &Uuid,
        column: &str,
    ) {
        // write an index
        let store = dataset.object_store.clone();
        let path = dataset
            .indices_dir()
            .child(index_uuid.to_string())
            .child(INDEX_FILE_NAME);
        let writer = store.create(&path).await.unwrap();

        let arrow_schema = Arc::new(Schema::new(vec![VECTOR_ID_FIELD.clone()]));
        let schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut writer: FileWriter<ManifestDescribing> =
            FileWriter::with_object_writer(writer, schema, &FileWriterOptions::default()).unwrap();
        writer.add_metadata(
            INDEX_METADATA_SCHEMA_KEY,
            json!(IndexMetadata {
                index_type: index_type.to_string(),
                distance_type: "cosine".to_string(),
            })
            .to_string()
            .as_str(),
        );

        writer
            .write(&[RecordBatch::new_empty(arrow_schema)])
            .await
            .unwrap();
        writer.finish().await.unwrap();

        // check in the metadat to point at the index

        let field = dataset.schema().field(column).unwrap();

        let new_idx = lance_table::format::Index {
            uuid: *index_uuid,
            name: "test".to_string(),
            fields: vec![field.id],
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(
                dataset
                    .get_fragments()
                    .iter()
                    .map(|f| f.id() as u32)
                    .collect(),
            ),
        };

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_idx],
                removed_indices: vec![],
            },
            None,
        );

        let new_manifest = commit_transaction(
            dataset,
            dataset.object_store(),
            dataset.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

        dataset.manifest = Arc::new(new_manifest);
    }

    #[tokio::test]
    async fn test_vector_index_extension() {
        // make dataset and index that is not supported natively
        let mut test_ds = TestVectorDataset::new().await.unwrap();

        let idx = test_ds.dataset.load_indices().await.unwrap();
        assert_eq!(idx.len(), 0);

        let index_uuid = Uuid::new_v4();

        make_empty_index(&mut test_ds.dataset, "TEST", &index_uuid, "vec").await;

        let idx = test_ds.dataset.load_indices().await.unwrap();
        assert_eq!(idx.len(), 1);

        // trying to open the index should fail as there is no extension loader
        assert!(test_ds
            .dataset
            .open_vector_index("vec", &index_uuid.to_string())
            .await
            .unwrap_err()
            .to_string()
            .contains("Unsupported index type: TEST"));

        // make a session with extension loader
        let mut session = Session::default();
        session
            .register_vector_index_extension("TEST".into(), Arc::new(MockIndexExtension {}))
            .unwrap();

        let ds_with_extension = DatasetBuilder::from_uri(test_ds.tmp_dir.path().to_str().unwrap())
            .with_session(Arc::new(session))
            .load()
            .await
            .unwrap();

        let vector_index = ds_with_extension
            .open_vector_index("vec", &index_uuid.to_string())
            .await
            .unwrap();

        // should be able to downcast to the mock index
        let _downcasted = vector_index.as_any().downcast_ref::<MockIndex>().unwrap();
    }
}
