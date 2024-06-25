// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_file::reader::FileReader;
use lance_index::{vector::VectorIndex, IndexParams, IndexType};

use crate::Dataset;

pub trait IndexExtension: Send + Sync + DeepSizeOf {
    fn index_type(&self) -> IndexType;

    // TODO: this shouldn't exist, as upcasting should be well defined
    // fix after https://github.com/rust-lang/rust/issues/65991
    fn to_generic(self: Arc<Self>) -> Arc<dyn IndexExtension>;

    fn to_scalar(self: Arc<Self>) -> Option<Arc<dyn ScalarIndexExtension>>;

    fn to_vector(self: Arc<Self>) -> Option<Arc<dyn VectorIndexExtension>>;
}

pub trait ScalarIndexExtension: IndexExtension {
    // TODO: implement this trait and wire it in
}

#[async_trait::async_trait]
pub trait VectorIndexExtension: IndexExtension {
    async fn create_index(
        &self,
        // Can't use Arc<Dataset> here
        // because we need &mut Dataset to call `create_index`
        // if we wrap into an Arc, the mutable reference is lost
        dataset: &Dataset,
        column: &str,
        uuid: &str,
        params: &dyn IndexParams,
    ) -> Result<()>;

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
        dataset::{builder::DatasetBuilder, scanner::test_dataset::TestVectorDataset},
        index::{DatasetIndexInternalExt, PreFilter},
        session::Session,
    };

    use super::*;

    use std::{
        any::Any,
        collections::HashMap,
        sync::{atomic::AtomicBool, Arc},
    };

    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::Schema;
    use deepsize::DeepSizeOf;
    use lance_file::writer::{FileWriter, FileWriterOptions};
    use lance_index::vector::ivf::storage::IvfModel;
    use lance_index::vector::quantizer::{QuantizationType, Quantizer};
    use lance_index::vector::v3::subindex::SubIndexType;
    use lance_index::{
        vector::{hnsw::VECTOR_ID_FIELD, Query},
        DatasetIndexExt, Index, IndexMetadata, IndexType, INDEX_FILE_NAME,
        INDEX_METADATA_SCHEMA_KEY,
    };
    use lance_io::traits::Reader;
    use lance_linalg::distance::MetricType;
    use lance_table::io::manifest::ManifestDescribing;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use serde_json::json;

    #[derive(Debug)]
    struct MockIndex;

    impl DeepSizeOf for MockIndex {
        fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
            todo!()
        }
    }

    #[async_trait::async_trait]
    impl Index for MockIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
            Ok(self)
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
        async fn search(&self, _: &Query, _: Arc<dyn PreFilter>) -> Result<RecordBatch> {
            unimplemented!()
        }

        fn find_partitions(&self, _: &Query) -> Result<UInt32Array> {
            unimplemented!()
        }

        async fn search_in_partition(
            &self,
            _: usize,
            _: &Query,
            _: Arc<dyn PreFilter>,
        ) -> Result<RecordBatch> {
            unimplemented!()
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
            unimplemented!()
        }

        fn row_ids(&self) -> Box<dyn Iterator<Item = &u64>> {
            unimplemented!()
        }

        fn remap(&mut self, _: &HashMap<u64, Option<u64>>) -> Result<()> {
            Ok(())
        }

        fn ivf_model(&self) -> IvfModel {
            unimplemented!()
        }
        fn quantizer(&self) -> Quantizer {
            unimplemented!()
        }

        /// the index type of this vector index.
        fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
            unimplemented!()
        }

        fn metric_type(&self) -> MetricType {
            MetricType::L2
        }
    }

    struct MockIndexExtension {
        create_index_called: AtomicBool,
        load_index_called: AtomicBool,
    }

    impl MockIndexExtension {
        fn new() -> Self {
            Self {
                create_index_called: AtomicBool::new(false),
                load_index_called: AtomicBool::new(false),
            }
        }
    }

    impl DeepSizeOf for MockIndexExtension {
        fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
            todo!()
        }
    }

    impl IndexExtension for MockIndexExtension {
        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        fn to_generic(self: Arc<Self>) -> Arc<dyn IndexExtension> {
            self
        }

        fn to_scalar(self: Arc<Self>) -> Option<Arc<dyn ScalarIndexExtension>> {
            None
        }

        fn to_vector(self: Arc<Self>) -> Option<Arc<dyn VectorIndexExtension>> {
            Some(self)
        }
    }

    #[async_trait::async_trait]
    impl VectorIndexExtension for MockIndexExtension {
        async fn create_index(
            &self,
            dataset: &Dataset,
            _column: &str,
            uuid: &str,
            _params: &dyn IndexParams,
        ) -> Result<()> {
            let store = dataset.object_store.clone();
            let path = dataset
                .indices_dir()
                .child(uuid.to_string())
                .child(INDEX_FILE_NAME);

            let writer = store.create(&path).await.unwrap();

            let arrow_schema = Arc::new(Schema::new(vec![VECTOR_ID_FIELD.clone()]));
            let schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap();
            let mut writer: FileWriter<ManifestDescribing> =
                FileWriter::with_object_writer(writer, schema, &FileWriterOptions::default())
                    .unwrap();
            writer.add_metadata(
                INDEX_METADATA_SCHEMA_KEY,
                json!(IndexMetadata {
                    index_type: "TEST".to_string(),
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

            self.create_index_called
                .store(true, std::sync::atomic::Ordering::Release);

            Ok(())
        }

        async fn load_index(
            &self,
            _dataset: Arc<Dataset>,
            _column: &str,
            _uuid: &str,
            _reader: FileReader,
        ) -> Result<Arc<dyn VectorIndex>> {
            self.load_index_called
                .store(true, std::sync::atomic::Ordering::Release);

            Ok(Arc::new(MockIndex))
        }
    }

    struct MockIndexParams;

    impl IndexParams for MockIndexParams {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        fn index_name(&self) -> &str {
            "TEST"
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_vector_index_extension_roundtrip(#[values(false, true)] use_legacy_format: bool) {
        // make dataset and index that is not supported natively
        let test_ds = TestVectorDataset::new(use_legacy_format).await.unwrap();
        let idx = test_ds.dataset.load_indices().await.unwrap();
        assert_eq!(idx.len(), 0);

        let idx_ext = Arc::new(MockIndexExtension::new());
        // make a new index with the extension
        let mut session = Session::default();
        session
            .register_index_extension("TEST".into(), idx_ext.clone())
            .unwrap();

        // neither has been called
        assert!(!idx_ext
            .create_index_called
            .load(std::sync::atomic::Ordering::Acquire));
        assert!(!idx_ext
            .load_index_called
            .load(std::sync::atomic::Ordering::Acquire));

        let mut ds_with_extension =
            DatasetBuilder::from_uri(test_ds.tmp_dir.path().to_str().unwrap())
                .with_session(Arc::new(session))
                .load()
                .await
                .unwrap();

        // create index
        ds_with_extension
            .create_index(&["vec"], IndexType::Vector, None, &MockIndexParams, false)
            .await
            .unwrap();

        // create index should have been called
        assert!(idx_ext
            .create_index_called
            .load(std::sync::atomic::Ordering::Acquire));
        assert!(!idx_ext
            .load_index_called
            .load(std::sync::atomic::Ordering::Acquire));

        // check that the index was created
        let ds_without_extension =
            DatasetBuilder::from_uri(test_ds.tmp_dir.path().to_str().unwrap())
                .load()
                .await
                .unwrap();
        let idx = ds_without_extension.load_indices().await.unwrap();
        assert_eq!(idx.len(), 1);
        // get the index uuid
        let index_uuid = idx.first().unwrap().uuid.to_string();

        // trying to open the index should fail as there is no extension loader
        assert!(ds_without_extension
            .open_vector_index("vec", &index_uuid)
            .await
            .unwrap_err()
            .to_string()
            .contains("Unsupported index type: TEST"));

        // trying to open the index should succeed with the extension loader
        let vector_index = ds_with_extension
            .open_vector_index("vec", &index_uuid)
            .await
            .unwrap();

        // load index should have been called
        assert!(idx_ext
            .create_index_called
            .load(std::sync::atomic::Ordering::Acquire));
        assert!(idx_ext
            .load_index_called
            .load(std::sync::atomic::Ordering::Acquire));

        // should be able to downcast to the mock index
        let _downcasted = vector_index.as_any().downcast_ref::<MockIndex>().unwrap();
    }
}
