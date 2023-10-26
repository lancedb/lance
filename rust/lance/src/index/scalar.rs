// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Utilities connecting scalar indices to the lance format

use std::{path::PathBuf, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;

use lance_core::{io::ReadBatchParams, Error, Result};
use lance_index::scalar::{IndexReader, IndexStore, IndexWriter};

use crate::io::{FileReader, FileWriter, FileWriterOptions, ObjectStore};

#[derive(Debug)]
pub struct LanceIndexStore {
    object_store: ObjectStore,
    index_dir: PathBuf,
}

impl LanceIndexStore {
    pub fn new(object_store: ObjectStore, index_dir: PathBuf) -> Self {
        Self {
            object_store,
            index_dir,
        }
    }
}

#[async_trait]
impl IndexWriter for FileWriter {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64> {
        let offset = self.tell().await?;
        self.write(&[batch]).await?;
        Ok(offset as u64)
    }

    async fn finish(&mut self) -> Result<()> {
        FileWriter::finish(self).await.map(|_| ())
    }
}

#[async_trait]
impl IndexReader for FileReader {
    async fn read_record_batch(&self, offset: u64) -> Result<RecordBatch> {
        self.read_batch(offset as i32, ReadBatchParams::RangeFull, self.schema())
            .await
    }
}

#[async_trait]
impl IndexStore for LanceIndexStore {
    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<Schema>,
    ) -> Result<Box<dyn IndexWriter>> {
        let path = self.index_dir.join(name);
        let path = path.as_os_str().to_str().ok_or_else(|| Error::Internal {
            message: format!("Could not parse path {path:?}"),
        })?;
        let path = object_store::path::Path::parse(path)?;
        let schema = schema.as_ref().try_into()?;
        let writer = FileWriter::try_new(
            &self.object_store,
            &path,
            schema,
            &FileWriterOptions::default(),
        )
        .await?;
        Ok(Box::new(writer))
    }

    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        let path = self.index_dir.join(name);
        let path = path.as_os_str().to_str().ok_or_else(|| Error::Internal {
            message: format!("Could not parse {path:?}"),
        })?;
        let path = object_store::path::Path::parse(path)?;
        let file_reader = FileReader::try_new(&self.object_store, &path).await?;
        Ok(Arc::new(file_reader))
    }
}

#[cfg(test)]
mod tests {

    use std::path::Path;

    use crate::io::object_store::ObjectStoreParams;

    use super::*;
    use arrow_array::{
        types::{UInt32Type, UInt64Type},
        RecordBatchReader,
    };
    use datafusion::scalar::ScalarValue;
    use futures::stream;
    use lance_core::Error;
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_index::scalar::{
        btree::{train_btree_index, BTreeIndex},
        flat::FlatIndexTrainer,
        ScalarIndex, ScalarQuery,
    };
    use tempfile::{tempdir, TempDir};

    fn test_data() -> impl RecordBatchReader {
        gen()
            .col(Some("values".to_string()), array::step::<UInt32Type>())
            .col(Some("row_ids".to_string()), array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(4096), BatchCount::from(100))
    }

    fn test_store(tempdir: &TempDir) -> Arc<dyn IndexStore> {
        let test_path: &Path = tempdir.path();
        let (object_store, _) = ObjectStore::from_path(
            test_path.as_os_str().to_str().unwrap(),
            &ObjectStoreParams::default(),
        )
        .unwrap();
        Arc::new(LanceIndexStore::new(object_store, test_path.to_owned()))
    }

    async fn train_index(index_store: &Arc<dyn IndexStore>) {
        let sub_index_trainer = FlatIndexTrainer::new(arrow_schema::DataType::UInt32);

        let data = stream::iter(test_data().map(|batch| batch.map_err(|err| Error::from(err))));
        train_btree_index(data, &sub_index_trainer, index_store.as_ref())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_btree() {
        let tempdir = tempdir().unwrap();
        let index_store = test_store(&tempdir);
        train_index(&index_store).await;
        let index = BTreeIndex::load(index_store).await.unwrap();

        let row_ids = index
            .search(&ScalarQuery::Equals(ScalarValue::UInt32(Some(10000))))
            .await
            .unwrap();

        dbg!(row_ids);
    }
}
