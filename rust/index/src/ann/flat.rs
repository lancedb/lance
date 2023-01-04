//! Flat index
//!

use std::io::Result;

use arrow_array::RecordBatch;
use async_trait::async_trait;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;

use crate::ann::{AnnIndex, AnnIndexType};
use crate::Index;
use crate::pb;
use crate::pb::{Footer, AnnIndex as PbAnnIndex};

/// Flat index.
///
#[derive(Debug)]
pub struct FlatIndex<'a> {
    columns: Vec<String>,
    // Index URI
    path: ObjectPath,
    object_store: &'a dyn ObjectStore,
}

impl<'a> FlatIndex<'a> {
    /// Create an empty index.
    pub fn new(object_store: &'a dyn ObjectStore, path: &str, columns: &[&str]) -> Result<Self> {
        Ok(Self {
            columns: columns.into_iter().map(|c| c.to_string()).collect(),
            path: ObjectPath::from(path),
            object_store,
        })
    }

    /// Open an index on a object store
    pub async fn open(object_store: &'a dyn ObjectStore, path: &str) -> Result<FlatIndex<'a>> {
        Ok(Self {
            columns: vec![],
            path: ObjectPath::from(path).clone(),
            object_store,
        })
    }
}

#[async_trait]
impl Index for FlatIndex<'_> {
    fn columns(&self) -> &[String] {
        self.columns.as_slice()
    }

    /// Build index
    async fn build(&self, _reader: &RecordBatch) -> Result<()> {
        let (multipart_id, writer) = self.object_store.put_multipart(&self.path).await?;

        let mut footer = Footer::default();
        footer.set_index_type(pb::footer::Type::Ann);
        footer.version = 1u64;
        footer.columns = self.columns.clone();

        Ok(())
    }
}

#[async_trait]
impl AnnIndex for FlatIndex<'_> {
    fn ann_index_type() -> AnnIndexType {
        todo!()
    }

    fn dim(&self) -> usize {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter::{repeat, repeat_with};
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::{
        types::Float32Type, Array, FixedSizeListArray, RecordBatch, RecordBatchReader,
    };
    use arrow_schema::{ArrowError, SchemaRef};
    use rand::Rng;
    use tempfile::NamedTempFile;

    /// Create a [rows, dim] of f32 matrix as `FixedSizeListArray[f32, dim]`.
    fn generate_vector_array(dim: i32, rows: i32) -> Arc<dyn Array> {
        let mut rng = rand::thread_rng();
        Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                repeat(0).take(rows as usize).map(|_| {
                    Some(
                        repeat_with(|| Some(rng.gen::<f32>()))
                            .take(dim as usize)
                            .collect::<Vec<Option<f32>>>(),
                    )
                }),
                dim,
            ),
        )
    }

    // TODO: move
    struct InMemoryBatchRecordReader<'a> {
        idx: usize,
        batches: &'a [RecordBatch],
    }

    impl<'a> InMemoryBatchRecordReader<'a> {
        fn new(batches: &'a [RecordBatch]) -> Self {
            Self { idx: 0, batches }
        }
    }

    impl RecordBatchReader for InMemoryBatchRecordReader<'_> {
        fn schema(&self) -> SchemaRef {
            assert!(!self.batches.is_empty());
            self.batches[0].schema()
        }
    }

    impl Iterator for InMemoryBatchRecordReader<'_> {
        type Item = std::result::Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.idx >= self.batches.len() {
                None
            } else {
                self.idx += 1;
                Some(Ok(self.batches[self.idx - 1].clone()))
            }
        }
    }

    #[tokio::test]
    async fn test_create_flat_index() {
        let arr = generate_vector_array(128, 1024);
        let pk_arr = Arc::new(Int64Array::from((1..1025).collect::<Vec<i64>>()));
        println!("vec arr={}, pk={}", arr.len(), pk_arr.len());
        let batch = RecordBatch::try_from_iter(vec![("vec", arr), ("pk", pk_arr)]).unwrap();

        let index_file = NamedTempFile::new().unwrap();
        let object_store = object_store::local::LocalFileSystem::new();
        let flat_index = FlatIndex::new(
            &object_store,
            &index_file.path().to_str().unwrap(),
            &["vec"],
        )
        .unwrap();

        flat_index.build(&batch).await.unwrap()
    }
}
