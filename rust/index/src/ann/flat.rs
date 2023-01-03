//! Flat index
//!

use std::io::Result;

use arrow_array::RecordBatchReader;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;

use crate::ann::{AnnIndex, AnnIndexType};
use crate::Index;

/// Flat index.
///
pub struct FlatIndex<'a> {
    columns: Vec<String>,
    // Index URI
    uri: ObjectPath,
    object_store: &'a dyn ObjectStore,
}

impl<'a> FlatIndex<'a> {
    /// Create an empty index.
    pub fn new(object_store: &'a dyn ObjectStore, path: &str, columns: &[&str]) -> Result<Self> {
        Ok(Self {
            columns: columns.into_iter().map(|c| c.to_string()).collect(),
            uri: ObjectPath::from(path),
            object_store,
        })
    }

    /// Open an index on a object store
    pub fn open(object_store: &'a dyn ObjectStore, path: &str) -> Result<Self> {
        Ok(Self {
            columns: vec![],
            uri: ObjectPath::from(path),
            object_store,
        })
    }
}

impl Index for FlatIndex<'_> {
    fn columns(&self) -> &[String] {
        self.columns.as_slice()
    }

    /// Build index
    fn build(&self, _reader: &impl RecordBatchReader) -> Result<()> {
        Ok(())
    }
}

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
    use arrow_array::{types::Float32Type, Array, FixedSizeListArray, RecordBatch};
    use arrow_schema::{ArrowError, SchemaRef};
    use rand::Rng;
    use tempfile::NamedTempFile;

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
            todo!()
        }
    }

    #[test]
    fn test_create_flat_index() {
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

        let batches = vec![batch];
        let reader = InMemoryBatchRecordReader::new(&batches);

        flat_index.build(&reader).unwrap()
    }
}
