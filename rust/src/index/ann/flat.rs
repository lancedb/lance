//! Flat index
//!

use std::io::Result;
use std::ops::DerefMut;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use arrow_select::concat::concat;
use async_trait::async_trait;
use object_store::path::Path as ObjectPath;

use super::{AnnIndex, AnnIndexType};
use crate::index::pb::{Footer};
use crate::index::{pb, Index};
use crate::io::object_writer::ObjectWriter;
use crate::io::{read_metadata_offset, AsyncWriteProtoExt, ObjectStore};
use crate::dataset::Dataset;
use crate::index::ann::distance::euclidean_distance;
use crate::index::ann::sort::find_topk;

/// Flat index.
///
#[derive(Debug)]
pub struct FlatIndex<'a> {
    columns: Vec<String>,
    // Index Path
    path: ObjectPath,

    // Object Store
    object_store: &'a ObjectStore,

    // The dataset this index is attached to.
    dataset: Option<Dataset>,
}

impl<'a> FlatIndex<'a> {

    /// Create an empty index.
    pub fn new(object_store: &'a ObjectStore, path: &str, columns: &[&str]) -> Result<Self> {
        Ok(Self {
            columns: columns.iter().map(|c| c.to_string()).collect(),
            path: ObjectPath::from(path),
            object_store,
            dataset: None,
        })
    }

    /// Open an index on object store.
    pub async fn open(
        object_store: &'a ObjectStore,
        path: &str,
    ) -> Result<FlatIndex<'a>> {
        let index_path = ObjectPath::from(path);
        let bytes = object_store.inner.get(&index_path).await?.bytes().await?;
        let metadata_offset = read_metadata_offset(&bytes)?;
        let mut object_reader = object_store.open(&index_path).await?;
        let footer = object_reader
            .read_message::<pb::Footer>(metadata_offset)
            .await?;
        Ok(Self {
            columns: footer.columns,
            path: index_path,
            object_store,
            dataset: None,
        })
    }

    /// Search Flat index.
    /// Returns a RecordBatch with {score:float, row_id:u64}.
    async fn search(&mut self, vector: &Float32Array, top_k: u32) -> core::result::Result<RecordBatch, ArrowError> {
        if !self.dataset.is_some() {
            return Err(ArrowError::ComputeError("Empty dataset".to_string()));
        }
        if vector.null_count() != 0 {
            return Err(ArrowError::ComputeError("Input vector must not contain nulls".to_string()));
        }
        if self.columns.len() != 1 {
            return Err(ArrowError::ComputeError(format!("1 and only 1 vector columns allowed in the index. Found {}.", self.columns.len())));
        }
        let scanner = self.dataset.as_ref().unwrap().scan()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("score", DataType::Float32, false),
        ])) as SchemaRef;

        let mut total = 0;
        let mut all_row_ids: Vec<u64> = vec![];
        let mut all_scores: Vec<f32> = vec![];

        // TODO parallelize each batch?
        for next in scanner {
            let batch = next?;
            let scores = compute_scores(&batch, self.columns[0].as_str(), vector, &total)?;
            let row_ids: Vec<u64> = (total..total+batch.num_rows() as u64).collect();
            total += batch.num_rows() as u64;
            all_row_ids.extend(&row_ids);
            all_scores.extend(&scores);
        }

        // How to handle NAs?
        let compare = |l: &f32, r: &f32| l.total_cmp(r);
        find_topk(&mut all_scores, &mut all_row_ids, top_k as usize, &compare)?;

        let top_scores: Vec<f32> = all_scores[0..top_k as usize].to_vec();
        let top_indices: Vec<u64> = all_row_ids[0..top_k as usize].to_vec();
        let columns = vec![
            Arc::new(UInt64Array::from(top_indices)) as ArrayRef,
            Arc::new(Float32Array::from(top_scores)) as ArrayRef
        ];
        RecordBatch::try_new(schema, columns)
    }
}

fn compute_scores(batch: &RecordBatch, column: &str, v: &Float32Array, total: &u64) -> core::result::Result<Vec<f32>, ArrowError> {
    let col_idx = batch.schema().index_of(column)?;
    let col = batch.column(col_idx);
    let vectors = col.as_any().downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ArrowError::SchemaError("Could not downcast to fixed size list".to_string()))?;
    euclidean_distance(v, vectors)
}

#[async_trait]
impl<'a> Index for FlatIndex<'a> {
    fn columns(&self) -> &[String] {
        self.columns.as_slice()
    }

    /// Build index
    async fn build(&self, _reader: &RecordBatch) -> Result<()> {
        let (_multipart_id, mut writer) = self.object_store.inner.put_multipart(&self.path).await?;
        let mut object_writer = ObjectWriter::new(writer.as_mut());

        let mut footer = Footer::default();
        footer.set_index_type(pb::footer::Type::Ann);
        footer.version = 1u64;
        footer.columns = self.columns.clone();

        let offset = object_writer.write_protobuf(&footer).await?;
        object_writer.write_footer(offset).await?;
        Ok(())
    }
}

#[async_trait]
impl<'a> AnnIndex for FlatIndex<'_> {
    fn ann_index_type() -> AnnIndexType {
        AnnIndexType::Flat
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
        let batch = RecordBatch::try_from_iter(vec![("vec", arr), ("pk", pk_arr)]).unwrap();

        let index_file = NamedTempFile::new().unwrap();
        let object_store = ObjectStore::new(":memory:").unwrap();
        let flat_index = FlatIndex::new(
            &object_store,
            &index_file.path().to_str().unwrap(),
            &["vec"],
        )
        .unwrap();

        flat_index.build(&batch).await.unwrap();
        assert_eq!(flat_index.columns, vec!["vec"]);
    }
}
