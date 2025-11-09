// rust/lance/src/dataset/blob_stream.rs

use arrow_array::{
    ArrayRef, RecordBatch, StructArray, UInt64Array,
};
use arrow_array::builder::LargeBinaryBuilder;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_physical_plan::{RecordBatchStream, SendableRecordBatchStream};
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::collections::HashSet;
use datafusion::common::HashMap;

use crate::dataset::Dataset;
use lance_core::utils::address::RowAddress;

pub fn wrap_blob_stream_if_needed(
    inner: SendableRecordBatchStream,
    dataset: Arc<Dataset>,
) -> SendableRecordBatchStream {
    let blob_fields: Vec<&lance_core::datatypes::Field> = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| field.is_blob())
        .collect();

    if blob_fields.is_empty() {
        // No blob fields → return original stream
        inner
    } else {
        Box::pin(ResolvedBlobStream::new(inner, dataset))
    }
}

/// Resolve blob column by reading from disk based on position/size metadata.
pub fn resolve_blob_column(
    batch: &RecordBatch,
    dataset: &Dataset,
    blob_field_name: &str,
    blob_field_id: u32,
    row_id_col: &UInt64Array,
) -> DFResult<ArrayRef> {
    let blob_struct = batch
        .column_by_name(blob_field_name)
        .ok_or_else(|| DataFusionError::Execution("blob column missing".to_string()))?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| DataFusionError::Execution("blob is not a struct".to_string()))?;

    let position_array = blob_struct
        .column_by_name("position")
        .ok_or_else(|| DataFusionError::Execution("position field missing".to_string()))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Execution("position is not UInt64".to_string()))?;

    let size_array = blob_struct
        .column_by_name("size")
        .ok_or_else(|| DataFusionError::Execution("size field missing".to_string()))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Execution("size is not UInt64".to_string()))?;

    let mut blobs = Vec::with_capacity(batch.num_rows());

    for i in 0..batch.num_rows() {
        let row_id = row_id_col.value(i);
        let position = position_array.value(i);
        let size = size_array.value(i) as usize;

        let frag_id = RowAddress::from(row_id).fragment_id();
        let frag = dataset
            .get_fragment(frag_id as usize)
            .ok_or_else(|| DataFusionError::Execution("fragment not found".to_string()))?;

        let data_file = frag
            .data_file_for_field(blob_field_id)
            .ok_or_else(|| DataFusionError::Execution("blob data file not found".to_string()))?;

        let path_str = dataset.data_dir().child(data_file.path.as_str()).to_string();
        let local_path = PathBuf::from("/".to_owned() + &path_str);

        let mut file = File::open(&local_path)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        file.seek(SeekFrom::Start(position))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut buffer = vec![0; size];
        file.read_exact(&mut buffer)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        blobs.push(buffer);
    }

    let mut builder = LargeBinaryBuilder::with_capacity(blobs.len(), blobs.iter().map(Vec::len).sum());
    for blob in blobs {
        builder.append_value(blob);
    }
    let binary_array = builder.finish();

    Ok(Arc::new(binary_array))
}

/// A stream that resolves blob columns from on-disk storage.
pub struct ResolvedBlobStream {
    inner: SendableRecordBatchStream,
    dataset: Arc<Dataset>,
    output_schema: SchemaRef,
    blob_field_name_to_id: Vec<(String, u32)>,
}

impl ResolvedBlobStream {
    /// Create a new [`ResolvedBlobStream`] that automatically resolves the **first** blob column
    /// found in the dataset schema.
    ///
    /// # Panics
    /// - If no blob field is found in the dataset schema
    /// - If more than one blob field is found (currently unsupported)
    /// - If the input stream does not contain the blob field
    pub fn new(
        inner: SendableRecordBatchStream,
        dataset: Arc<Dataset>,
    ) -> Self {
        // 🔍 自动查找所有 blob 字段
        let blob_field_name_to_id: Vec<(String, u32)> = dataset
            .schema()
            .fields
            .iter()
            .filter(|field| field.is_blob())
            .map(|f| (f.name.clone(), f.id as u32))
            .collect();

        let input_schema = inner.schema();
        let blob_names: HashSet<&String> = blob_field_name_to_id.iter().map(|(n, _)| n).collect();

        for (name, _) in &blob_field_name_to_id {
            if input_schema.column_with_name(name).is_none() {
                panic!("Input schema missing blob field: {}", name);
            }
        }

        let fields: Vec<Field> = input_schema
            .fields()
            .iter()
            .map(|f| {
                if blob_names.contains(f.name()) {
                    Field::new(f.name(), DataType::LargeBinary, f.is_nullable())
                } else {
                    f.as_ref().clone()
                }
            })
            .collect();
        let output_schema = Arc::new(Schema::new(fields));

        Self {
            inner,
            dataset,
            output_schema,
            blob_field_name_to_id,
        }
    }
}

impl Stream for ResolvedBlobStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let row_ids = batch
                    .column_by_name("_rowid")
                    .ok_or_else(|| DataFusionError::Execution("must have _rowid".to_string()))?
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| DataFusionError::Execution("_rowid is not UInt64".to_string()))?;

                let mut resolved_blobs: HashMap<String, ArrayRef> = HashMap::new();
                for (name, field_id) in &self.blob_field_name_to_id {
                    let resolved = resolve_blob_column(
                        &batch,
                        &self.dataset,
                        name,
                        *field_id,
                        row_ids,
                    )?;
                    resolved_blobs.insert(name.clone(), resolved);
                }

                let mut new_columns = Vec::with_capacity(batch.num_columns());
                for (i, col) in batch.columns().iter().enumerate() {
                    let field_name = batch.schema_ref().field(i).name();
                    if let Some(resolved) = resolved_blobs.get(field_name) {
                        new_columns.push(resolved.clone());
                    } else {
                        new_columns.push(col.clone());
                    }
                }

                let new_batch = RecordBatch::try_new(self.output_schema.clone(), new_columns)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

                Poll::Ready(Some(Ok(new_batch)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(DataFusionError::Execution(
                format!("inner stream error: {:?}", e),
            )))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for ResolvedBlobStream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}
