use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use lance::{dataset::WriteParams, Dataset};
use std::sync::Arc;

#[tokio::test]
async fn create_dataset() {
    let ids_raw: Vec<Option<u32>> = vec![None, None, None, None];
    let ids = UInt32Array::from(
        // vec![0,0,0,0,0,0,0,0]
        vec![None, None, None],
        // ids_raw,
    );
    // let ids = UInt32Array::from_iter_values(0..100_000);
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, true)]));
    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids) as ArrayRef]);
    let reader = RecordBatchIterator::new(vec![record_batch], schema.clone());
    let write_params = WriteParams {
        use_legacy_format: false,
        ..Default::default()
    };

    Dataset::write(
        reader,
        "~/Desktop/lance_datasets/test1.lance",
        Some(write_params),
    )
    .await
    .unwrap();
}
