use std::sync::Arc;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::array::UInt32Array;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use lance::dataset::{WriteParams, WriteMode};
use lance::Dataset;
use futures::StreamExt;

// Writes sample dataset to the given path
async fn write_dataset(data_path: &str) {

    // Define new schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]));

    // Create new record batch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(UInt32Array::from(vec![6, 7, 8, 9, 10, 11])),
            ],
    )
    .unwrap();
    
    let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());

    // Define write parameters (e.g. overwrite dataset)
    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        use_experimental_writer: true, // use lance v2 writer
        .. Default::default()
    };

    Dataset::write(batches, data_path, Some(write_params)).await.unwrap();

}

// Reads dataset from the given path and prints batch size, schema for all record batches. Also extracts and prints a slice from the first batch
async fn read_dataset(data_path: &str) {

    let dataset = Dataset::open(data_path).await.unwrap();
    let scanner = dataset.scan();

    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await
        .unwrap()
        .map(|b| b.unwrap())
        .collect::<Vec<RecordBatch>>()
        .await;

    // Iterate over record batches
    for (batch_index, batch) in batches.iter().enumerate() {
        println!("RecordBatch #{}:", batch_index);
        println!("Batch size: {}, {}", batch.num_rows(), batch.num_columns());  // print size of batch
        println!("Schema: {:?}", batch.schema());         // print schema of recordbatch
    }

    // extract a slice (rows 3-4) from the first batch
    let target_batch = &batches[0];
    let slice_batch = target_batch.slice(3, 1);    
    println!("Sliced batch: {:?}", slice_batch)
}

#[tokio::main]
async fn main() {
    let data_path: &str = "./temp_data.lance";

    write_dataset(data_path).await;
    read_dataset(data_path).await;
}