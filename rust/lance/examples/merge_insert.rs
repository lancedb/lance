use lance::dataset::MergeInsertBuilder;
// use futures::StreamExt;
use arrow::array::{StringArray, UInt32Array};
use arrow::datatypes::Schema;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::RecordBatchIterator;
use lance::dataset::WhenMatched;
use lance::dataset::WhenNotMatched;
use lance::dataset::WhenNotMatchedBySource;
use lance::{dataset::WriteParams, Dataset};
use std::sync::Arc;

use futures::StreamExt;
use lance::dataset::WriteMode;
use lance_datafusion::utils::reader_to_stream;

#[tokio::main]
async fn main() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
        Field::new("filterme", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(UInt32Array::from(vec![1, 2, 1, 1, 1, 1])),
            Arc::new(StringArray::from(vec!["A", "B", "A", "A", "B", "A"])),
        ],
    )
    .unwrap();

    // create lance dataset from recordbatch
    let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());

    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        use_experimental_writer: true,
        ..Default::default()
    };

    let ds = Arc::new(
        Dataset::write(batches, "./examples/temp_data.lance", Some(write_params))
            .await
            .unwrap(),
    );

    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![4, 5, 6, 7, 8, 9, 10])),
            Arc::new(UInt32Array::from(vec![2, 2, 2, 2, 2, 2, 3])),
            Arc::new(StringArray::from(vec!["A", "B", "C", "A", "B", "C", "D"])),
        ],
    )
    .unwrap();

    let keys = vec!["key".to_string()];
    // find-or-create, no delete
    // let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
    //     .unwrap()
    //     .when_matched(WhenMatched::UpdateAll)
    //     .try_build()
    //     .unwrap();

    // update only, with delete all
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll) // src has no match in target (new rows)
        .when_not_matched_by_source(
            WhenNotMatchedBySource::delete_if(&ds, "source.value == target.value").unwrap(),
        ) //
        .try_build()
        .unwrap();
    // should update keys 4,5,6. should insert rows 7,8,9,10

    let schema = new_batch.schema();
    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);
    let (merged_dataset, inserted, updated, deleted) = job.execute(new_stream).await.unwrap();
    // println!("Merged dataset: {:?}", merged_dataset);

    let scanner = merged_dataset.scan();
    let batches: Vec<RecordBatch> = scanner
        .try_into_stream()
        .await
        .unwrap()
        .map(|b| b.unwrap())
        .collect::<Vec<RecordBatch>>()
        .await;

    println!("Batches: {:?}", batches);
    println!("Inserted: {:?}", inserted);
    println!("Updated: {:?}", updated);
    println!("Deleted: {:?}", deleted);
}
