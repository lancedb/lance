# 使用 Lance 读写数据集

在本示例中，我们将向磁盘写入一个简单的 Lance 数据集。然后读取它并打印一些基本属性，如 Schema 和数据集中每个 Record Batch 的大小。
本示例只使用了一个 Record Batch，但它也适用于更大的数据集（多个 Record Batch）。

## 写入原始数据集

```rust
// Writes sample dataset to the given path
async fn write_dataset(data_path: &str) {
    // Define new schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]));

    // Create new record batches
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
        ..Default::default()
    };

    Dataset::write(batches, data_path, Some(write_params))
        .await
        .unwrap();
} // End write dataset
```

首先我们为数据集定义一个 Schema，并根据该 Schema 创建一个 Record Batch。接下来遍历 Record Batch（本例中只有一个）并将其写入磁盘。我们还定义了写入参数（设置为覆盖模式），然后将数据集写入磁盘。

## 读取 Lance 数据集

现在我们已经将数据集写入了新目录，可以读取它并打印一些基本属性。

```rust
// Reads dataset from the given path and prints batch size, schema for all record batches. Also extracts and prints a slice from the first batch
async fn read_dataset(data_path: &str) {
    let dataset = Dataset::open(data_path).await.unwrap();
    let scanner = dataset.scan();

    let mut batch_stream = scanner.try_into_stream().await.unwrap().map(|b| b.unwrap());

    while let Some(batch) = batch_stream.next().await {
        println!("Batch size: {}, {}", batch.num_rows(), batch.num_columns()); // print size of batch
        println!("Schema: {:?}", batch.schema()); // print schema of recordbatch

        println!("Batch: {:?}", batch); // print the entire recordbatch (schema and data)
    }
} // End read dataset
```

首先打开数据集并创建一个 scanner 对象。我们使用它创建一个 `batch_stream`，让我们可以访问数据集中的每个 Record Batch。
然后遍历 Record Batch 并打印每个 Batch 的大小和 Schema。

## 完整示例

```rust
use arrow::array::UInt32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::StreamExt;
use lance::dataset::{WriteMode, WriteParams};
use lance::Dataset;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let data_path: &str = "./temp_data.lance";

    write_dataset(data_path).await;
    read_dataset(data_path).await;
}
```
