Writing and reading a dataset using Lance
---------------------------------------------------

In this example, we will write a simple lance dataset to disk. Then we will read it and print out some basic properties like the schema and sizes for each record batch in the dataset.
The example uses only one record batch, however it should work for larger datasets (multiple record batches) as well. 

Writing the raw dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

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
            .. Default::default()
        };

        Dataset::write(batches, data_path, Some(write_params)).await.unwrap();
    }

First we define a schema for our dataset, and create a record batch from that schema. Next we iterate over the record batches (only one in this case) and write them to disk. We also define the write parameters (set to overwrite) and then write the dataset to disk.

Reading a Lance dataset
~~~~~~~~~~~~~~~~~~~~~~~~
Now that we have written the dataset to a new directory, we can read it back and print out some basic properties.

.. code-block:: rust

    async fn read_dataset(data_path: &str) {

        let dataset = Dataset::open(data_path).await.unwrap();
        let scanner = dataset.scan();

        let mut batch_stream = scanner
                        .try_into_stream()
                        .await
                        .unwrap()
                        .map(|b| b.unwrap());

        while let Some(batch) = batch_stream.next().await {
            println!("Batch size: {}, {}", batch.num_rows(), batch.num_columns());  // print size of batch
            println!("Schema: {:?}", batch.schema());  // print schema of recordbatch
        }
    }

First we open the dataset, and create a scanner object. We use it to create a `batch_stream` that will let us access each record batch in the dataset.
Then we iterate over the record batches and print out the size and schema of each one.