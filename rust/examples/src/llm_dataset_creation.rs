use anyhow::Result;
use arrow::array::{Array, Int64Array, ListArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::RecordBatchReader;
use futures::StreamExt;
use hf_hub::{api::sync::Api, Repo, RepoType};
use lance::dataset::WriteParams;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::FileReader;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Debug)]
struct SimpleError(String);

impl std::fmt::Display for SimpleError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SimpleError {}

struct WikiTextBatchReader {
    schema: Arc<Schema>,
    parquet_readers: Vec<Option<ParquetRecordBatchReaderBuilder<File>>>,
    current_reader_idx: usize,
    current_reader: Option<Box<dyn RecordBatchReader + Send>>,
    tokenizer: Tokenizer,
}

// Implement Send for WikiTextBatchReader
unsafe impl Send for WikiTextBatchReader {}

impl WikiTextBatchReader {
    fn new(
        parquet_readers: Vec<ParquetRecordBatchReaderBuilder<File>>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "input_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        )]));

        Ok(Self {
            schema,
            parquet_readers: parquet_readers.into_iter().map(Some).collect(),
            current_reader_idx: 0,
            current_reader: None,
            tokenizer,
        })
    }

    fn process_batch(
        &self,
        input_batch: &RecordBatch,
    ) -> Result<RecordBatch, arrow::error::ArrowError> {
        let num_rows = input_batch.num_rows();
        let mut tokenized_vectors = Vec::with_capacity(num_rows);

        if let Some(column) = input_batch.column_by_name("text") {
            if let Some(string_array) = column.as_any().downcast_ref::<arrow::array::StringArray>()
            {
                for i in 0..num_rows {
                    if !Array::is_null(string_array, i) {
                        let text = string_array.value(i);
                        if let Ok(encoding) = self.tokenizer.encode(text, true) {
                            let input_ids: Vec<i64> =
                                encoding.get_ids().iter().map(|&id| id as i64).collect();
                            tokenized_vectors.push(input_ids);
                        }
                    }
                }
            }
        }

        let input_ids_array = create_list_array_from_vectors(&tokenized_vectors).map_err(|e| {
            arrow::error::ArrowError::ExternalError(Box::new(SimpleError(e.to_string())))
        })?;
        RecordBatch::try_new(self.schema.clone(), vec![Arc::new(input_ids_array)])
    }
}

impl RecordBatchReader for WikiTextBatchReader {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

impl Iterator for WikiTextBatchReader {
    type Item = Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a current reader, try to get next batch
            if let Some(reader) = &mut self.current_reader {
                if let Some(batch_result) = reader.next() {
                    return Some(batch_result.and_then(|batch| self.process_batch(&batch)));
                }
            }

            // If no current reader or current reader is exhausted, try to get next reader
            if self.current_reader_idx < self.parquet_readers.len() {
                if let Some(builder) = self.parquet_readers[self.current_reader_idx].take() {
                    match builder.build() {
                        Ok(reader) => {
                            self.current_reader = Some(Box::new(reader));
                            self.current_reader_idx += 1;
                            continue;
                        }
                        Err(e) => {
                            return Some(Err(arrow::error::ArrowError::ExternalError(Box::new(e))))
                        }
                    }
                }
            }

            // No more readers available
            return None;
        }
    }
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // Load tokenizer
        let tokenizer = load_tokenizer("gpt2")?;

        // Set up dataset path
        let dataset_path = Path::new(
            "/Users/haochengliu/Documents/projects/lance/rust/examples/src/wikitext-103-raw-v1/data",
        );
        println!("Looking for data in: {}", dataset_path.display());

        // Collect parquet readers
        let train_files = vec![
            "train-00000-of-00002-b755d19de94348c6.parquet",
            "train-00001-of-00002-0bf6d0c487c2e75b.parquet",
        ];

        let mut parquet_readers = Vec::new();
        for file in &train_files {
            let file_path = dataset_path.join(file);
            if file_path.exists() {
                println!("Processing file: {}", file_path.display());
                let file = File::open(file_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                parquet_readers.push(builder);
            }
        }

        if parquet_readers.is_empty() {
            println!("No parquet files found to process.");
            return Ok(());
        }

        // Create batch reader
        let batch_reader = WikiTextBatchReader::new(parquet_readers, tokenizer)?;

        // Save as Lance dataset
        println!("Writing to Lance dataset...");
        let lance_dataset_path = "rust_wikitext_lance_dataset.lance";

        let write_params = WriteParams::default();
        lance::Dataset::write(batch_reader, lance_dataset_path, Some(write_params)).await?;

        // Verify the dataset
        let ds = lance::Dataset::open(lance_dataset_path).await?;
        let scanner = ds.scan();
        let mut stream = scanner.try_into_stream().await?;

        let mut total_rows = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            total_rows += batch.num_rows();
        }

        println!(
            "Lance dataset created successfully with {} rows",
            total_rows
        );
        println!("Dataset location: {}", lance_dataset_path);

        Ok(())
    })
}

fn load_tokenizer(model_name: &str) -> Result<Tokenizer> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_name.into(),
        RepoType::Model,
        "main".into(),
    ));

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(tokenizer)
}

fn create_list_array_from_vectors(vectors: &[Vec<i64>]) -> Result<ListArray> {
    let mut builder = arrow::array::ListBuilder::new(arrow::array::Int64Builder::new());

    for vector in vectors {
        let values_builder = builder.values();
        for &value in vector {
            values_builder.append_value(value);
        }
        builder.append(true);
    }

    let list_array = builder.finish();
    Ok(list_array)
}
