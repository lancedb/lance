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
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // Set up tokio runtime
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // Load tokenizer from Hugging Face
        let tokenizer = load_tokenizer("gpt2")?;

        // Load dataset
        println!("Loading dataset...");
        let mut samples = Vec::new();

        // Path to the data directory
        let dataset_path = Path::new(
            "/Users/haochengliu/Documents/projects/lance/rust/examples/src/wikitext-103-raw-v1/data",
        );
        println!("Looking for data in: {}", dataset_path.display());

        // The parquet files we're looking for
        let train_files = vec![
            "train-00000-of-00002-b755d19de94348c6.parquet",
            "train-00001-of-00002-0bf6d0c487c2e75b.parquet",
        ];

        // Track if we found and processed any files
        let mut files_processed = false;

        for file in &train_files {
            let file_path = dataset_path.join(file);
            if file_path.exists() {
                files_processed = true;
                println!("Processing file: {}", file_path.display());

                // Read parquet file
                let file = File::open(file_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                println!("schema is {}", builder.schema());
                let reader = builder.build()?;

                // Process record batches instead of row groups
                for batch in reader {
                    let batch = batch?;
                    let num_rows = batch.num_rows();
                    println!("Processing batch with {} rows", num_rows);
                    
                    // Here you would extract text from the batch and tokenize it
                    // For example, if there's a "text" column:
                    if let Some(column) = batch.column_by_name("text") {
                        if let Some(string_array) = column.as_any().downcast_ref::<arrow::array::StringArray>() {
                            for i in 0..num_rows {
                                if !Array::is_null(string_array, i) {
                                    let text = string_array.value(i);
                                    // Tokenize the text using the loaded tokenizer
                                    if let Ok(encoding) = tokenizer.encode(text, true) {
                                        // Convert to i64 and add to samples
                                        let input_ids: Vec<i64> = encoding.get_ids()
                                            .iter()
                                            .map(|&id| id as i64)
                                            .collect();
                                        
                                        samples.push(input_ids);
                                    }
                                }
                            }
                        }
                    }

                    // Just collect enough samples for demonstration
                    if samples.len() >= 1000 {
                        break;
                    }
                }

                if samples.len() >= 1000 {
                    println!("Collected {} samples", samples.len());
                    break;
                }
            } else {
                println!("File not found: {}", file_path.display());
            }
        }

        // If we have no samples (files exist but processing failed), exit gracefully
        if samples.is_empty() {
            println!("No samples were processed from the parquet files.");
            println!("Please verify the file format and try again.");
            return Ok(());
        }

        // Display the first few samples
        println!("First 3 tokenized samples (showing first 10 tokens each):");
        for (i, sample) in samples.iter().take(3).enumerate() {
            let preview: Vec<_> = sample.iter().take(10).collect();
            println!("Sample {}: {:?}...", i + 1, preview);
        }

        // Shuffle the samples
        let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
        samples.shuffle(&mut rng);

        // Create schema
        let schema = Arc::new(Schema::new(vec![Field::new(
            "input_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        )]));

        // Create record batch
        let input_ids_array = create_list_array_from_vectors(&samples)?;
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(input_ids_array)])?;

        // Create a simple RecordBatchReader from our single batch
        let batch_reader = SingleBatchReader {
            schema: schema.clone(),
            batch: Some(Ok(batch)),
        };

        // Save as Lance dataset
        println!("Writing to Lance dataset...");
        let lance_dataset_path = "rust_wikitext_lance_dataset.lance";

        // Use Dataset::write with our RecordBatchReader
        let write_params = WriteParams::default();
        lance::Dataset::write(batch_reader, lance_dataset_path, Some(write_params)).await?;

        // Read and verify the dataset
        let ds = lance::Dataset::open(lance_dataset_path).await?;

        // Scan and count rows
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

// Simple RecordBatchReader implementation that yields a single batch
struct SingleBatchReader {
    schema: Arc<Schema>,
    batch: Option<Result<RecordBatch, arrow::error::ArrowError>>,
}

impl RecordBatchReader for SingleBatchReader {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

impl Iterator for SingleBatchReader {
    type Item = Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batch.take()
    }
}

fn load_tokenizer(model_name: &str) -> Result<Tokenizer> {
    // Download tokenizer from Hugging Face
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
    // Create list array using the builder approach
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
