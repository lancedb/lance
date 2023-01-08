use clap::{Parser, Subcommand};
use lance::dataset::Dataset;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Dataset inspection
    Inspect {
        /// The URI of the dataset.
        uri: String,

        /// AWS profile
        aws_profile: Option<String>,
    },

    /// Query the dataset
    Query {
        uri: String,

        /// The counts of record to print.
        #[arg(short, default_value_t = 100)]
        n: i64,
    },
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    match &args.command {
        Commands::Inspect { uri, aws_profile } => {
            let dataset = Dataset::open(uri).await.unwrap();
            println!("Dataset URI: {}", uri);
            println!(
                "Latest version: {}, Total versions: {}",
                dataset.version().version,
                dataset.versions().await.unwrap().len()
            );
            println!("Schema:\n{}", dataset.schema())
        }
        Commands::Query { uri, n } => {
            let dataset = Dataset::open(uri).await.unwrap();
            let mut scanner = dataset.scan().unwrap();
            scanner.limit(*n, None);
            println!("{:?}", scanner.next_batch().await.unwrap().unwrap());
        }
    }
}
