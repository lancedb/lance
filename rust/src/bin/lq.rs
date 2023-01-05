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
    }
}
