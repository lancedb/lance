// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_array::RecordBatch;
use clap::{Parser, Subcommand, ValueEnum};
use futures::stream::{Stream, StreamExt};
use futures::TryStreamExt;

use lance::dataset::Dataset;
use lance::Result;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
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

    /// Manipulate indices
    Index {
        /// Actions on index
        #[arg(value_enum)]
        action: IndexAction,

        // Dataset URI.
        uri: String,

        /// The column to build index on.
        #[arg(short, long)]
        column: Option<String>,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum IndexAction {
    Create,
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
            let mut scanner = dataset.scan();
            scanner.limit(*n, None);
            let stream = scanner.into_stream();
            let batch: Vec<RecordBatch> = stream.take(1).try_collect::<Vec<_>>().await.unwrap();
            println!("{:?}", batch);
        }
        Commands::Index {
            action,
            uri,
            column,
        } => {}
    }
}
