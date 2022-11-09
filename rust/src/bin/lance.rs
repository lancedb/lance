//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

use std::fs::{read, File};

use clap::{Parser, Subcommand};

use lance::io::FileReader;

#[derive(Parser)]
#[clap(version, about = "Lance CLI", long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect Lance files
    Inspect {
        /// Path to the lance dataset.
        #[clap(value_parser)]
        path: std::path::PathBuf,
    },
    Show {
        /// Path to the lance dataset.
        #[clap(value_parser)]
        path: std::path::PathBuf,
    },
}

fn main() {
    let args = Args::parse();

    match &args.command {
        Commands::Inspect { path } => {
            let f = File::open(path).unwrap();
            let reader = FileReader::new(f).unwrap();
            println!("Number of RecordBatch: {}", reader.num_chunks());
            println!("Schema: {}\n", reader.schema());
            use std::any::TypeId;
            let is_little_endian =
                TypeId::of::<byteorder::NativeEndian>() == TypeId::of::<byteorder::LittleEndian>();
            println!("Is little endian {:?}", is_little_endian)
        }
        Commands::Show { path } => {
            let f = File::open(path).unwrap();
            let mut reader = FileReader::new(f).unwrap();
            reader.get(0).iter().enumerate().for_each(|(idx, x)| {
                println!("field: {:?}", reader.schema().fields[idx].name);
                println!("example value: {:?}", x)
            });
        }
    }
}
