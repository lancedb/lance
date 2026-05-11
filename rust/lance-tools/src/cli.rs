// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::{Args, Parser, Subcommand};
use lance_core::Result;

fn parse_remove_data_file(value: &str) -> std::result::Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err("remove-data-files entries must not be empty".to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "lance-tools",
    about = "Tools for interacting with Lance files and tables",
    version
)]
pub struct LanceToolsArgs {
    /// Subcommand to run
    #[command(subcommand)]
    command: LanceToolsCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceToolsCommand {
    /// Commands for interacting with Lance files.
    File(LanceFileArgs),
    /// Commands for inspecting and repairing Lance datasets.
    Dataset(LanceDatasetArgs),
}

#[derive(Parser, Debug)]
pub struct LanceFileArgs {
    #[command(subcommand)]
    command: LanceFileCommand,
}

#[derive(Parser, Debug)]
pub struct LanceDatasetArgs {
    #[command(subcommand)]
    command: LanceDatasetCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceFileCommand {
    /// Display Lance file metadata.
    Meta(LanceFileMetaArgs),
}

#[derive(Subcommand, Debug)]
pub enum LanceDatasetCommand {
    /// Locate the first manifest in the latest run that references a data file
    LocateDataFile(LanceDatasetLocateDataFileArgs),
    /// Verify manifest-referenced data files for missing or unreadable objects
    VerifyDataFiles(LanceDatasetVerifyDataFilesArgs),
    /// Repair a dataset manifest with explicit repair actions
    RepairManifest(LanceDatasetRepairManifestArgs),
    /// Restore an old dataset version as the latest version
    Restore(LanceDatasetRestoreArgs),
}

#[derive(Args, Debug)]
pub struct LanceFileMetaArgs {
    // The Lance file to examine.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,
}

#[derive(Args, Debug)]
pub struct LanceDatasetLocateDataFileArgs {
    /// The Lance dataset to examine.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,

    /// JSON object of storage options to pass to the object store.
    #[arg(long, alias = "storage-options", value_name = "json")]
    pub(crate) storage_options: Option<String>,

    /// Data file path or suffix to locate. Accepts full object keys, data/foo.lance, or foo.lance.
    #[arg(short = 'f', long, value_name = "data-file")]
    pub(crate) data_file: String,

    /// Start searching at this dataset version. Defaults to latest.
    #[arg(long)]
    pub(crate) version: Option<u64>,

    /// Restore the version immediately before the located manifest.
    #[arg(long)]
    pub(crate) restore: bool,
}

#[derive(Args, Debug)]
pub struct LanceDatasetVerifyDataFilesArgs {
    /// The Lance dataset to examine.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,

    /// JSON object of storage options to pass to the object store.
    #[arg(long, alias = "storage-options", value_name = "json")]
    pub(crate) storage_options: Option<String>,

    /// Dataset version to check. Defaults to latest.
    #[arg(long)]
    pub(crate) version: Option<u64>,

    /// Read Lance file metadata to detect corrupt data files, not only missing objects.
    #[arg(long)]
    pub(crate) deep: bool,

    /// Return an error when any missing or corrupt data file is found.
    #[arg(long)]
    pub(crate) fail: bool,
}

#[derive(Args, Debug)]
pub struct LanceDatasetRepairManifestArgs {
    /// The Lance dataset manifest to repair.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,

    /// JSON object of storage options to pass to the object store.
    #[arg(long, alias = "storage-options", value_name = "json")]
    pub(crate) storage_options: Option<String>,

    /// Remove the entire fragment (all its data files) if its file list includes this path or suffix. Can be comma-separated or specified multiple times.
    #[arg(
        long = "remove-data-files",
        value_name = "data-file",
        required = true,
        value_delimiter = ',',
        value_parser = parse_remove_data_file
    )]
    pub(crate) remove_data_files: Vec<String>,

    /// Show the affected fragments without committing a new version.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

#[derive(Args, Debug)]
pub struct LanceDatasetRestoreArgs {
    /// The Lance dataset to restore.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,

    /// JSON object of storage options to pass to the object store.
    #[arg(long, alias = "storage-options", value_name = "json")]
    pub(crate) storage_options: Option<String>,

    /// Version to restore as the latest version.
    #[arg(long)]
    pub(crate) version: u64,
}

impl LanceToolsArgs {
    pub async fn run(&self, writer: impl std::io::Write) -> Result<()> {
        match &self.command {
            LanceToolsCommand::File(args) => match &args.command {
                LanceFileCommand::Meta(args) => crate::meta::show_file_meta(writer, args).await,
            },
            LanceToolsCommand::Dataset(args) => match &args.command {
                LanceDatasetCommand::LocateDataFile(args) => {
                    crate::dataset::locate_data_file(writer, args).await
                }
                LanceDatasetCommand::VerifyDataFiles(args) => {
                    crate::dataset::verify_data_files(writer, args).await
                }
                LanceDatasetCommand::RepairManifest(args) => {
                    crate::dataset::repair_manifest(writer, args).await
                }
                LanceDatasetCommand::Restore(args) => crate::dataset::restore(writer, args).await,
            },
        }
    }
}
