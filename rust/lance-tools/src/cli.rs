// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::{Args, Parser, Subcommand};
use lance_core::Result;

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
    /// Commands for interacting with Lance tables.
    Table(LanceTableArgs),
}

#[derive(Parser, Debug)]
pub struct LanceFileArgs {
    #[command(subcommand)]
    command: LanceFileCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceFileCommand {
    /// Display Lance file metadata.
    Meta(LanceFileMetaArgs),
}

#[derive(Args, Debug)]
pub struct LanceFileMetaArgs {
    // The Lance file to examine.
    #[arg(short = 's', long, value_name = "source")]
    pub(crate) source: String,
}

#[derive(Parser, Debug)]
pub struct LanceTableArgs {
    #[command(subcommand)]
    pub(crate) command: LanceTableCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceTableCommand {
    /// Convert a single-base Lance table into a multi-base table.
    ///
    /// The caller must have already copied the full dataset directory to each
    /// additional base URI (e.g. with `azcopy` or `gsutil rsync`) before
    /// running this command. Only metadata is updated; no data is moved.
    ///
    /// Example:
    ///   lance-tools table to-multi-base \
    ///       --source az://container1/mydata \
    ///       --additional-base az://container2/mydata \
    ///       --additional-base az://container3/mydata
    ToMultiBase(LanceTableToMultiBaseArgs),
}

#[derive(Args, Debug)]
pub struct LanceTableToMultiBaseArgs {
    /// URI of the existing (source) Lance dataset.
    #[arg(short = 's', long, value_name = "source")]
    pub source: String,

    /// URI of an additional copy of the dataset. Specify once per copy.
    #[arg(long = "additional-base", value_name = "URI")]
    pub additional_base: Vec<String>,
}

impl LanceToolsArgs {
    pub async fn run(&self, writer: impl std::io::Write) -> Result<()> {
        match &self.command {
            LanceToolsCommand::File(args) => match &args.command {
                LanceFileCommand::Meta(args) => crate::meta::show_file_meta(writer, args).await,
            },
            LanceToolsCommand::Table(args) => match &args.command {
                LanceTableCommand::ToMultiBase(args) => crate::table::to_multi_base(args).await,
            },
        }
    }
}
