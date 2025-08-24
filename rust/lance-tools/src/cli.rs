// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use clap::{Parser, Subcommand};
use std::io::Error;
use std::io::ErrorKind;
use lance_core::Result as LanceResult;

#[derive(Parser, Debug)]
#[command(
    name = "lance-tools",
    about = "Tools for working with Lance files",
    version
)]
pub struct LanceToolsArgs {
    /// Subcommand to run
    #[command(subcommand)]
    pub command: LanceToolsCommand,
}

#[derive(Subcommand, Debug)]
pub enum LanceToolsCommand {
    Meta(crate::meta::MetaArgs),
}

pub async fn run(args: LanceToolsArgs) -> Result<(), std::io::Error> {
    match args.command {
        LanceToolsCommand::Meta(meta_args) => {
            return crate::meta::run(meta_args).await;
        },
    };
}

pub fn lance_result_to_std_result<T>(lance_result: LanceResult<T>) -> Result<T, std::io::Error> {
    return match lance_result {
        Ok(t) => Result::Ok(t),
        Err(e) => Result::Err(Error::new(ErrorKind::Other, e.to_string())),
    };
}