// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cli::LanceTableToMultiBaseArgs;
use lance::Dataset;
use lance_core::Result;
use std::sync::Arc;

pub(crate) async fn to_multi_base(args: &LanceTableToMultiBaseArgs) -> Result<()> {
    let dataset = Arc::new(Dataset::open(&args.source).await?);
    let result = dataset
        .to_multi_base(args.additional_base.clone(), None)
        .await?;

    let n_bases = result.manifest.base_paths.len();
    let n_frags = result.fragments().len();

    // Print a brief summary to stdout.
    println!(
        "Converted '{}' to multi-base: {} fragments distributed across {} additional base(s).",
        args.source, n_frags, n_bases,
    );
    Ok(())
}
