// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::RT;
use lance_tools::util::run_cli;
use pyo3::prelude::*;

#[pyfunction(name = "_lance_tools_cli")]
pub fn lance_tools_cli(args: Vec<String>) -> PyResult<()> {
    let _ = RT.block_on(None, run_cli(args));

    Ok(())
}
