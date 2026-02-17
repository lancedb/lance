// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::prelude::SessionContext;

pub mod bbox;

pub fn register_functions(ctx: &SessionContext) {
    geodatafusion::register(ctx);
}
