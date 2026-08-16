// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: lowering the take node to a read keyed by the rows the input found.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use lance_core::ROW_ID;

use super::super::LanceTakeNode;
use super::super::source::v1;
use crate::Result;
use crate::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};

pub fn plan_take(
    node: &LanceTakeNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Both forms emit the input's columns followed by the fetched ones, which is why the node's
    // schema holds either way. See [`v1::take`] for why v1 needs its own.
    if v1::is_legacy(node.dataset()) {
        return v1::take(
            node.dataset(),
            input,
            node.projection().clone(),
            node.settings().batch_size,
        );
    }
    let options = take_options(node, input.schema().as_ref());
    Ok(Arc::new(FilteredReadExec::try_new(
        node.dataset().clone(),
        options,
        Some(input),
    )?))
}

/// Carry through whichever identity columns the input already has, so the take does not drop
/// them. Mirrors `Scanner::take_current`.
pub fn take_options(
    node: &LanceTakeNode,
    input_schema: &arrow_schema::Schema,
) -> FilteredReadOptions {
    let mut projection = node.projection().clone();
    projection.with_row_id |= input_schema.column_with_name(ROW_ID).is_some();
    projection.with_row_addr |= input_schema
        .column_with_name(lance_core::ROW_ADDR)
        .is_some();
    let mut options = FilteredReadOptions::new(projection);
    if let Some(batch_size) = node.settings().batch_size {
        options = options.with_batch_size(batch_size);
    }
    if let Some(fragments) = &node.settings().fragments {
        options = options.with_fragments(fragments.clone());
    }
    options
}
