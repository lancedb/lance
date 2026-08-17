// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Late materialization as a logical node: fetch the columns a search did not
//! produce, keyed by row id.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use lance_core::ROW_ID;
use lance_core::datatypes::Projection;
use lance_table::format::Fragment;

use crate::Result;
use crate::dataset::Dataset;
use crate::io::exec::TakeExec;

/// The read settings every take in a plan has to honor, carried down from the `Scanner`.
///
/// A take is a read, so a fragment restriction applies to it: rows the search found outside the
/// target fragments are dropped here. That is how `with_fragments` reaches a search whose index
/// covers more fragments than the scan asked for.
#[derive(Debug, Clone, Default)]
pub struct TakeSettings {
    pub fragments: Option<Arc<Vec<Fragment>>>,
    pub batch_size: Option<u32>,
}

/// Late materialization: fetch `projection`'s columns for rows the input has already identified,
/// keyed by `_rowid`.
///
/// The physical form is a row-stream `FilteredReadExec`, whose output is the input's columns
/// followed by the newly fetched ones. That ordering is reproduced here with the same helper the
/// physical node uses, so the logical and physical schemas cannot drift.
#[derive(Debug, Clone)]
pub struct LanceTakeNode {
    input: LogicalPlan,
    dataset: Arc<Dataset>,
    projection: Projection,
    settings: TakeSettings,
    schema: DFSchemaRef,
}

impl LanceTakeNode {
    pub fn try_new(
        input: LogicalPlan,
        dataset: Arc<Dataset>,
        projection: Projection,
        settings: TakeSettings,
    ) -> Result<Self> {
        let schema = Self::output_schema(&input, &dataset, &projection)?;
        Ok(Self {
            input,
            dataset,
            projection,
            settings,
            schema,
        })
    }

    /// Whether a take is needed at all: if the input already carries every projected column,
    /// the node is a no-op and the builder should skip it.
    pub fn is_noop(input: &LogicalPlan, projection: &Projection) -> Result<bool> {
        let input_schema = input.schema().as_arrow().clone();
        let missing = projection
            .clone()
            .subtract_arrow_schema(&input_schema, lance_core::datatypes::OnMissing::Ignore)?;
        Ok(!missing.has_data_fields() && !missing.with_row_id && !missing.with_row_addr)
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    pub fn settings(&self) -> &TakeSettings {
        &self.settings
    }

    fn output_schema(
        input: &LogicalPlan,
        dataset: &Dataset,
        projection: &Projection,
    ) -> Result<DFSchemaRef> {
        let input_schema = input.schema().as_arrow().clone();
        let fields_to_read = projection
            .clone()
            .subtract_arrow_schema(&input_schema, lance_core::datatypes::OnMissing::Ignore)?;
        let output =
            TakeExec::calculate_output_schema(dataset.schema(), &input_schema, &fields_to_read);
        Ok(Arc::new(DFSchema::try_from(ArrowSchema::from(&output))?))
    }
}

impl PartialEq for LanceTakeNode {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.dataset.base == other.dataset.base
            && self.projection.field_ids == other.projection.field_ids
    }
}

impl Eq for LanceTakeNode {}

impl Hash for LanceTakeNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.dataset.base.hash(state);
        let mut ids = self
            .projection
            .field_ids
            .iter()
            .copied()
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.hash(state);
    }
}

impl PartialOrd for LanceTakeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.input.partial_cmp(&other.input)
    }
}

impl UserDefinedLogicalNodeCore for LanceTakeNode {
    fn name(&self) -> &str {
        "LanceTake"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut columns = self
            .projection
            .to_bare_schema()
            .fields
            .iter()
            .map(|field| field.name.clone())
            .collect::<Vec<_>>();
        columns.sort();
        write!(
            f,
            "LanceTake: columns=[{}] by={}",
            columns.join(", "),
            ROW_ID
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        if !exprs.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "LanceTake takes no expressions".into(),
            ));
        }
        if inputs.len() != 1 {
            return Err(datafusion::common::DataFusionError::Internal(format!(
                "LanceTake takes exactly one input, got {}",
                inputs.len()
            )));
        }
        Self::try_new(
            inputs.remove(0),
            self.dataset.clone(),
            self.projection.clone(),
            self.settings.clone(),
        )
        .map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))
    }

    /// Every input column carries through to the output, so nothing below can be dropped.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        Some(vec![(0..self.input.schema().fields().len()).collect()])
    }
}
