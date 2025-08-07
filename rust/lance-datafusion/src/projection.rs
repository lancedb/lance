// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use datafusion::{logical_expr::Expr, physical_plan::projection::ProjectionExec};
use datafusion_common::DFSchema;
use datafusion_physical_expr::{expressions, PhysicalExpr};
use futures::TryStreamExt;
use snafu::location;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use lance_core::{
    datatypes::{OnMissing, Projectable, Projection, Schema},
    Error, Result,
};

use crate::{
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
    planner::Planner,
};

#[derive(Clone, Debug)]
pub struct ProjectionPlan {
    /// The base thing we are projecting from (e.g. a dataset)
    base: Arc<dyn Projectable>,
    /// The physical schema that must be loaded from the dataset
    pub physical_projection: Projection,

    /// True if the user wants the row id in the final output
    ///
    /// Note: this is related, but slightly different, to physical_projection.with_row_id
    /// which only tracks if the row id is needed.
    ///
    /// desires_row_id implies with_row_id is true
    /// However, it is possible to have desires_row_id=false and with_row_id=true (e.g. when
    /// the row id is needed to perform a late materialization take)
    pub desires_row_id: bool,
    /// True if the user wants the row address in the final output
    ///
    /// Note: this is related, but slightly different, to physical_projection.with_row_addr
    /// which only tracks if the row address is needed.
    ///
    /// desires_row_addr implies with_row_addr is true
    /// However, it is possible to have deisres_row_addr=false and with_row_addr=true (e.g. during
    /// a count query)
    pub desires_row_addr: bool,

    /// If present, expressions that represent the output columns.  These expressions
    /// run on the output of the physical projection.
    ///
    /// If not present, the output is the physical projection.
    ///
    /// Note: this doesn't include _distance, and _rowid
    pub requested_output_expr: Option<Vec<(Expr, String)>>,
}

impl ProjectionPlan {
    /// Create a new projection plan which projects all columns and does not include any expressions
    pub fn new(base: Arc<dyn Projectable>) -> Self {
        let physical_projection = Projection::full(base.clone());
        Self {
            base,
            physical_projection,
            requested_output_expr: None,
            desires_row_addr: false,
            desires_row_id: false,
        }
    }

    /// Set the projection from SQL expressions
    pub fn project_from_expressions(
        &mut self,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<()> {
        // Save off values of with_row_id / with_row_addr
        let had_row_id = self.physical_projection.with_row_id;
        let had_row_addr = self.physical_projection.with_row_addr;

        // First, look at the expressions to figure out which physical columns are needed
        let full_schema = Arc::new(Projection::full(self.base.clone()).to_arrow_schema());
        let planner = Planner::new(full_schema);
        let mut output = HashMap::new();
        let mut physical_cols_set = HashSet::new();
        let mut physical_cols = vec![];
        for (output_name, raw_expr) in columns {
            if output.contains_key(output_name.as_ref()) {
                return Err(Error::io(
                    format!("Duplicate column name: {}", output_name.as_ref()),
                    location!(),
                ));
            }
            let expr = planner.parse_expr(raw_expr.as_ref())?;
            for col in Planner::column_names_in_expr(&expr) {
                if physical_cols_set.contains(&col) {
                    continue;
                }
                physical_cols.push(col.clone());
                physical_cols_set.insert(col);
            }
            output.insert(output_name.as_ref().to_string(), expr);
        }

        // Now, calculate the physical projection from the columns referenced by the expressions
        //
        // If a column is missing it might be a metadata column (_rowid, _distance, etc.) and so
        // we ignore it.  We don't need to load that column from disk at least, which is all we are
        // trying to calculate here.
        let mut physical_projection = Projection::empty(self.base.clone())
            .union_columns(&physical_cols, OnMissing::Ignore)?;

        // Restore the row_id and row_addr flags
        physical_projection.with_row_id = had_row_id;
        physical_projection.with_row_addr = had_row_addr;

        self.physical_projection = physical_projection;

        // Save off the expressions (they will be evaluated later to run the projection)
        let mut output_cols = vec![];
        for (name, _) in columns {
            output_cols.push((output[name.as_ref()].clone(), name.as_ref().to_string()));
        }
        self.requested_output_expr = Some(output_cols);

        Ok(())
    }

    /// Set the projection from a schema
    ///
    /// This plan will have no complex expressions
    pub fn project_from_schema(&mut self, projection: &Schema) {
        let had_row_id = self.physical_projection.with_row_id;
        let had_row_addr = self.physical_projection.with_row_addr;

        let mut physical_projection = Projection::empty(self.base.clone()).union_schema(projection);

        physical_projection.with_row_id = had_row_id;
        physical_projection.with_row_addr = had_row_addr;

        self.physical_projection = physical_projection;
    }

    /// Convert the projection to a list of physical expressions
    ///
    /// This is used to apply the final projection (including dynamic expressions) to the data.
    pub fn to_physical_exprs(
        &self,
        current_schema: &ArrowSchema,
    ) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        let physical_df_schema = Arc::new(DFSchema::try_from(current_schema.clone())?);
        if let Some(output_expr) = &self.requested_output_expr {
            output_expr
                .iter()
                .map(|(expr, name)| {
                    Ok((
                        datafusion::physical_expr::create_physical_expr(
                            expr,
                            physical_df_schema.as_ref(),
                            &Default::default(),
                        )?,
                        name.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()
        } else {
            let projection_schema = self.physical_projection.to_schema();
            projection_schema
                .fields
                .iter()
                .map(|f| {
                    Ok((
                        expressions::col(f.name.as_str(), physical_df_schema.as_arrow())?.clone(),
                        f.name.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()
        }
    }

    /// Include the row id in the output
    pub fn include_row_id(&mut self) {
        self.physical_projection.with_row_id = true;
        self.desires_row_id = true;
    }

    /// Include the row address in the output
    pub fn include_row_addr(&mut self) {
        self.physical_projection.with_row_addr = true;
        self.desires_row_addr = true;
    }

    /// Check if the projection has any output columns
    ///
    /// This doesn't mean there is a physical projection.  For example, we may someday support
    /// something like `SELECT 1 AS foo` which would have an output column (foo) but no physical projection
    pub fn has_output_cols(&self) -> bool {
        if self.desires_row_id || self.desires_row_addr {
            return true;
        }
        if let Some(exprs) = &self.requested_output_expr {
            if !exprs.is_empty() {
                return true;
            }
        }
        self.physical_projection.has_non_meta_cols()
    }

    pub fn output_schema(&self) -> Result<ArrowSchema> {
        let exprs = self.to_physical_exprs(&self.physical_projection.to_arrow_schema())?;
        let physical_schema = self.physical_projection.to_arrow_schema();
        let fields = exprs
            .iter()
            .map(|(expr, name)| {
                Ok(ArrowField::new(
                    name,
                    expr.data_type(&physical_schema)?,
                    expr.nullable(&physical_schema)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ArrowSchema::new(fields))
    }

    pub async fn project_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if self.requested_output_expr.is_none() {
            return Ok(batch);
        }
        let src = Arc::new(OneShotExec::from_batch(batch));
        let physical_exprs = self.to_physical_exprs(&self.physical_projection.to_arrow_schema())?;
        let projection = Arc::new(ProjectionExec::try_new(physical_exprs, src)?);
        let stream = execute_plan(projection, LanceExecutionOptions::default())?;
        let batches = stream.try_collect::<Vec<_>>().await?;
        if batches.len() != 1 {
            Err(Error::Internal {
                message: "Expected exactly one batch".to_string(),
                location: location!(),
            })
        } else {
            Ok(batches.into_iter().next().unwrap())
        }
    }
}
