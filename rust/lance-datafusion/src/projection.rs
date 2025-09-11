// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::{logical_expr::Expr, physical_plan::projection::ProjectionExec};
use datafusion_common::{Column, DFSchema};
use datafusion_physical_expr::PhysicalExpr;
use futures::TryStreamExt;
use snafu::location;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use lance_core::{
    datatypes::{OnMissing, Projectable, Projection, Schema},
    Error, Result, ROW_ADDR, ROW_ID, ROW_OFFSET,
};

use crate::{
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
    planner::Planner,
};

#[derive(Clone, Debug)]
pub struct OutputColumn {
    /// The expression that represents the output column
    pub expr: Expr,
    /// The name of the output column
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct ProjectionPlan {
    /// The physical schema that must be loaded from the dataset
    pub physical_projection: Projection,

    /// Needs the row address converted into a row offset
    pub must_add_row_offset: bool,

    /// The desired output columns
    pub requested_output_expr: Vec<OutputColumn>,
}

impl ProjectionPlan {
    fn add_system_columns(schema: &ArrowSchema) -> ArrowSchema {
        let mut fields = Vec::from_iter(schema.fields.iter().cloned());
        fields.push(Arc::new(ArrowField::new(ROW_ID, DataType::UInt64, true)));
        fields.push(Arc::new(ArrowField::new(ROW_ADDR, DataType::UInt64, true)));
        fields.push(Arc::new(ArrowField::new(
            ROW_OFFSET,
            DataType::UInt64,
            true,
        )));
        ArrowSchema::new(fields)
    }

    /// Set the projection from SQL expressions
    pub fn from_expressions(
        base: Arc<dyn Projectable>,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<Self> {
        // First, look at the expressions to figure out which physical columns are needed
        let full_schema = Arc::new(Projection::full(base.clone()).to_arrow_schema());
        let full_schema = Arc::new(Self::add_system_columns(&full_schema));
        let planner = Planner::new(full_schema);
        let mut output = HashMap::new();
        let mut physical_cols_set = HashSet::new();
        let mut physical_cols = vec![];
        let mut needs_row_id = false;
        let mut needs_row_addr = false;
        let mut must_add_row_offset = false;
        for (output_name, raw_expr) in columns {
            if output.contains_key(output_name.as_ref()) {
                return Err(Error::io(
                    format!("Duplicate column name: {}", output_name.as_ref()),
                    location!(),
                ));
            }

            let expr = planner.parse_expr(raw_expr.as_ref())?;

            // If the expression is a bare column reference to a system column, mark that we need it
            if let Expr::Column(Column {
                name,
                relation: None,
                ..
            }) = &expr
            {
                if name == ROW_ID {
                    needs_row_id = true;
                } else if name == ROW_ADDR {
                    needs_row_addr = true;
                } else if name == ROW_OFFSET {
                    must_add_row_offset = true;
                }
            }

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
        let mut physical_projection =
            Projection::empty(base.clone()).union_columns(&physical_cols, OnMissing::Ignore)?;

        physical_projection.with_row_id = needs_row_id;
        physical_projection.with_row_addr = needs_row_addr || must_add_row_offset;

        // Save off the expressions (they will be evaluated later to run the projection)
        let mut output_cols = vec![];
        for (name, _) in columns {
            output_cols.push(OutputColumn {
                expr: output[name.as_ref()].clone(),
                name: name.as_ref().to_string(),
            });
        }

        Ok(Self {
            physical_projection,
            must_add_row_offset,
            requested_output_expr: output_cols,
        })
    }

    /// Set the projection from a schema
    ///
    /// This plan will have no complex expressions, the schema must be a subset of the dataset schema.
    ///
    /// With this approach it is possible to refer to portions of nested fields.
    ///
    /// For example, if the schema is:
    ///
    /// ```ignore
    /// {
    ///   "metadata": {
    ///     "location": {
    ///       "x": f32,
    ///       "y": f32,
    ///     },
    ///     "age": i32,
    ///   }
    /// }
    /// ```
    ///
    /// It is possible to project a partial schema that drops `y` like:
    ///
    /// ```ignore
    /// {
    ///   "metadata": {
    ///     "location": {
    ///       "x": f32,
    ///     },
    ///     "age": i32,
    ///   }
    /// }
    /// ```
    ///
    /// This is something that cannot be done easily using expressions.
    pub fn from_schema(base: Arc<dyn Projectable>, projection: &Schema) -> Result<Self> {
        // Calculate the physical projection directly from the schema
        //
        // The _rowid and _rowaddr columns will be recognized and added to the physical projection
        //
        // Any columns with an id of -1 (e.g. _rowoffset) will be ignored
        let physical_projection = Projection::empty(base).union_schema(projection);
        let mut must_add_row_offset = false;
        // Now calculate the output expressions.  This will only reorder top-level columns.  We don't
        // support reordering nested fields.
        let exprs = projection
            .fields
            .iter()
            .map(|f| {
                if f.name == ROW_ADDR {
                    must_add_row_offset = true;
                }
                OutputColumn {
                    expr: Expr::Column(Column::from_name(&f.name)),
                    name: f.name.clone(),
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            physical_projection,
            requested_output_expr: exprs,
            must_add_row_offset,
        })
    }

    pub fn full(base: Arc<dyn Projectable>) -> Result<Self> {
        let projection = base
            .schema()
            .fields
            .iter()
            .map(|f| (f.name.as_str(), format!("`{}`", f.name.as_str())))
            .collect::<Vec<_>>();
        Self::from_expressions(base.clone(), &projection)
    }

    /// Convert the projection to a list of physical expressions
    ///
    /// This is used to apply the final projection (including dynamic expressions) to the data.
    pub fn to_physical_exprs(
        &self,
        current_schema: &ArrowSchema,
    ) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        let physical_df_schema = Arc::new(DFSchema::try_from(current_schema.clone())?);
        self.requested_output_expr
            .iter()
            .map(|output_column| {
                Ok((
                    datafusion::physical_expr::create_physical_expr(
                        &output_column.expr,
                        physical_df_schema.as_ref(),
                        &Default::default(),
                    )?,
                    output_column.name.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Include the row id in the output
    pub fn include_row_id(&mut self) {
        self.physical_projection.with_row_id = true;
        if !self
            .requested_output_expr
            .iter()
            .any(|OutputColumn { name, .. }| name == ROW_ID)
        {
            self.requested_output_expr.push(OutputColumn {
                expr: Expr::Column(Column::from_name(ROW_ID)),
                name: ROW_ID.to_string(),
            });
        }
    }

    /// Include the row address in the output
    pub fn include_row_addr(&mut self) {
        self.physical_projection.with_row_addr = true;
        if !self
            .requested_output_expr
            .iter()
            .any(|OutputColumn { name, .. }| name == ROW_ADDR)
        {
            self.requested_output_expr.push(OutputColumn {
                expr: Expr::Column(Column::from_name(ROW_ADDR)),
                name: ROW_ADDR.to_string(),
            });
        }
    }

    pub fn include_row_offset(&mut self) {
        // Need row addr to get row offset
        self.physical_projection.with_row_addr = true;
        self.must_add_row_offset = true;
        if !self
            .requested_output_expr
            .iter()
            .any(|OutputColumn { name, .. }| name == ROW_OFFSET)
        {
            self.requested_output_expr.push(OutputColumn {
                expr: Expr::Column(Column::from_name(ROW_OFFSET)),
                name: ROW_OFFSET.to_string(),
            });
        }
    }

    /// Check if the projection has any output columns
    ///
    /// This doesn't mean there is a physical projection.  For example, we may someday support
    /// something like `SELECT 1 AS foo` which would have an output column (foo) but no physical projection
    pub fn has_output_cols(&self) -> bool {
        !self.requested_output_expr.is_empty()
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
