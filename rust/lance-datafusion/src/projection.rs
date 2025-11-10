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
    Error, Result, ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION,
    ROW_OFFSET, WILDCARD,
};

use crate::{
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
    planner::Planner,
};

struct ProjectionBuilder {
    base: Arc<dyn Projectable>,
    planner: Planner,
    output: HashMap<String, Expr>,
    output_cols: Vec<OutputColumn>,
    physical_cols_set: HashSet<String>,
    physical_cols: Vec<String>,
    needs_row_id: bool,
    needs_row_addr: bool,
    needs_row_last_updated_at: bool,
    needs_row_created_at: bool,
    must_add_row_offset: bool,
    has_wildcard: bool,
}

impl ProjectionBuilder {
    fn new(base: Arc<dyn Projectable>) -> Self {
        let full_schema = Arc::new(Projection::full(base.clone()).to_arrow_schema());
        let full_schema = Arc::new(ProjectionPlan::add_system_columns(&full_schema));
        let planner = Planner::new(full_schema);

        Self {
            base,
            planner,
            output: HashMap::default(),
            output_cols: Vec::default(),
            physical_cols_set: HashSet::default(),
            physical_cols: Vec::default(),
            needs_row_id: false,
            needs_row_addr: false,
            needs_row_created_at: false,
            needs_row_last_updated_at: false,
            must_add_row_offset: false,
            has_wildcard: false,
        }
    }

    fn check_duplicate_column(&self, name: &str) -> Result<()> {
        if self.output.contains_key(name) {
            return Err(Error::io(
                format!("Duplicate column name: {}", name),
                location!(),
            ));
        }
        Ok(())
    }

    fn add_column(&mut self, output_name: &str, raw_expr: &str) -> Result<()> {
        self.check_duplicate_column(output_name)?;

        let expr = self.planner.parse_expr(raw_expr)?;

        // If the expression is a bare column reference to a system column, mark that we need it
        if let Expr::Column(Column {
            name,
            relation: None,
            ..
        }) = &expr
        {
            if name == ROW_ID {
                self.needs_row_id = true;
            } else if name == ROW_ADDR {
                self.needs_row_addr = true;
            } else if name == ROW_OFFSET {
                self.must_add_row_offset = true;
            } else if name == ROW_LAST_UPDATED_AT_VERSION {
                self.needs_row_last_updated_at = true;
            } else if name == ROW_CREATED_AT_VERSION {
                self.needs_row_created_at = true;
            }
        }

        for col in Planner::column_names_in_expr(&expr) {
            if self.physical_cols_set.contains(&col) {
                continue;
            }
            self.physical_cols.push(col.clone());
            self.physical_cols_set.insert(col);
        }
        self.output.insert(output_name.to_string(), expr.clone());

        self.output_cols.push(OutputColumn {
            expr,
            name: output_name.to_string(),
        });

        Ok(())
    }

    fn add_columns(&mut self, columns: &[(impl AsRef<str>, impl AsRef<str>)]) -> Result<()> {
        for (output_name, raw_expr) in columns {
            if raw_expr.as_ref() == WILDCARD {
                self.has_wildcard = true;
                for col in self.base.schema().fields.iter().map(|f| f.name.as_str()) {
                    self.check_duplicate_column(col)?;
                    self.output_cols.push(OutputColumn {
                        expr: Expr::Column(Column::from_name(col)),
                        name: col.to_string(),
                    });
                    // Throw placeholder expr in self.output, this will trigger error on duplicates
                    self.output.insert(col.to_string(), Expr::default());
                }
            } else {
                self.add_column(output_name.as_ref(), raw_expr.as_ref())?;
            }
        }
        Ok(())
    }

    fn build(self) -> Result<ProjectionPlan> {
        // Now, calculate the physical projection from the columns referenced by the expressions
        //
        // If a column is missing it might be a system column (_rowid, _distance, etc.) and so
        // we ignore it.  We don't need to load that column from disk at least, which is all we are
        // trying to calculate here.
        let mut physical_projection = if self.has_wildcard {
            Projection::full(self.base.clone())
        } else {
            Projection::empty(self.base.clone())
                .union_columns(&self.physical_cols, OnMissing::Ignore)?
        };

        physical_projection.with_row_id = self.needs_row_id;
        physical_projection.with_row_addr = self.needs_row_addr || self.must_add_row_offset;
        physical_projection.with_row_last_updated_at_version = self.needs_row_last_updated_at;
        physical_projection.with_row_created_at_version = self.needs_row_created_at;

        Ok(ProjectionPlan {
            physical_projection,
            must_add_row_offset: self.must_add_row_offset,
            requested_output_expr: self.output_cols,
        })
    }
}

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
        fields.push(Arc::new(
            (*lance_core::ROW_LAST_UPDATED_AT_VERSION_FIELD).clone(),
        ));
        fields.push(Arc::new(
            (*lance_core::ROW_CREATED_AT_VERSION_FIELD).clone(),
        ));
        ArrowSchema::new(fields)
    }

    /// Set the projection from SQL expressions
    pub fn from_expressions(
        base: Arc<dyn Projectable>,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<Self> {
        let mut builder = ProjectionBuilder::new(base);
        builder.add_columns(columns)?;
        builder.build()
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
        // Separate data columns from system columns
        // System columns (_rowid, _rowaddr, etc.) are handled via flags in Projection,
        // not as fields in the Schema
        let mut data_fields = Vec::new();
        let mut with_row_id = false;
        let mut with_row_addr = false;
        let mut must_add_row_offset = false;

        for field in projection.fields.iter() {
            if lance_core::is_system_column(&field.name) {
                // Handle known system columns that can be included in projections
                if field.name == ROW_ID {
                    with_row_id = true;
                    must_add_row_offset = true;
                } else if field.name == ROW_ADDR {
                    with_row_addr = true;
                    must_add_row_offset = true;
                }
                // Note: Other system columns like _rowoffset are computed differently
                // and shouldn't appear in the schema at this point
            } else {
                // Regular data column - validate it exists in base schema
                if base.schema().field(&field.name).is_none() {
                    return Err(Error::io(
                        format!("Column '{}' not found in schema", field.name),
                        location!(),
                    ));
                }
                data_fields.push(field.clone());
            }
        }

        // Create a schema with only data columns for the physical projection
        let data_schema = Schema {
            fields: data_fields,
            metadata: projection.metadata.clone(),
        };

        // Calculate the physical projection from data columns only
        let mut physical_projection = Projection::empty(base).union_schema(&data_schema);
        physical_projection.with_row_id = with_row_id;
        physical_projection.with_row_addr = with_row_addr;

        // Build output expressions preserving the original order (including system columns)
        let exprs = projection
            .fields
            .iter()
            .map(|f| OutputColumn {
                expr: Expr::Column(Column::from_name(&f.name)),
                name: f.name.clone(),
            })
            .collect::<Vec<_>>();

        Ok(Self {
            physical_projection,
            requested_output_expr: exprs,
            must_add_row_offset,
        })
    }

    pub fn full(base: Arc<dyn Projectable>) -> Result<Self> {
        let physical_cols: Vec<&str> = base
            .schema()
            .fields
            .iter()
            .map(|f| f.name.as_ref())
            .collect::<Vec<_>>();

        let physical_projection =
            Projection::empty(base.clone()).union_columns(&physical_cols, OnMissing::Ignore)?;

        let requested_output_expr = physical_cols
            .into_iter()
            .map(|col_name| OutputColumn {
                expr: Expr::Column(Column::from_name(col_name)),
                name: col_name.to_string(),
            })
            .collect();

        Ok(Self {
            physical_projection,
            must_add_row_offset: false,
            requested_output_expr,
        })
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
