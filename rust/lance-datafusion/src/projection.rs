// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::datatypes::{Field as ArrowField, Schema as ArrowSchema};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, SchemaRef};
use datafusion::{
    execution::SendableRecordBatchStream, logical_expr::Expr,
    physical_plan::projection::ProjectionExec,
};
use datafusion_common::DFSchema;
use datafusion_physical_expr::{expressions, PhysicalExpr};
use futures::TryStreamExt;
use snafu::{location, Location};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use lance_core::{
    datatypes::{Field, Schema, BLOB_DESC_FIELDS, BLOB_META_KEY},
    Error, Result,
};

use crate::{
    exec::{execute_plan, LanceExecutionOptions, OneShotExec},
    planner::Planner,
};

#[derive(Debug)]
pub struct ProjectionPlan {
    /// The physical schema (before dynamic projection) that must be loaded from the dataset
    pub physical_schema: Arc<Schema>,
    pub physical_df_schema: Arc<DFSchema>,

    /// The schema of the sibling fields that must be loaded
    pub sibling_schema: Option<Arc<Schema>>,

    /// The expressions for all the columns to be in the output
    /// Note: this doesn't include _distance, and _rowid
    pub requested_output_expr: Option<Vec<(Expr, String)>>,
}

impl ProjectionPlan {
    fn unload_blobs(schema: &Arc<Schema>) -> Arc<Schema> {
        let mut modified = false;
        let fields = schema
            .fields
            .iter()
            .map(|f| {
                if f.metadata.contains_key(BLOB_META_KEY) {
                    debug_assert!(f.data_type() == DataType::LargeBinary);
                    modified = true;
                    let mut unloaded_field = Field::try_from(ArrowField::new(
                        f.name.clone(),
                        DataType::Struct(BLOB_DESC_FIELDS.clone()),
                        f.nullable,
                    ))
                    .unwrap();
                    unloaded_field.id = f.id;
                    unloaded_field
                } else {
                    f.clone()
                }
            })
            .collect();

        if modified {
            let mut schema = schema.as_ref().clone();
            schema.fields = fields;
            Arc::new(schema)
        } else {
            schema.clone()
        }
    }

    pub fn try_new(
        base_schema: &Schema,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
        load_blobs: bool,
    ) -> Result<Self> {
        let arrow_schema = Arc::new(ArrowSchema::from(base_schema));
        let planner = Planner::new(arrow_schema);
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

        let physical_schema = Arc::new(base_schema.project(&physical_cols)?);
        let (physical_schema, sibling_schema) = physical_schema.partition_by_storage_class();
        let mut physical_schema = Arc::new(physical_schema);
        if !load_blobs {
            physical_schema = Self::unload_blobs(&physical_schema);
        }

        let mut output_cols = vec![];
        for (name, _) in columns {
            output_cols.push((output[name.as_ref()].clone(), name.as_ref().to_string()));
        }
        let requested_output_expr = Some(output_cols);
        let physical_arrow_schema = ArrowSchema::from(physical_schema.as_ref());
        let physical_df_schema = Arc::new(DFSchema::try_from(physical_arrow_schema).unwrap());
        Ok(Self {
            physical_schema,
            sibling_schema: sibling_schema.map(Arc::new),
            physical_df_schema,
            requested_output_expr,
        })
    }

    pub fn new_empty(base_schema: Arc<Schema>, load_blobs: bool) -> Self {
        let (physical_schema, sibling_schema) = base_schema.partition_by_storage_class();
        Self::inner_new(
            Arc::new(physical_schema),
            load_blobs,
            sibling_schema.map(Arc::new),
        )
    }

    pub fn inner_new(
        base_schema: Arc<Schema>,
        load_blobs: bool,
        sibling_schema: Option<Arc<Schema>>,
    ) -> Self {
        let physical_schema = if !load_blobs {
            Self::unload_blobs(&base_schema)
        } else {
            base_schema
        };

        let physical_arrow_schema = ArrowSchema::from(physical_schema.as_ref());
        let physical_df_schema = Arc::new(DFSchema::try_from(physical_arrow_schema).unwrap());
        Self {
            physical_schema,
            sibling_schema,
            physical_df_schema,
            requested_output_expr: None,
        }
    }

    pub fn arrow_schema(&self) -> &ArrowSchema {
        self.physical_df_schema.as_arrow()
    }

    pub fn arrow_schema_ref(&self) -> SchemaRef {
        Arc::new(self.physical_df_schema.as_arrow().clone())
    }

    pub fn to_physical_exprs(&self) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        if let Some(output_expr) = &self.requested_output_expr {
            output_expr
                .iter()
                .map(|(expr, name)| {
                    Ok((
                        datafusion::physical_expr::create_physical_expr(
                            expr,
                            self.physical_df_schema.as_ref(),
                            &Default::default(),
                        )?,
                        name.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()
        } else {
            self.physical_schema
                .fields
                .iter()
                .map(|f| {
                    Ok((
                        expressions::col(f.name.as_str(), self.physical_df_schema.as_arrow())?
                            .clone(),
                        f.name.clone(),
                    ))
                })
                .collect::<Result<Vec<_>>>()
        }
    }

    pub fn output_schema(&self) -> Result<ArrowSchema> {
        let exprs = self.to_physical_exprs()?;
        let fields = exprs
            .iter()
            .map(|(expr, name)| {
                Ok(ArrowField::new(
                    name,
                    expr.data_type(self.arrow_schema())?,
                    expr.nullable(self.arrow_schema())?,
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
        let physical_exprs = self.to_physical_exprs()?;
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

    pub fn project_stream(
        &self,
        stream: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        if self.requested_output_expr.is_none() {
            return Ok(stream);
        }
        let src = Arc::new(OneShotExec::new(stream));
        let physical_exprs = self.to_physical_exprs()?;
        let projection = Arc::new(ProjectionExec::try_new(physical_exprs, src)?);
        execute_plan(projection, LanceExecutionOptions::default())
    }
}
