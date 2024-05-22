// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, sync::Arc};

use crate::io::commit::commit_transaction;
use crate::{io::exec::Planner, Error, Result};
use arrow::compute::CastOptions;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::stream::{StreamExt, TryStreamExt};
use lance_arrow::SchemaExt;
use lance_core::datatypes::{Field, Schema};
use lance_table::format::Fragment;
use snafu::{location, Location};

use super::{
    transaction::{Operation, Transaction},
    Dataset,
};

#[derive(Debug, Clone, PartialEq)]
pub struct BatchInfo {
    pub fragment_id: u32,
    pub batch_index: usize,
}

/// A mechanism for saving UDF results.
///
/// This is used to determine if a UDF has already been run on a given input,
/// and to store the results of a UDF for future use.
pub trait UDFCheckpointStore: Send + Sync {
    fn get_batch(&self, info: &BatchInfo) -> Result<Option<RecordBatch>>;
    fn insert_batch(&self, info: BatchInfo, batch: RecordBatch) -> Result<()>;
    fn get_fragment(&self, fragment_id: u32) -> Result<Option<Fragment>>;
    fn insert_fragment(&self, fragment: Fragment) -> Result<()>;
}

pub struct BatchUDF {
    #[allow(clippy::type_complexity)]
    pub mapper: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>,
    /// The schema of the returned RecordBatch
    pub output_schema: Arc<ArrowSchema>,
    /// A checkpoint store for the UDF results
    pub result_checkpoint: Option<Arc<dyn UDFCheckpointStore>>,
}

/// A way to define one or more new columns in a dataset
pub enum NewColumnTransform {
    /// A UDF that takes a RecordBatch of existing data and returns a
    /// RecordBatch with the new columns for those corresponding rows. The returned
    /// batch must return the same number of rows as the input batch.
    BatchUDF(BatchUDF),
    /// A set of SQL expressions that define new columns.
    SqlExpressions(Vec<(String, String)>),
}

/// Definition of a change to a column in a dataset
pub struct ColumnAlteration {
    /// Path to the existing column to be altered.
    pub path: String,
    /// The new name of the column. If None, the column name will not be changed.
    pub rename: Option<String>,
    /// Whether the column is nullable. If None, the nullability will not be changed.
    pub nullable: Option<bool>,
    /// The new data type of the column. If None, the data type will not be changed.
    pub data_type: Option<DataType>,
}

impl ColumnAlteration {
    pub fn new(path: String) -> Self {
        Self {
            path,
            rename: None,
            nullable: None,
            data_type: None,
        }
    }

    pub fn rename(mut self, name: String) -> Self {
        self.rename = Some(name);
        self
    }

    pub fn set_nullable(mut self, nullable: bool) -> Self {
        self.nullable = Some(nullable);
        self
    }

    pub fn cast_to(mut self, data_type: DataType) -> Self {
        self.data_type = Some(data_type);
        self
    }
}

/// Limit casts to same type. This is mostly to filter out weird casts like
/// casting a string to a boolean or float to string.
fn is_upcast_downcast(from_type: &DataType, to_type: &DataType) -> bool {
    use DataType::*;
    match from_type {
        from_type if from_type.is_integer() => to_type.is_integer(),
        from_type if from_type.is_floating() => to_type.is_floating(),
        from_type if from_type.is_temporal() => to_type.is_temporal(),
        Boolean => matches!(to_type, Boolean),
        Utf8 | LargeUtf8 => matches!(to_type, Utf8 | LargeUtf8),
        Binary | LargeBinary => matches!(to_type, Binary | LargeBinary),
        Decimal128(_, _) | Decimal256(_, _) => {
            matches!(to_type, Decimal128(_, _) | Decimal256(_, _))
        }
        List(from_field) | LargeList(from_field) | FixedSizeList(from_field, _) => match to_type {
            List(to_field) | LargeList(to_field) | FixedSizeList(to_field, _) => {
                is_upcast_downcast(from_field.data_type(), to_field.data_type())
            }
            _ => false,
        },
        _ => false,
    }
}

pub(super) async fn add_columns(
    dataset: &mut Dataset,
    transforms: NewColumnTransform,
    read_columns: Option<Vec<String>>,
) -> Result<()> {
    // We just transform the SQL expression into a UDF backed by DataFusion
    // physical expressions.
    let (
        BatchUDF {
            mapper,
            output_schema,
            result_checkpoint,
        },
        read_columns,
    ) = match transforms {
        NewColumnTransform::BatchUDF(udf) => (udf, read_columns),
        NewColumnTransform::SqlExpressions(expressions) => {
            let arrow_schema = Arc::new(ArrowSchema::from(dataset.schema()));
            let planner = Planner::new(arrow_schema);
            let exprs = expressions
                .into_iter()
                .map(|(name, expr)| {
                    let expr = planner.parse_expr(&expr)?;
                    let expr = planner.optimize_expr(expr)?;
                    Ok((name, expr))
                })
                .collect::<Result<Vec<_>>>()?;

            let needed_columns = exprs
                .iter()
                .flat_map(|(_, expr)| Planner::column_names_in_expr(expr))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let read_schema = dataset.schema().project(&needed_columns)?;
            let read_schema = Arc::new(ArrowSchema::from(&read_schema));
            // Need to re-create the planner with the read schema because physical
            // expressions use positional column references.
            let planner = Planner::new(read_schema.clone());
            let exprs = exprs
                .into_iter()
                .map(|(name, expr)| {
                    let expr = planner.create_physical_expr(&expr)?;
                    Ok((name, expr))
                })
                .collect::<Result<Vec<_>>>()?;

            let output_schema = Arc::new(ArrowSchema::new(
                exprs
                    .iter()
                    .map(|(name, expr)| {
                        Ok(ArrowField::new(
                            name,
                            expr.data_type(read_schema.as_ref())?,
                            expr.nullable(read_schema.as_ref())?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ));

            let schema_ref = output_schema.clone();
            let mapper = move |batch: &RecordBatch| {
                let num_rows = batch.num_rows();
                let columns = exprs
                    .iter()
                    .map(|(_, expr)| Ok(expr.evaluate(batch)?.into_array(num_rows)?))
                    .collect::<Result<Vec<_>>>()?;

                let batch = RecordBatch::try_new(schema_ref.clone(), columns)?;
                Ok(batch)
            };
            let mapper = Box::new(mapper);

            let read_columns = Some(read_schema.field_names().into_iter().cloned().collect());
            (
                BatchUDF {
                    mapper,
                    output_schema,
                    result_checkpoint: None,
                },
                read_columns,
            )
        }
    };

    {
        let new_names = output_schema.field_names();
        for field in &dataset.schema().fields {
            if new_names.contains(&&field.name) {
                return Err(Error::invalid_input(
                    format!("Column {} already exists in the dataset", field.name),
                    location!(),
                ));
            }
        }
    }

    let mut schema = dataset.schema().merge(output_schema.as_ref())?;
    schema.set_field_id(Some(dataset.manifest.max_field_id()));

    let fragments =
        add_columns_impl(dataset, read_columns, mapper, result_checkpoint, None).await?;
    let operation = Operation::Merge { fragments, schema };
    let transaction = Transaction::new(dataset.manifest.version, operation, None);
    let new_manifest = commit_transaction(
        dataset,
        &dataset.object_store,
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
    )
    .await?;

    dataset.manifest = Arc::new(new_manifest);

    Ok(())
}

#[allow(clippy::type_complexity)]
async fn add_columns_impl(
    dataset: &Dataset,
    read_columns: Option<Vec<String>>,
    mapper: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>,
    result_cache: Option<Arc<dyn UDFCheckpointStore>>,
    schemas: Option<(Schema, Schema)>,
) -> Result<Vec<Fragment>> {
    let read_columns_ref = read_columns.as_deref();
    let mapper_ref = mapper.as_ref();
    let fragments = futures::stream::iter(dataset.get_fragments())
        .then(|fragment| {
            let cache_ref = result_cache.clone();
            let schemas_ref = &schemas;
            async move {
                if let Some(cache) = &cache_ref {
                    let fragment_id = fragment.id() as u32;
                    let fragment = cache.get_fragment(fragment_id)?;
                    if let Some(fragment) = fragment {
                        return Ok(fragment);
                    }
                }

                let mut updater = fragment
                    .updater(read_columns_ref, schemas_ref.clone())
                    .await?;

                let mut batch_index = 0;
                // TODO: the structure of the updater prevents batch-level parallelism here,
                //       but there is no reason why we couldn't do this in parallel.
                while let Some(batch) = updater.next().await? {
                    let batch_info = BatchInfo {
                        fragment_id: fragment.id() as u32,
                        batch_index,
                    };

                    let new_batch = if let Some(cache) = &cache_ref {
                        if let Some(batch) = cache.get_batch(&batch_info)? {
                            batch
                        } else {
                            let new_batch = mapper_ref(batch)?;
                            cache.insert_batch(batch_info, new_batch.clone())?;
                            new_batch
                        }
                    } else {
                        mapper_ref(batch)?
                    };

                    updater.update(new_batch).await?;
                    batch_index += 1;
                }

                let fragment = updater.finish().await?;

                if let Some(cache) = &cache_ref {
                    cache.insert_fragment(fragment.clone())?;
                }

                Ok::<_, Error>(fragment)
            }
        })
        .try_collect::<Vec<_>>()
        .await?;
    Ok(fragments)
}

/// Modify columns in the dataset, changing their name, type, or nullability.
///
/// If a column has an index, it's index will be preserved.
pub(super) async fn alter_columns(
    dataset: &mut Dataset,
    alterations: &[ColumnAlteration],
) -> Result<()> {
    // Validate we aren't making nullable columns non-nullable and that all
    // the referenced columns actually exist.
    let mut new_schema = dataset.schema().clone();

    // Mapping of old to new fields that need to be casted.
    let mut cast_fields: Vec<(Field, Field)> = Vec::new();

    let mut next_field_id = dataset.manifest.max_field_id() + 1;

    for alteration in alterations {
        let field_src = dataset.schema().field(&alteration.path).ok_or_else(|| {
            Error::invalid_input(
                format!(
                    "Column \"{}\" does not exist in the dataset",
                    alteration.path
                ),
                location!(),
            )
        })?;
        if let Some(nullable) = alteration.nullable {
            // TODO: in the future, we could check the values of the column to see if
            //       they are all non-null and thus the column could be made non-nullable.
            if field_src.nullable && !nullable {
                return Err(Error::invalid_input(
                    format!(
                        "Column \"{}\" is already nullable and thus cannot be made non-nullable",
                        alteration.path
                    ),
                    location!(),
                ));
            }
        }

        let field_dest = new_schema.mut_field_by_id(field_src.id).unwrap();
        if let Some(rename) = &alteration.rename {
            field_dest.name.clone_from(rename);
        }
        if let Some(nullable) = alteration.nullable {
            field_dest.nullable = nullable;
        }

        if let Some(data_type) = &alteration.data_type {
            if !(lance_arrow::cast::can_cast_types(&field_src.data_type(), data_type)
                && is_upcast_downcast(&field_src.data_type(), data_type))
            {
                return Err(Error::invalid_input(
                    format!(
                        "Cannot cast column \"{}\" from {:?} to {:?}",
                        alteration.path,
                        field_src.data_type(),
                        data_type
                    ),
                    location!(),
                ));
            }

            let arrow_field = ArrowField::new(
                field_dest.name.clone(),
                data_type.clone(),
                field_dest.nullable,
            );
            *field_dest = Field::try_from(&arrow_field)?;
            field_dest.set_id(field_src.parent_id, &mut next_field_id);

            cast_fields.push((field_src.clone(), field_dest.clone()));
        }
    }

    new_schema.validate()?;

    // If we aren't casting a column, we don't need to touch the fragments.
    let transaction = if cast_fields.is_empty() {
        Transaction::new(
            dataset.manifest.version,
            Operation::Project { schema: new_schema },
            None,
        )
    } else {
        // Otherwise, we need to re-write the relevant fields.
        let read_columns = cast_fields
            .iter()
            .map(|(old, _new)| {
                let parts = dataset.schema().field_ancestry_by_id(old.id).unwrap();
                let part_names = parts.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
                part_names.join(".")
            })
            .collect::<Vec<_>>();

        let new_ids = cast_fields
            .iter()
            .map(|(_old, new)| new.id)
            .collect::<Vec<_>>();
        // This schema contains the exact field ids we want to write the new fields with.
        let new_col_schema = new_schema.project_by_ids(&new_ids);

        let mapper = move |batch: &RecordBatch| {
            let mut fields = Vec::with_capacity(cast_fields.len());
            let mut columns = Vec::with_capacity(batch.num_columns());
            for (old, new) in &cast_fields {
                let old_column = batch[&old.name].clone();
                let new_column = lance_arrow::cast::cast_with_options(
                    &old_column,
                    &new.data_type(),
                    // Safe: false means it will error if the cast is lossy.
                    &CastOptions {
                        safe: false,
                        ..Default::default()
                    },
                )?;
                columns.push(new_column);
                fields.push(Arc::new(ArrowField::from(new)));
            }
            let schema = Arc::new(ArrowSchema::new(fields));
            Ok(RecordBatch::try_new(schema, columns)?)
        };
        let mapper = Box::new(mapper);

        let fragments = add_columns_impl(
            dataset,
            Some(read_columns),
            mapper,
            None,
            Some((new_col_schema, new_schema.clone())),
        )
        .await?;

        // Some data files may no longer contain any columns in the dataset (e.g. if every
        // remaining column has been altered into a different data file) and so we remove them
        let schema_field_ids = new_schema.field_ids().into_iter().collect::<Vec<_>>();
        let fragments = fragments
            .into_iter()
            .map(|mut frag| {
                frag.files.retain(|f| {
                    f.fields
                        .iter()
                        .any(|field| schema_field_ids.contains(field))
                });
                frag
            })
            .collect::<Vec<_>>();

        Transaction::new(
            dataset.manifest.version,
            Operation::Merge {
                schema: new_schema,
                fragments,
            },
            None,
        )
    };

    // TODO: adjust the indices here for the new schema

    let manifest = commit_transaction(
        dataset,
        &dataset.object_store,
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
    )
    .await?;

    dataset.manifest = Arc::new(manifest);

    Ok(())
}

/// Remove columns from the dataset.
///
/// This is a metadata-only operation and does not remove the data from the
/// underlying storage. In order to remove the data, you must subsequently
/// call `compact_files` to rewrite the data without the removed columns and
/// then call `cleanup_files` to remove the old files.
pub(super) async fn drop_columns(dataset: &mut Dataset, columns: &[&str]) -> Result<()> {
    // Check if columns are present in the dataset and construct the new schema.
    for col in columns {
        if dataset.schema().field(col).is_none() {
            return Err(Error::invalid_input(
                format!("Column {} does not exist in the dataset", col),
                location!(),
            ));
        }
    }

    let columns_to_remove = dataset.manifest.schema.project(columns)?;
    let new_schema = dataset.manifest.schema.exclude(columns_to_remove)?;

    if new_schema.fields.is_empty() {
        return Err(Error::invalid_input(
            "Cannot drop all columns from a dataset",
            location!(),
        ));
    }

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::Project { schema: new_schema },
        None,
    );

    let manifest = commit_transaction(
        dataset,
        &dataset.object_store,
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
    )
    .await?;

    dataset.manifest = Arc::new(manifest);

    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::Mutex;

    use crate::dataset::WriteParams;

    use super::*;
    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::Fields as ArrowFields;
    use rstest::rstest;

    // Used to validate that futures returned are Send.
    fn require_send<T: Send>(t: T) -> T {
        t
    }

    #[tokio::test]
    async fn test_append_columns_exprs() -> Result<()> {
        let num_rows = 5;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                use_experimental_writer: false,
                ..Default::default()
            }),
        )
        .await?;
        dataset.validate().await?;

        // Adding a duplicate column name will break
        let fut = dataset.add_columns(
            NewColumnTransform::SqlExpressions(vec![("id".into(), "id + 1".into())]),
            None,
        );
        // (Quick validation that the future is Send)
        let res = require_send(fut).await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Can add a column that is independent of any existing ones
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("value".into(), "2 * random()".into())]),
                None,
            )
            .await?;

        // Can add a column derived from an existing one.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("double_id".into(), "2 * id".into())]),
                None,
            )
            .await?;

        // Can derive a column from existing ones across multiple data files.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "triple_id".into(),
                    "id + double_id".into(),
                )]),
                None,
            )
            .await?;

        // These can be read back, the dataset is valid
        dataset.validate().await?;

        let data = dataset.scan().try_into_batch().await?;
        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Float64, true),
            ArrowField::new("double_id", DataType::Int32, false),
            ArrowField::new("triple_id", DataType::Int32, false),
        ]);
        assert_eq!(data.schema().as_ref(), &expected_schema);
        assert_eq!(data.num_rows(), num_rows);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_append_columns_udf(
        #[values(false, true)] use_experimental_writer: bool,
    ) -> Result<()> {
        use arrow_array::Float64Array;

        let num_rows = 5;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                use_experimental_writer,
                ..Default::default()
            }),
        )
        .await?;
        dataset.validate().await?;

        // Adding a duplicate column name will break
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(|_| unimplemented!()),
            output_schema: Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                DataType::Int32,
                false,
            )])),
            result_checkpoint: None,
        });
        let res = dataset.add_columns(transforms, None).await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Can add a column that independent (empty read_schema)
        let output_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let output_schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            Ok(RecordBatch::try_new(
                output_schema_ref.clone(),
                vec![Arc::new(Float64Array::from_iter_values(
                    (0..batch.num_rows()).map(|i| i as f64),
                ))],
            )?)
        };
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: None,
        });
        dataset.add_columns(transforms, None).await?;

        // Can add a column that depends on another column (double id)
        let output_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "double_id",
            DataType::Int32,
            false,
        )]));
        let output_schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            Ok(RecordBatch::try_new(
                output_schema_ref.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    id.values().iter().map(|i| i * 2),
                ))],
            )?)
        };
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: None,
        });
        dataset.add_columns(transforms, None).await?;
        // These can be read back, the dataset is valid
        dataset.validate().await?;

        let data = dataset.scan().try_into_batch().await?;
        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Float64, true),
            ArrowField::new("double_id", DataType::Int32, false),
        ]);
        assert_eq!(data.schema().as_ref(), &expected_schema);
        assert_eq!(data.num_rows(), num_rows);

        Ok(())
    }

    #[tokio::test]
    async fn test_append_columns_udf_cache() -> Result<()> {
        let num_rows = 100;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 50,
                max_rows_per_group: 25,
                use_experimental_writer: false,
                ..Default::default()
            }),
        )
        .await?;
        dataset.validate().await?;

        #[derive(Default)]
        struct RequestCounter {
            pub get_batch_requests: Mutex<Vec<BatchInfo>>,
            pub insert_batch_requests: Mutex<Vec<BatchInfo>>,
            pub get_fragment_requests: Mutex<Vec<u32>>,
            pub insert_fragment_requests: Mutex<Vec<u32>>,
        }

        impl UDFCheckpointStore for RequestCounter {
            fn get_batch(&self, info: &BatchInfo) -> Result<Option<RecordBatch>> {
                self.get_batch_requests.lock().unwrap().push(info.clone());

                if info.fragment_id == 1 && info.batch_index == 0 {
                    Ok(Some(RecordBatch::try_new(
                        Arc::new(ArrowSchema::new(vec![ArrowField::new(
                            "double_id",
                            DataType::Int32,
                            false,
                        )])),
                        vec![Arc::new(Int32Array::from_iter_values(50..75))],
                    )?))
                } else {
                    Ok(None)
                }
            }

            fn insert_batch(&self, info: BatchInfo, _value: RecordBatch) -> Result<()> {
                self.insert_batch_requests.lock().unwrap().push(info);
                Ok(())
            }

            fn get_fragment(&self, fragment_id: u32) -> Result<Option<Fragment>> {
                self.get_fragment_requests.lock().unwrap().push(fragment_id);
                if fragment_id == 0 {
                    Ok(Some(Fragment {
                        files: vec![],
                        id: 0,
                        deletion_file: None,
                        row_id_meta: None,
                        physical_rows: Some(50),
                    }))
                } else {
                    Ok(None)
                }
            }

            fn insert_fragment(&self, fragment: Fragment) -> Result<()> {
                self.insert_fragment_requests
                    .lock()
                    .unwrap()
                    .push(fragment.id as u32);
                Ok(())
            }
        }

        let request_counter = Arc::new(RequestCounter::default());

        let output_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "double_id",
            DataType::Int32,
            false,
        )]));
        let output_schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            Ok(RecordBatch::try_new(
                output_schema_ref.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    id.values().iter().map(|i| i * 2),
                ))],
            )?)
        };
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: Some(request_counter.clone()),
        });
        dataset.add_columns(transforms, None).await?;

        // Should have requested both fragments
        assert_eq!(
            request_counter
                .get_fragment_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[0, 1]
        );
        // Should have only inserted the second fragment, since the first one was already cached
        assert_eq!(
            request_counter
                .insert_fragment_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[1]
        );

        // Should have only requested the second two batches, since the first fragment was already cached
        assert_eq!(
            request_counter
                .get_batch_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[
                BatchInfo {
                    fragment_id: 1,
                    batch_index: 0,
                },
                BatchInfo {
                    fragment_id: 1,
                    batch_index: 1,
                },
            ]
        );
        // Should have only saved the last batch, since the first batch of second fragment was already cached
        assert_eq!(
            request_counter
                .insert_batch_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[BatchInfo {
                fragment_id: 1,
                batch_index: 1,
            },]
        );

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_rename_columns(
        #[values(false, true)] use_experimental_writer: bool,
    ) -> Result<()> {
        use std::collections::HashMap;

        use arrow_array::{ArrayRef, StructArray};

        let metadata: HashMap<String, String> = [("k1".into(), "v1".into())].into();

        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("a", DataType::Int32, false),
                ArrowField::new(
                    "b",
                    DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                        "c",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
            ],
            metadata.clone(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::from(vec![(
                    Arc::new(ArrowField::new("c", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                )])),
            ],
        )?;

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                use_experimental_writer,
                ..Default::default()
            }),
        )
        .await?;

        let original_fragments = dataset.fragments().to_vec();

        // Rename a top-level column
        dataset
            .alter_columns(&[ColumnAlteration::new("a".into())
                .rename("x".into())
                .set_nullable(true)])
            .await?;
        dataset.validate().await?;
        assert_eq!(dataset.manifest.version, 2);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        let expected_schema = ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("x", DataType::Int32, true),
                ArrowField::new(
                    "b",
                    DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                        "c",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
            ],
            metadata.clone(),
        );
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // Rename to duplicate name fails
        let err = dataset
            .alter_columns(&[ColumnAlteration::new("b".into()).rename("x".into())])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Duplicate field name \"x\""));

        // Rename a nested column.
        dataset
            .alter_columns(&[ColumnAlteration::new("b.c".into()).rename("d".into())])
            .await?;
        dataset.validate().await?;
        assert_eq!(dataset.manifest.version, 3);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        let expected_schema = ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("x", DataType::Int32, true),
                ArrowField::new(
                    "b",
                    DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                        "d",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
            ],
            metadata.clone(),
        );
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_cast_column(#[values(false, true)] use_experimental_writer: bool) -> Result<()> {
        // Create a table with 2 scalar columns, 1 vector column

        use arrow::datatypes::{Int32Type, Int64Type};
        use arrow_array::{Float16Array, Float32Array, Int64Array, ListArray};
        use half::f16;
        use lance_arrow::FixedSizeListArrayExt;
        use lance_index::{DatasetIndexExt, IndexType};
        use lance_linalg::distance::MetricType;
        use lance_testing::datagen::generate_random_array;

        use crate::index::{scalar::ScalarIndexParams, vector::VectorIndexParams};
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("f", DataType::Float32, false),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                false,
            ),
            ArrowField::new("l", DataType::new_list(DataType::Int32, true), true),
        ]));

        let nrows = 512;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..nrows)),
                Arc::new(Float32Array::from_iter_values((0..nrows).map(|i| i as f32))),
                Arc::new(
                    <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                        generate_random_array(128 * nrows as usize),
                        128,
                    )
                    .unwrap(),
                ),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                    (0..nrows).map(|i| Some(vec![Some(i), Some(i + 1)])),
                )),
            ],
        )?;

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
            test_uri,
            Some(WriteParams {
                use_experimental_writer,
                ..Default::default()
            }),
        )
        .await?;

        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &params, false)
            .await?;
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await?;
        dataset.validate().await?;

        let indices = dataset.load_indices().await?;
        assert_eq!(indices.len(), 2);

        // Cast a scalar column to another type, nullability
        dataset
            .alter_columns(&[ColumnAlteration::new("f".into())
                .cast_to(DataType::Float16)
                .set_nullable(true)])
            .await?;
        dataset.validate().await?;
        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("f", DataType::Float16, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                false,
            ),
            ArrowField::new("l", DataType::new_list(DataType::Int32, true), true),
        ]);
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // Each fragment gains a file with the new columns
        dataset.fragments().iter().for_each(|f| {
            assert_eq!(f.files.len(), 2);
        });

        // Cast scalar column with index, should not keep index (TODO: keep it)
        dataset
            .alter_columns(&[ColumnAlteration::new("i".into()).cast_to(DataType::Int64)])
            .await?;
        dataset.validate().await?;

        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, false),
            ArrowField::new("f", DataType::Float16, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                false,
            ),
            ArrowField::new("l", DataType::new_list(DataType::Int32, true), true),
        ]);
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // We currently lose the index when casting a column
        let indices = dataset.load_indices().await?;
        assert_eq!(indices.len(), 1);

        // Each fragment gains a file with the new columns
        dataset.fragments().iter().for_each(|f| {
            assert_eq!(f.files.len(), 3);
        });

        // Cast vector column, should not keep index (TODO: keep it)
        dataset
            .alter_columns(&[
                ColumnAlteration::new("vec".into()).cast_to(DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float16, true)),
                    128,
                )),
            ])
            .await?;
        dataset.validate().await?;

        // Finally, case list column to show we can handle children.
        dataset
            .alter_columns(&[ColumnAlteration::new("l".into())
                .cast_to(DataType::new_list(DataType::Int64, true))])
            .await?;
        dataset.validate().await?;

        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, false),
            ArrowField::new("f", DataType::Float16, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float16, true)),
                    128,
                ),
                false,
            ),
            ArrowField::new("l", DataType::new_list(DataType::Int64, true), true),
        ]);
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // We currently lose the index when casting a column
        let indices = dataset.load_indices().await?;
        assert_eq!(indices.len(), 0);

        // Each fragment gains a file with the new columns, but then the original file is dropped
        dataset.fragments().iter().for_each(|f| {
            assert_eq!(f.files.len(), 4);
        });

        let expected_data = RecordBatch::try_new(
            Arc::new(expected_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..nrows as i64)),
                Arc::new(Float16Array::from_iter_values(
                    (0..nrows).map(|i| f16::from_f32(i as f32)),
                )),
                lance_arrow::cast::cast_with_options(
                    batch["vec"].as_ref(),
                    &DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float16, true)),
                        128,
                    ),
                    &Default::default(),
                )?,
                Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                    (0..nrows as i64).map(|i| Some(vec![Some(i), Some(i + 1)])),
                )),
            ],
        )?;
        let actual_data = dataset.scan().try_into_batch().await?;
        assert_eq!(actual_data, expected_data);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_drop_columns(#[values(false, true)] use_experimental_writer: bool) -> Result<()> {
        use std::collections::HashMap;

        use arrow_array::{ArrayRef, Float32Array, StructArray};

        let metadata: HashMap<String, String> = [("k1".into(), "v1".into())].into();

        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("i", DataType::Int32, false),
                ArrowField::new(
                    "s",
                    DataType::Struct(ArrowFields::from(vec![
                        ArrowField::new("d", DataType::Int32, true),
                        ArrowField::new("l", DataType::Int32, true),
                    ])),
                    true,
                ),
                ArrowField::new("x", DataType::Float32, false),
            ],
            metadata.clone(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("d", DataType::Int32, true)),
                        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("l", DataType::Int32, true)),
                        Arc::new(Int32Array::from(vec![1, 2])),
                    ),
                ])),
                Arc::new(Float32Array::from(vec![1.0, 2.0])),
            ],
        )?;

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                use_experimental_writer,
                ..Default::default()
            }),
        )
        .await?;

        let lance_schema = dataset.schema().clone();
        let original_fragments = dataset.fragments().to_vec();

        dataset.drop_columns(&["x"]).await?;
        dataset.validate().await?;

        let expected_schema = lance_schema.project(&["i", "s"])?;
        assert_eq!(dataset.schema(), &expected_schema);

        assert_eq!(dataset.version().version, 2);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        dataset.drop_columns(&["s.d"]).await?;
        dataset.validate().await?;

        let expected_schema = expected_schema.project(&["i", "s.l"])?;
        assert_eq!(dataset.schema(), &expected_schema);

        let expected_data = RecordBatch::try_new(
            Arc::new(ArrowSchema::from(&expected_schema)),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::from(vec![(
                    Arc::new(ArrowField::new("l", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                )])),
            ],
        )?;
        let actual_data = dataset.scan().try_into_batch().await?;
        assert_eq!(actual_data, expected_data);

        assert_eq!(dataset.version().version, 3);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_drop_add_columns(
        #[values(false, true)] use_experimental_writer: bool,
    ) -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))])?;

        let test_dir = tempfile::tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                use_experimental_writer,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(dataset.manifest.max_field_id(), 0);

        // Test we can add 1 column, drop it, then add another column. Validate
        // the field ids are as expected.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("x".into(), "i + 1".into())]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 1);

        dataset.drop_columns(&["x"]).await?;
        assert_eq!(dataset.manifest.max_field_id(), 0);

        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("y".into(), "2 * i".into())]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 1);

        let data = dataset.scan().try_into_batch().await?;
        let expected_data = RecordBatch::try_new(
            Arc::new(schema.try_with_column(ArrowField::new("y", DataType::Int32, false))?),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![2, 4])),
            ],
        )?;
        assert_eq!(data, expected_data);
        dataset.drop_columns(&["y"]).await?;
        assert_eq!(dataset.manifest.max_field_id(), 0);

        // Test we can add 2 columns, drop 1, then add another column. Validate
        // the field ids are as expected.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![
                    ("a".into(), "i + 3".into()),
                    ("b".into(), "i + 7".into()),
                ]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 2);

        dataset.drop_columns(&["b"]).await?;
        // Even though we dropped a column, we still have the fragment with a and
        // b. So it should still act as if that field id is still in play.
        assert_eq!(dataset.manifest.max_field_id(), 2);

        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("c".into(), "i + 11".into())]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 3);

        let data = dataset.scan().try_into_batch().await?;
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, false),
        ]));
        let expected_data = RecordBatch::try_new(
            expected_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(Int32Array::from(vec![12, 13])),
            ],
        )?;
        assert_eq!(data, expected_data);

        Ok(())
    }
}
