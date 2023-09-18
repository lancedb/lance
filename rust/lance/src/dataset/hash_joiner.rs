// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HashJoiner

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::{new_null_array, Array, RecordBatch, RecordBatchReader};
use arrow_row::{OwnedRow, RowConverter, Rows, SortField};
use arrow_schema::{DataType as ArrowDataType, SchemaRef};
use arrow_select::interleave::interleave;
use dashmap::{DashMap, ReadOnlyView};
use futures::{StreamExt, TryStreamExt};
use tokio::task;

use crate::datatypes::lance_supports_nulls;
use crate::{Error, Result};

/// `HashJoiner` does hash join on two datasets.
pub struct HashJoiner {
    index_map: ReadOnlyView<OwnedRow, (usize, usize)>,

    index_type: ArrowDataType,

    batches: Vec<RecordBatch>,

    out_schema: SchemaRef,
}

fn column_to_rows(column: ArrayRef) -> Result<Rows> {
    let mut row_converter = RowConverter::new(vec![SortField::new(column.data_type().clone())])?;
    let rows = row_converter.convert_columns(&[column])?;
    Ok(rows)
}

impl HashJoiner {
    /// Create a new `HashJoiner`, building the hash index.
    ///
    /// Will run in parallel over batches using all available cores.
    pub async fn try_new(reader: Box<dyn RecordBatchReader + Send>, on: &str) -> Result<Self> {
        // Check column exist
        let schema = reader.schema();
        schema.field_with_name(on)?;

        // Hold all data in memory for simple implementation. Can do external sort later.
        let batches = reader.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
        if batches.is_empty() {
            return Err(Error::IO {
                message: "HashJoiner: No data".to_string(),
            });
        };

        let map = DashMap::new();

        let keep_indices: Vec<usize> = schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(i, field)| if field.name() == on { None } else { Some(i) })
            .collect();
        let out_schema: Arc<arrow_schema::Schema> = Arc::new(schema.project(&keep_indices)?);
        let right_batches = batches
            .iter()
            .map(|batch| {
                let mut columns = Vec::with_capacity(keep_indices.len());
                for i in &keep_indices {
                    columns.push(batch.column(*i).clone());
                }
                RecordBatch::try_new(out_schema.clone(), columns).unwrap()
            })
            .collect::<Vec<_>>();

        let map = Arc::new(map);

        futures::stream::iter(batches.iter().enumerate().map(Ok::<_, Error>))
            .try_for_each_concurrent(num_cpus::get(), |(batch_i, batch)| {
                // A clone of map we can send to a new thread
                let map = map.clone();
                async move {
                    let column = batch[on].clone();
                    let task_result = task::spawn_blocking(move || {
                        let rows = column_to_rows(column)?;
                        for (row_i, row) in rows.iter().enumerate() {
                            map.insert(row.owned(), (batch_i, row_i));
                        }
                        Ok(())
                    })
                    .await;
                    match task_result {
                        Ok(Ok(_)) => Ok(()),
                        Ok(Err(err)) => Err(err),
                        Err(err) => Err(Error::IO {
                            message: format!("HashJoiner: {}", err),
                        }),
                    }
                }
            })
            .await?;

        let map = Arc::try_unwrap(map)
            .expect("HashJoiner: No remaining tasks should still be referencing map.");
        let index_type = batches[0]
            .schema()
            .field_with_name(on)
            .unwrap()
            .data_type()
            .clone();
        Ok(Self {
            index_map: map.into_read_only(),
            index_type,
            batches: right_batches,
            out_schema,
        })
    }

    /// Returns the schema of data yielded by `collect()`.
    ///
    /// This excludes the index column on the right-hand side.
    pub fn out_schema(&self) -> &SchemaRef {
        &self.out_schema
    }

    /// Collecting the data using the index column from left table.
    ///
    /// Will run in parallel over columns using all available cores.
    pub(super) async fn collect(&self, index_column: ArrayRef) -> Result<RecordBatch> {
        if index_column.data_type() != &self.index_type {
            return Err(Error::invalid_input(format!(
                "Index column type mismatch: expected {}, got {}",
                self.index_type,
                index_column.data_type()
            )));
        }

        // Index to use for null values
        let null_index = self.batches.len();

        // Indices are a pair of (batch_i, row_i). We'll add a null batch at the
        // end with one null element, and that's what we resolve when no match is
        // found.
        let indices = column_to_rows(index_column)?
            .into_iter()
            .map(|row| {
                self.index_map
                    .get(&row.owned())
                    .map(|(batch_i, row_i)| (*batch_i, *row_i))
                    .unwrap_or((null_index, 0))
            })
            .collect::<Vec<_>>();
        let indices = Arc::new(indices);

        // Do this in parallel over the columns
        let columns = futures::stream::iter(0..self.batches[0].num_columns())
            .map(|column_i| {
                // Use interleave to get the data
                // https://docs.rs/arrow/40.0.0/arrow/compute/kernels/interleave/fn.interleave.html
                // To handle nulls, we'll add an extra null array at the end
                let mut arrays = Vec::with_capacity(self.batches.len() + 1);
                for batch in &self.batches {
                    arrays.push(batch.column(column_i).clone());
                }
                arrays.push(Arc::new(new_null_array(arrays[0].data_type(), 1)));

                // Clone of indices we can send to a new thread
                let indices = indices.clone();

                async move {
                    let task_result = task::spawn_blocking(move || {
                        let array_refs = arrays.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                        interleave(array_refs.as_ref(), indices.as_ref())
                            .map_err(|err| Error::IO {
                                message: format!("HashJoiner: {}", err),
                            })
                    })
                    .await;
                    match task_result {
                        Ok(Ok(array)) => {
                            if array.null_count() > 0 && !lance_supports_nulls(array.data_type()) {
                                return Err(Error::invalid_input(format!(
                                    "Found rows on LHS that do not match any rows on RHS. Lance would need to write \
                                    nulls on the RHS, but Lance does not yet support nulls for type {:?}.",
                                    array.data_type()
                                )));
                            }
                            Ok(array)
                        },
                        Ok(Err(err)) => Err(err),
                        Err(err) => Err(Error::IO {
                            message: format!("HashJoiner: {}", err),
                        }),
                    }
                }
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;

        Ok(RecordBatch::try_new(self.batches[0].schema(), columns)?)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatchIterator, StringArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};

    #[tokio::test]
    async fn test_joiner_collect() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..5)
            .map(|v| {
                let values = (v * 10..v * 10 + 10).collect::<Vec<_>>();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter(values.iter().copied())),
                        Arc::new(StringArray::from_iter_values(
                            values.iter().map(|v| format!("str_{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let batches: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema.clone(),
        ));
        let joiner = HashJoiner::try_new(batches, "i").await.unwrap();

        let indices = Arc::new(Int32Array::from_iter(&[
            Some(15),
            None,
            Some(10),
            Some(0),
            None,
            None,
            Some(22),
            Some(11111), // not found
        ]));
        let results = joiner.collect(indices).await.unwrap();

        assert_eq!(
            results.column_by_name("s").unwrap().as_ref(),
            &StringArray::from(vec![
                Some("str_15"),
                None,
                Some("str_10"),
                Some("str_0"),
                None,
                None,
                Some("str_22"),
                None // 11111 not found
            ])
        );

        assert_eq!(results.num_columns(), 1);
    }

    #[tokio::test]
    async fn test_reject_invalid() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let batches: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![batch].into_iter().map(Ok),
            schema.clone(),
        ));

        let joiner = HashJoiner::try_new(batches, "i").await.unwrap();

        // Wrong type: was Int32, passing UInt32.
        let indices = Arc::new(UInt32Array::from_iter(&[Some(15)]));
        let result = joiner.collect(indices).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Index column type mismatch: expected Int32, got UInt32"));
    }
}
