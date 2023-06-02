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

use arrow_array::{new_null_array, Array, RecordBatch, RecordBatchReader};
use arrow_array::ArrayRef;
use arrow_row::{OwnedRow, RowConverter, Rows, SortField};
use arrow_schema::SchemaRef;
use arrow_select::interleave::interleave;
use dashmap::{DashMap, ReadOnlyView};
use futures::{StreamExt, TryStreamExt};
use tokio::task;

use crate::{Error, Result};

/// `HashJoiner` does hash join on two datasets.
pub(crate) struct HashJoiner {
    index_map: ReadOnlyView<OwnedRow, (usize, usize)>,

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
    pub async fn try_new(reader: &mut dyn RecordBatchReader, on: &str) -> Result<Self> {
        // Check column exist
        reader.schema().field_with_name(on)?;

        // Hold all data in memory for simple implementation. Can do external sort later.
        let batches = reader.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
        if batches.is_empty() {
            return Err(Error::IO {
                message: "HashJoiner: No data".to_string(),
            });
        };

        let map = DashMap::new();

        let schema = reader.schema();

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

        let map_ref = &map;

        futures::stream::iter(batches.iter().enumerate().map(Ok::<_, Error>))
            // Not sure if this can actually run in parallel though
            // TODO: use spawn_blocking instead
            .try_for_each_concurrent(num_cpus::get(), |(batch_i, batch)| async move {
                let column = batch[on].clone();
                let map_ref = map_ref.clone();
                let task_result = task::spawn(async move {
                    let rows = column_to_rows(column)?;
                    for (row_i, row) in rows.iter().enumerate() {
                        map_ref.insert(row.owned(), (batch_i, row_i));
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
            })
            .await?;

        Ok(Self {
            index_map: map.into_read_only(),
            batches: right_batches,
            out_schema,
        })
    }

    pub fn out_schema(&self) -> &SchemaRef {
        &self.out_schema
    }

    /// Collecting the data using the index column from left table.
    ///
    /// Will run in parallel over columns using all available cores.
    pub(super) async fn collect(&self, index_column: ArrayRef) -> Result<RecordBatch> {
        // Index to use for null values
        let null_index = self.batches.len();

        let index_type = index_column.data_type().clone();

        // collect indices
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

        // Use interleave to get the data
        // https://docs.rs/arrow/40.0.0/arrow/compute/kernels/interleave/fn.interleave.html
        // To handle nulls, we'll add an extra null array at the end
        let null_array = Arc::new(new_null_array(&index_type, 1));

        // Do this in parallel over the columns
        let columns = futures::stream::iter(0..self.batches[0].num_columns())
            .map(|column_i| {
                // TODO: we need to drop the join_on column

                let mut arrays = Vec::with_capacity(self.batches.len() + 1);
                for batch in &self.batches {
                    arrays.push(batch.column(column_i).clone());
                }
                arrays.push(null_array.clone());

                let indices = indices.clone();

                async move {
                    let task_result = task::spawn(async move {
                        let array_refs = arrays.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
                        interleave(array_refs.as_ref(), indices.as_ref())
                            .map_err(|err| Error::IO {
                                message: format!("HashJoiner: {}", err),
                            })
                            .map(|x| x.clone())
                    })
                    .await;
                    match task_result {
                        Ok(Ok(array)) => Ok(array),
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

        Ok(RecordBatch::try_new(
            self.batches[0].schema().clone(),
            columns,
        )?)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::sync::Arc;

    use arrow_array::{Int32Array, StringArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};

    use crate::arrow::RecordBatchBuffer;

    #[tokio::test]
    async fn test_joiner_collect() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]));

        let mut batch_buffer: RecordBatchBuffer = (0..5)
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
        let joiner = HashJoiner::try_new(&mut batch_buffer, "i").await.unwrap();

        let indices = Arc::new(UInt32Array::from_iter(&[
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
    }
}
