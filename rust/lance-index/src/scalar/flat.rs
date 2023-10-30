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

use std::{ops::Bound, sync::Arc};

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;

use datafusion_physical_expr::expressions::{in_list, lit, Column};
use lance_core::Result;

use super::{
    btree::{SubIndexLoader, SubIndexTrainer},
    IndexReader, IndexStore, IndexWriter, ScalarIndex, ScalarQuery,
};

/// A flat index is just a batch of value/row-id pairs
///
/// The batch always has two columns.  The first column "values" contains
/// the values.  The second column "row_ids" contains the row ids
///
/// Evaluating a query requires O(N) time where N is the # of rows
#[derive(Debug)]
pub struct FlatIndex {
    data: Arc<RecordBatch>,
}

impl FlatIndex {
    fn values(&self) -> &ArrayRef {
        self.data.column(0)
    }

    fn ids(&self) -> &ArrayRef {
        self.data.column(1)
    }
}

/// Trains a flat index from a record batch of values & ids by simply storing the batch
///
/// This allows the flat index to be used as a sub-index
pub struct FlatIndexTrainer {
    schema: Arc<Schema>,
}

impl FlatIndexTrainer {
    pub fn new(value_type: DataType) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", value_type, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        Self { schema }
    }
}

/// Loads a flat index from a serialized record batch by simply loading the batch
///
/// This allows the flat index to be used as a sub-index
#[derive(Debug)]
pub struct FlatIndexLoader {}

#[async_trait]
impl SubIndexLoader for FlatIndexLoader {
    async fn load_subindex(
        &self,
        offset: u64,
        index_reader: Arc<dyn IndexReader>,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let batch = index_reader.read_record_batch(offset).await?;
        Ok(Arc::new(FlatIndex {
            data: Arc::new(batch),
        }))
    }
}

#[async_trait]
impl SubIndexTrainer for FlatIndexTrainer {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    async fn train(&self, batch: RecordBatch, writer: &mut dyn IndexWriter) -> Result<u64> {
        writer.write_record_batch(batch).await
    }
}

#[async_trait]
impl ScalarIndex for FlatIndex {
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array> {
        // Since we have all the values in memory we can use basic arrow-rs compute
        // functions to satisfy scalar queries.
        let predicate = match query {
            ScalarQuery::Equals(value) => arrow_ord::cmp::eq(self.values(), &value.to_scalar())?,
            ScalarQuery::IsNull() => arrow::compute::is_null(self.values())?,
            ScalarQuery::IsIn(values) => {
                let choices = values
                    .iter()
                    .map(|val| lit(val.clone()))
                    .collect::<Vec<_>>();
                let in_list_expr = in_list(
                    Arc::new(Column::new("values", 0)),
                    choices,
                    &false,
                    &self.data.schema(),
                )?;
                let result_col = in_list_expr.evaluate(&self.data)?;
                result_col
                    .into_array(self.data.num_rows())
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("InList evaluation should return boolean array")
                    .clone()
            }
            ScalarQuery::Range(lower_bound, upper_bound) => match (lower_bound, upper_bound) {
                (Bound::Unbounded, Bound::Unbounded) => {
                    panic!("Scalar range query received with no upper or lower bound")
                }
                (Bound::Included(lower), Bound::Unbounded) => {
                    arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar())?
                }
                (Bound::Unbounded, Bound::Excluded(upper)) => {
                    arrow_ord::cmp::lt(self.values(), &upper.to_scalar())?
                }
                (Bound::Included(lower), Bound::Excluded(upper)) => arrow::compute::and(
                    &arrow_ord::cmp::gt_eq(self.values(), &lower.to_scalar())?,
                    &arrow_ord::cmp::lt(self.values(), &upper.to_scalar())?,
                )?,
                _ => panic!("Scalar range query had excluded lower bound or included upper bound"),
            },
        };
        Ok(arrow_select::filter::filter(self.ids(), &predicate)?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type")
            .clone())
    }

    // Note that there is no write/train method for flat index at the moment and so it isn't
    // really possible for this method to be called.  If there was we assume it will write all
    // data as a single batch named data.lance
    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        let batches = store.open_index_file("data.lance").await?;
        let batch = batches.read_record_batch(0).await?;
        Ok(Arc::new(Self {
            data: Arc::new(batch),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int32Type;
    use arrow_array::types::UInt64Type;
    use datafusion_common::ScalarValue;
    use lance_datagen::{array, gen, RowCount};

    fn example_index() -> FlatIndex {
        let batch = gen()
            .col(
                Some("values".to_string()),
                array::cycle::<Int32Type>(vec![10, 100, 1000, 1234]),
            )
            .col(
                Some("ids".to_string()),
                array::cycle::<UInt64Type>(vec![5, 0, 3, 100]),
            )
            .into_batch_rows(RowCount::from(4))
            .unwrap();

        FlatIndex {
            data: Arc::new(batch),
        }
    }

    async fn check_index(query: &ScalarQuery, expected: &[u64]) {
        let index = example_index();
        let actual = index.search(query).await.unwrap();
        let expected = UInt64Array::from_iter_values(expected.iter().copied());
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_equality() {
        check_index(&ScalarQuery::Equals(ScalarValue::from(100)), &[0]).await;
        check_index(&ScalarQuery::Equals(ScalarValue::from(10)), &[5]).await;
        check_index(&ScalarQuery::Equals(ScalarValue::from(5)), &[]).await;
    }

    #[tokio::test]
    async fn test_range() {
        check_index(
            &ScalarQuery::Range(
                Bound::Included(ScalarValue::from(100)),
                Bound::Excluded(ScalarValue::from(1234)),
            ),
            &[0, 3],
        )
        .await;
        check_index(
            &ScalarQuery::Range(Bound::Unbounded, Bound::Excluded(ScalarValue::from(1000))),
            &[5, 0],
        )
        .await;
        check_index(
            &ScalarQuery::Range(Bound::Included(ScalarValue::from(0)), Bound::Unbounded),
            &[5, 0, 3, 100],
        )
        .await;
        check_index(
            &ScalarQuery::Range(Bound::Included(ScalarValue::from(100000)), Bound::Unbounded),
            &[],
        )
        .await;
    }

    #[tokio::test]
    async fn test_is_in() {
        check_index(
            &ScalarQuery::IsIn(vec![
                ScalarValue::from(100),
                ScalarValue::from(1234),
                ScalarValue::from(3000),
            ]),
            &[0, 100],
        )
        .await;
    }
}
