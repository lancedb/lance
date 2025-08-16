// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::fragment::session::FragmentSession;
use crate::dataset::take::{
    do_take_rows, row_offsets_to_row_addresses, TakeFunction, TakeRangeFunction,
};
use crate::dataset::{ProjectionRequest, TakeBuilder};
use crate::Dataset;
use arrow_array::RecordBatch;
use futures::FutureExt;
use lance_core::Result;
use lance_datafusion::projection::ProjectionPlan;
use std::collections::HashMap;
use std::sync::Arc;

/// A [`DatasetReader`] manages a short-lived context of [`Dataset`], allowing us to maintain
/// internal states instead of creating new ones each time.
///
/// This API works well for users making repeated requests over the same projection.
pub struct DatasetReader {
    dataset: Arc<Dataset>,
    fragment_sessions: Arc<HashMap<usize, FragmentSession>>,
    fragment_sessions_with_row_addr: Arc<HashMap<usize, FragmentSession>>,
    projection: Arc<ProjectionPlan>,
}

impl DatasetReader {
    pub async fn new(dataset: &Dataset, projection: impl Into<ProjectionRequest>) -> Result<Self> {
        let dataset = Arc::new(dataset.clone());
        let projection = Arc::new(projection.into().into_projection_plan(dataset.clone())?);
        let schema = projection.physical_projection.to_bare_schema();

        let mut fragment_sessions = HashMap::new();
        let mut fragment_sessions_with_row_addr = HashMap::new();

        for ff in dataset.get_fragments().iter() {
            let session = ff.open_session(&schema, false).await?;
            fragment_sessions.insert(ff.id(), session);

            let session = ff.open_session(&schema, true).await?;
            fragment_sessions_with_row_addr.insert(ff.id(), session);
        }

        Ok(Self {
            dataset,
            fragment_sessions: Arc::new(fragment_sessions),
            fragment_sessions_with_row_addr: Arc::new(fragment_sessions_with_row_addr),
            projection,
        })
    }

    /// Take rows by indices.
    pub async fn take(&self, row_indices: &[u64]) -> Result<RecordBatch> {
        if row_indices.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(
                self.projection.output_schema()?,
            )));
        }

        // convert the dataset offsets into row addresses
        let addrs = row_offsets_to_row_addresses(&self.dataset, row_indices).await?;

        let builder = TakeBuilder::try_new_from_addresses(
            self.dataset.clone(),
            addrs.to_vec(),
            self.projection.clone(),
        )?;
        do_take_rows(
            builder,
            self.projection.clone(),
            self.do_take(),
            self.do_take_range(),
        )
        .await
    }

    /// Take rows by the internal row ids.
    pub async fn take_rows(&self, row_ids: &[u64]) -> Result<RecordBatch> {
        let builder = TakeBuilder::try_new_from_ids(
            self.dataset.clone(),
            row_ids.to_vec(),
            self.projection.clone(),
        )?;
        do_take_rows(
            builder,
            self.projection.clone(),
            self.do_take(),
            self.do_take_range(),
        )
        .await
    }

    fn do_take(&self) -> TakeFunction {
        let fragment_sessions = self.fragment_sessions.clone();
        let fragment_sessions_with_row_addr = self.fragment_sessions_with_row_addr.clone();

        Box::new(move |fragment, row_offsets, _, with_row_addresses| {
            let fragment_sessions = fragment_sessions.clone();
            let fragment_sessions_with_row_addr = fragment_sessions_with_row_addr.clone();

            async move {
                let session = match with_row_addresses {
                    true => fragment_sessions_with_row_addr.get(&fragment.id()).unwrap(),
                    false => fragment_sessions.get(&fragment.id()).unwrap(),
                };

                session.take_rows(&row_offsets).await
            }
            .boxed()
        })
    }

    fn do_take_range(&self) -> TakeRangeFunction {
        let fragment_sessions = self.fragment_sessions.clone();

        Box::new(move |fragment, range, _| {
            let fragment_sessions = fragment_sessions.clone();

            async move {
                let session = fragment_sessions.get(&fragment.id()).unwrap();
                session.take_range(range).await
            }
            .boxed()
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::scanner::test_dataset::TestVectorDataset;
    use crate::dataset::{ProjectionRequest, WriteParams};
    use crate::Dataset;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_encoding::version::LanceFileVersion;
    use rstest::rstest;
    use std::ops::Range;
    use std::sync::Arc;

    fn test_batch(i_range: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("s", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(i_range.clone())),
                Arc::new(StringArray::from_iter_values(
                    i_range.map(|i| format!("str-{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_reader_take(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_move_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            enable_move_stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        pretty_assertions::assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let reader = dataset.open_reader(projection).await.unwrap();
        let values = reader
            .take(&[
                200, // 200
                199, // 199
                39,  // 39
                40,  // 40
                199, // 199
                40,  // 40
                125, // 125
            ])
            .await
            .unwrap();

        pretty_assertions::assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([
                        200, 199, 39, 40, 199, 40, 125
                    ])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 199, 40, 125]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        for i in 50..60 {
            let values = reader.take(&[i]).await.unwrap();

            pretty_assertions::assert_eq!(
                RecordBatch::try_new(
                    data.schema(),
                    vec![
                        Arc::new(Int32Array::from_iter_values([i as i32])),
                        Arc::new(StringArray::from_iter_values([format!("str-{i}")])),
                    ],
                )
                .unwrap(),
                values
            );
        }
    }

    #[tokio::test]
    async fn test_reader_take_with_deletion() {
        let data = test_batch(0..120);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (40, 77, 78, 79)").await.unwrap();

        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let reader = dataset.open_reader(projection).await.unwrap();
        let values = reader
            .take(&[
                0,   // 0
                39,  // 39
                40,  // 41
                75,  // 76
                76,  // 80
                77,  // 81
                115, // 119
            ])
            .await
            .unwrap();

        pretty_assertions::assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([0, 39, 41, 76, 80, 81, 119])),
                    Arc::new(StringArray::from_iter_values(
                        [0, 39, 41, 76, 80, 81, 119]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_reader_take_with_projection(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_move_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_move_stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        pretty_assertions::assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![("foo", "i"), ("bar", "i*2")]);
        let reader = dataset.open_reader(projection).await.unwrap();
        let values = reader.take(&[10, 50, 100]).await.unwrap();

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
        ]));
        pretty_assertions::assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = reader.take_rows(&[10, 50, 100]).await.unwrap();
        pretty_assertions::assert_eq!(values, values2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_reader_take_rows_out_of_bound(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // a dataset with 1 fragment and 400 rows
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let ds = test_ds.dataset;
        let reader = ds.open_reader(ds.schema().clone()).await.unwrap();

        // take the last row of first fragment
        // this triggers the contiguous branch
        let indices = &[(1 << 32) - 1];

        fn require_send<T: Send>(t: T) -> T {
            t
        }
        let fut = require_send(reader.take_rows(indices));
        let err = fut.await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid read params"),
            "{}",
            err.to_string()
        );

        // this triggers the sorted branch, but not contiguous
        let indices = &[(1 << 32) - 3, (1 << 32) - 1];
        let err = reader.take_rows(indices).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );

        // this triggers the catch all branch
        let indices = &[(1 << 32) - 1, (1 << 32) - 3];
        let err = reader.take_rows(indices).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_reader_take_rows(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        pretty_assertions::assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let reader = dataset.open_reader(dataset.schema().clone()).await.unwrap();
        let indices = &[
            5_u64 << 32,        // 200
            (4_u64 << 32) + 39, // 199
            39,                 // 39
            1_u64 << 32,        // 40
            (2_u64 << 32) + 20, // 100
        ];
        let values = reader.take_rows(indices).await.unwrap();
        pretty_assertions::assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 199, 39, 40, 100])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 100].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Delete some rows from a fragment
        dataset.delete("i in (199, 100)").await.unwrap();
        dataset.validate().await.unwrap();
        let reader = dataset.open_reader(dataset.schema().clone()).await.unwrap();
        let values = reader.take_rows(indices).await.unwrap();
        pretty_assertions::assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 39, 40])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 39, 40].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Take an empty selection.
        let values = reader.take_rows(&[]).await.unwrap();
        pretty_assertions::assert_eq!(RecordBatch::new_empty(data.schema()), values);
    }
}
