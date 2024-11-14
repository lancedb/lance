// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BTreeMap, ops::Range, pin::Pin, sync::Arc};

use crate::dataset::fragment::FragReadConfig;
use crate::dataset::rowids::get_row_id_index;
use crate::{Error, Result};
use arrow::{array::as_struct_array, compute::concat_batches, datatypes::UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, StructArray, UInt64Array};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{Future, Stream, StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::ROW_ADDR;
use lance_datafusion::projection::ProjectionPlan;
use snafu::{location, Location};

use super::ProjectionRequest;
use super::{fragment::FileFragment, scanner::DatasetRecordBatchStream, Dataset};

pub async fn take(
    dataset: &Dataset,
    offsets: &[u64],
    projection: ProjectionRequest,
) -> Result<RecordBatch> {
    let projection = projection.into_projection_plan(dataset.schema())?;

    if offsets.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(
            projection.output_schema()?,
        )));
    }

    // First, convert the dataset offsets into row addresses
    let fragments = dataset.get_fragments();

    let mut perm = permutation::sort(offsets);
    let sorted_offsets = perm.apply_slice(offsets);

    let mut frag_iter = fragments.iter();
    let mut cur_frag = frag_iter.next();
    let mut cur_frag_rows = if let Some(cur_frag) = cur_frag {
        cur_frag.count_rows().await? as u64
    } else {
        0
    };
    let mut frag_offset = 0;

    let mut addrs = Vec::with_capacity(sorted_offsets.len());
    for sorted_offset in sorted_offsets.into_iter() {
        while cur_frag.is_some() && sorted_offset >= frag_offset + cur_frag_rows {
            frag_offset += cur_frag_rows;
            cur_frag = frag_iter.next();
            cur_frag_rows = if let Some(cur_frag) = cur_frag {
                cur_frag.count_rows().await? as u64
            } else {
                0
            };
        }
        let Some(cur_frag) = cur_frag else {
            addrs.push(RowAddress::TOMBSTONE_ROW);
            continue;
        };
        let row_addr =
            RowAddress::new_from_parts(cur_frag.id() as u32, (sorted_offset - frag_offset) as u32);
        addrs.push(u64::from(row_addr));
    }

    // Restore the original order
    perm.apply_inv_slice_in_place(&mut addrs);

    let builder = TakeBuilder::try_new_from_addresses(
        Arc::new(dataset.clone()),
        addrs,
        Arc::new(projection),
    )?;

    take_rows(builder).await
}

/// Take rows by the internal ROW ids.
async fn do_take_rows(
    mut builder: TakeBuilder,
    projection: Arc<ProjectionPlan>,
) -> Result<RecordBatch> {
    let row_addrs = builder.get_row_addrs().await?.clone();

    if row_addrs.is_empty() {
        // It is possible that `row_id_index` returns None when a fragment has been wholly deleted
        return Ok(RecordBatch::new_empty(Arc::new(
            builder.projection.output_schema()?,
        )));
    }

    let row_addr_stats = check_row_addrs(&row_addrs);

    // This method is mostly to annotate the send bound to avoid the
    // higher-order lifetime error.
    // manually implemented async for Send bound
    #[allow(clippy::manual_async_fn)]
    fn do_take(
        fragment: FileFragment,
        row_offsets: Vec<u32>,
        projection: Arc<Schema>,
        with_row_addresses: bool,
    ) -> impl Future<Output = Result<RecordBatch>> + Send {
        async move {
            fragment
                .take_rows(&row_offsets, projection.as_ref(), with_row_addresses)
                .await
        }
    }

    let batch = if row_addr_stats.contiguous {
        // Fastest path: Can use `read_range` directly
        let start = row_addrs.first().expect("empty range passed to take_rows");
        let fragment_id = (start >> 32) as usize;
        let range_start = *start as u32 as usize;
        let range_end = *row_addrs.last().expect("empty range passed to take_rows") as u32 as usize;
        let range = range_start..(range_end + 1);

        let fragment = builder.dataset.get_fragment(fragment_id).ok_or_else(|| {
            Error::invalid_input(
                format!("_rowaddr belongs to non-existent fragment: {start}"),
                location!(),
            )
        })?;

        let reader = fragment
            .open(&projection.physical_schema, FragReadConfig::default(), None)
            .await?;
        reader.legacy_read_range_as_batch(range).await
    } else if row_addr_stats.sorted {
        // Don't need to re-arrange data, just concatenate

        let mut batches: Vec<_> = Vec::new();
        let mut current_fragment = row_addrs[0] >> 32;
        let mut current_start = 0;
        let mut row_addr_iter = row_addrs.iter().enumerate();
        'outer: loop {
            let (fragment_id, range) = loop {
                if let Some((i, row_id)) = row_addr_iter.next() {
                    let fragment_id = row_id >> 32;
                    if fragment_id != current_fragment {
                        let next = (current_fragment, current_start..i);
                        current_fragment = fragment_id;
                        current_start = i;
                        break next;
                    }
                } else if current_start != row_addrs.len() {
                    let next = (current_fragment, current_start..row_addrs.len());
                    current_start = row_addrs.len();
                    break next;
                } else {
                    break 'outer;
                }
            };

            let fragment = builder
                .dataset
                .get_fragment(fragment_id as usize)
                .ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "_rowaddr {} belongs to non-existent fragment: {}",
                            row_addrs[range.start], fragment_id
                        ),
                        location!(),
                    )
                })?;
            let row_offsets: Vec<u32> = row_addrs[range].iter().map(|x| *x as u32).collect();

            let batch_fut = do_take(
                fragment,
                row_offsets,
                projection.physical_schema.clone(),
                false,
            );
            batches.push(batch_fut);
        }
        let batches: Vec<RecordBatch> = futures::stream::iter(batches)
            .buffered(builder.dataset.object_store.io_parallelism())
            .try_collect()
            .await?;
        Ok(concat_batches(&batches[0].schema(), &batches)?)
    } else {
        let projection_with_row_addr = Schema::merge(
            projection.physical_schema.as_ref(),
            &ArrowSchema::new(vec![ArrowField::new(
                ROW_ADDR,
                arrow::datatypes::DataType::UInt64,
                false,
            )]),
        )?;
        let schema_with_row_addr = Arc::new(ArrowSchema::from(&projection_with_row_addr));

        // Slow case: need to re-map data into expected order
        let mut sorted_row_addrs = row_addrs.clone();
        sorted_row_addrs.sort();
        // Go ahead and dedup, we will reinsert duplicates during the remapping
        sorted_row_addrs.dedup();
        // Group ROW Ids by the fragment
        let mut row_addrs_per_fragment: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        sorted_row_addrs.iter().for_each(|row_addr| {
            let row_addr = RowAddress::from(*row_addr);
            let fragment_id = row_addr.fragment_id();
            let offset = row_addr.row_offset();
            row_addrs_per_fragment
                .entry(fragment_id)
                .and_modify(|v| v.push(offset))
                .or_insert_with(|| vec![offset]);
        });

        let fragments = builder.dataset.get_fragments();
        let fragment_and_indices = fragments.into_iter().filter_map(|f| {
            let row_offset = row_addrs_per_fragment.remove(&(f.id() as u32))?;
            Some((f, row_offset))
        });

        let mut batches = futures::stream::iter(fragment_and_indices)
            .map(|(fragment, indices)| {
                do_take(fragment, indices, projection.physical_schema.clone(), true)
            })
            .buffered(builder.dataset.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;

        let one_batch = if batches.len() > 1 {
            concat_batches(&schema_with_row_addr, &batches)?
        } else {
            batches.pop().unwrap()
        };
        // Note: one_batch may contains fewer rows than the number of requested
        // row ids because some rows may have been deleted. Because of this, we
        // get the results with row ids so that we can re-order the results
        // to match the requested order.

        let returned_row_addr = one_batch
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| Error::Internal {
                message: "_rowaddr column not found".into(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>()
            .values();

        let remapping_index: UInt64Array = row_addrs
            .iter()
            .filter_map(|o| {
                returned_row_addr
                    .iter()
                    .position(|id| id == o)
                    .map(|pos| pos as u64)
            })
            .collect();

        // remapping_index may be greater than the number of rows in one_batch
        // if there are duplicates in the requested row ids. This is expected.
        debug_assert!(remapping_index.len() >= one_batch.num_rows());

        // Remove the rowaddr column.
        let keep_indices = (0..one_batch.num_columns() - 1).collect::<Vec<_>>();
        let one_batch = one_batch.project(&keep_indices)?;
        let struct_arr: StructArray = one_batch.into();
        let reordered = arrow_select::take::take(&struct_arr, &remapping_index, None)?;
        Ok(as_struct_array(&reordered).into())
    }?;

    let batch = projection.project_batch(batch).await?;

    if builder.with_row_address {
        if batch.num_rows() != row_addrs.len() {
            return Err(Error::NotSupported  {
            source: format!(
                "Expected {} rows, got {}.  A take operation that includes row addresses must not target deleted rows.",
                row_addrs.len(),
                batch.num_rows()
            ).into(),
            location: location!(),
            });
        }

        let row_addr_col = Arc::new(UInt64Array::from(row_addrs));
        let row_addr_field = ArrowField::new(ROW_ADDR, arrow::datatypes::DataType::UInt64, false);
        Ok(batch.try_with_column(row_addr_field, row_addr_col)?)
    } else {
        Ok(batch)
    }
}

// Given a local take and a sibling take this function zips the results together
async fn zip_takes(
    local: RecordBatch,
    sibling: RecordBatch,
    orig_projection_plan: &ProjectionPlan,
) -> Result<RecordBatch> {
    let mut all_cols = Vec::with_capacity(local.num_columns() + sibling.num_columns());
    all_cols.extend(local.columns().iter().cloned());
    all_cols.extend(sibling.columns().iter().cloned());

    let mut all_fields = Vec::with_capacity(local.num_columns() + sibling.num_columns());
    all_fields.extend(local.schema().fields().iter().cloned());
    all_fields.extend(sibling.schema().fields().iter().cloned());

    let all_batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(all_fields)), all_cols).unwrap();

    orig_projection_plan.project_batch(all_batch).await
}

async fn take_rows(builder: TakeBuilder) -> Result<RecordBatch> {
    if builder.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(
            builder.projection.output_schema()?,
        )));
    }

    let projection = builder.projection.clone();
    let sibling_ds: Arc<Dataset>;

    // If we have sibling columns then we load those in parallel to the local
    // columns and zip the results together.
    let sibling_take = if let Some(sibling_schema) = projection.sibling_schema.as_ref() {
        let filtered_row_ids = builder.get_filtered_ids().await?;
        if filtered_row_ids.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(
                builder.projection.output_schema()?,
            )));
        }
        sibling_ds = builder
            .dataset
            .blobs_dataset()
            .await?
            .ok_or_else(|| Error::Internal {
                message: "schema referenced sibling columns but there was no blob dataset".into(),
                location: location!(),
            })?;
        // The sibling take only takes valid row ids and sibling columns
        let mut builder = builder.clone();
        builder.dataset = sibling_ds;
        builder.row_ids = Some(filtered_row_ids);
        builder.row_addrs = None;
        let blobs_projection = Arc::new(ProjectionPlan::inner_new(
            sibling_schema.clone(),
            false,
            None,
        ));
        Some(async move { do_take_rows(builder, blobs_projection).await })
    } else {
        None
    };

    if let Some(sibling_take) = sibling_take {
        if projection.physical_schema.fields.is_empty() {
            // Nothing we need from local dataset, just take from blob dataset
            sibling_take.await
        } else {
            // Need to take from both and zip together
            let local_projection = ProjectionPlan {
                physical_df_schema: projection.physical_df_schema.clone(),
                physical_schema: projection.physical_schema.clone(),
                sibling_schema: None,
                // These will be applied in zip_takes
                requested_output_expr: None,
            };
            let local_take = do_take_rows(builder, Arc::new(local_projection));
            let (local, blobs) = futures::join!(local_take, sibling_take);

            zip_takes(local?, blobs?, &projection).await
        }
    } else {
        do_take_rows(builder, projection).await
    }
}

/// Get a stream of batches based on iterator of ranges of row numbers.
///
/// This is an experimental API. It may change at any time.
pub fn take_scan(
    dataset: &Dataset,
    row_ranges: Pin<Box<dyn Stream<Item = Result<Range<u64>>> + Send>>,
    projection: Arc<Schema>,
    batch_readahead: usize,
) -> DatasetRecordBatchStream {
    let arrow_schema = Arc::new(projection.as_ref().into());
    let dataset = Arc::new(dataset.clone());
    let batch_stream = row_ranges
        .map(move |res| {
            let dataset = dataset.clone();
            let projection = projection.clone();
            let fut = async move {
                let range = res.map_err(|err| DataFusionError::External(Box::new(err)))?;
                let row_pos: Vec<u64> = (range.start..range.end).collect();
                dataset
                    .take(&row_pos, ProjectionRequest::Schema(projection.clone()))
                    .await
                    .map_err(|err| DataFusionError::External(Box::new(err)))
            };
            async move { tokio::task::spawn(fut).await.unwrap() }
        })
        .buffered(batch_readahead);

    DatasetRecordBatchStream::new(Box::pin(RecordBatchStreamAdapter::new(
        arrow_schema,
        batch_stream,
    )))
}

struct RowAddressStats {
    sorted: bool,
    contiguous: bool,
}

fn check_row_addrs(row_ids: &[u64]) -> RowAddressStats {
    let mut sorted = true;
    let mut contiguous = true;

    if row_ids.is_empty() {
        return RowAddressStats { sorted, contiguous };
    }

    let mut last_id = row_ids[0];
    let first_fragment_id = row_ids[0] >> 32;

    for id in row_ids.iter().skip(1) {
        sorted &= *id > last_id;
        contiguous &= *id == last_id + 1;
        // Contiguous also requires the fragment ids are all the same
        contiguous &= (*id >> 32) == first_fragment_id;
        last_id = *id;
    }

    RowAddressStats { sorted, contiguous }
}

/// Builder for the `take` operation.
#[derive(Clone, Debug)]
pub struct TakeBuilder {
    dataset: Arc<Dataset>,
    row_ids: Option<Vec<u64>>,
    row_addrs: Option<Vec<u64>>,
    projection: Arc<ProjectionPlan>,
    with_row_address: bool,
}

impl TakeBuilder {
    /// Create a new `TakeBuilder` for taking by id
    pub fn try_new_from_ids(
        dataset: Arc<Dataset>,
        row_ids: Vec<u64>,
        projection: ProjectionRequest,
    ) -> Result<Self> {
        Ok(Self {
            row_ids: Some(row_ids),
            row_addrs: None,
            projection: Arc::new(projection.into_projection_plan(dataset.schema())?),
            dataset,
            with_row_address: false,
        })
    }

    /// Create a new `TakeBuilder` for taking by address
    pub fn try_new_from_addresses(
        dataset: Arc<Dataset>,
        addresses: Vec<u64>,
        projection: Arc<ProjectionPlan>,
    ) -> Result<Self> {
        Ok(Self {
            row_ids: None,
            row_addrs: Some(addresses),
            projection,
            dataset,
            with_row_address: false,
        })
    }

    /// Adds row addresses to the output
    pub fn with_row_address(mut self, with_row_address: bool) -> Self {
        self.with_row_address = with_row_address;
        self
    }

    /// Execute the take operation and return a single batch
    pub async fn execute(self) -> Result<RecordBatch> {
        take_rows(self).await
    }

    pub fn is_empty(&self) -> bool {
        match (self.row_ids.as_ref(), self.row_addrs.as_ref()) {
            (Some(ids), _) => ids.is_empty(),
            (_, Some(addrs)) => addrs.is_empty(),
            _ => unreachable!(),
        }
    }

    async fn get_filtered_ids(&self) -> Result<Vec<u64>> {
        match (self.row_ids.as_ref(), self.row_addrs.as_ref()) {
            (Some(ids), _) => self.dataset.filter_deleted_ids(ids).await,
            (_, Some(addrs)) => {
                let _filtered_addresses = self.dataset.filter_deleted_addresses(addrs).await?;
                // TODO: Create an inverse mapping from addresses to ids
                // This path is currently encountered in the "take by dataset offsets" case.
                // Another solution could be to translate dataset offsets into row ids instead
                // of translating them into row addresses.
                Err(Error::NotSupported {
                    source: "mapping from row addresses to row ids".into(),
                    location: location!(),
                })
            }
            _ => unreachable!(),
        }
    }

    async fn get_row_addrs(&mut self) -> Result<&Vec<u64>> {
        if self.row_addrs.is_none() {
            let row_ids = self
                .row_ids
                .as_ref()
                .expect("row_ids must be set if row_addrs is not");
            let addrs = if let Some(row_id_index) = get_row_id_index(&self.dataset).await? {
                let addresses = row_ids
                    .iter()
                    .filter_map(|id| row_id_index.get(*id).map(|address| address.into()))
                    .collect::<Vec<_>>();
                addresses
            } else {
                row_ids.clone()
            };
            self.row_addrs = Some(addrs);
        }
        Ok(self.row_addrs.as_ref().unwrap())
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::DataType;
    use lance_file::version::LanceFileVersion;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dataset::{scanner::test_dataset::TestVectorDataset, WriteParams};

    use super::*;

    // Used to validate that futures returned are Send.
    fn require_send<T: Send>(t: T) -> T {
        t
    }

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
    async fn test_take(
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

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    200, // 200
                    199, // 199
                    39,  // 39
                    40,  // 40
                    199, // 40
                    40,  // 40
                    125, // 125
                ],
                projection,
            )
            .await
            .unwrap();
        assert_eq!(
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
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_with_projection(
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

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![("foo", "i"), ("bar", "i*2")]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
        ]));
        assert_eq!(
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

        let values2 = dataset.take_rows(&[10, 50, 100], projection).await.unwrap();
        assert_eq!(values, values2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows_out_of_bound(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // a dataset with 1 fragment and 400 rows
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let ds = test_ds.dataset;

        // take the last row of first fragment
        // this triggers the contiguous branch
        let indices = &[(1 << 32) - 1];
        let fut = require_send(ds.take_rows(indices, ds.schema().clone()));
        let err = fut.await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid read params"),
            "{}",
            err.to_string()
        );

        // this triggers the sorted branch, but not contiguous
        let indices = &[(1 << 32) - 3, (1 << 32) - 1];
        let err = ds
            .take_rows(indices, ds.schema().clone())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );

        // this triggers the catch all branch
        let indices = &[(1 << 32) - 1, (1 << 32) - 3];
        let err = ds
            .take_rows(indices, ds.schema().clone())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows(
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

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let indices = &[
            5_u64 << 32,        // 200
            (4_u64 << 32) + 39, // 199
            39,                 // 39
            1_u64 << 32,        // 40
            (2_u64 << 32) + 20, // 100
        ];
        let values = dataset
            .take_rows(indices, projection.clone())
            .await
            .unwrap();
        assert_eq!(
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
        let values = dataset
            .take_rows(indices, projection.clone())
            .await
            .unwrap();
        assert_eq!(
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
        let values = dataset.take_rows(&[], projection).await.unwrap();
        assert_eq!(RecordBatch::new_empty(data.schema()), values);
    }

    #[rstest]
    #[tokio::test]
    async fn take_scan_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use arrow::datatypes::Int32Type;

        let data = test_batch(1..5);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        let projection = Arc::new(dataset.schema().project(&["i"]).unwrap());
        let ranges = [0_u64..3, 1..4, 0..1];
        let range_stream = futures::stream::iter(ranges).map(Ok).boxed();
        let results = dataset
            .take_scan(range_stream, projection.clone(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let expected_schema = projection.as_ref().into();
        for batch in &results {
            assert_eq!(batch.schema().as_ref(), &expected_schema);
        }
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].column(0).as_primitive::<Int32Type>().values(),
            &[1, 2, 3],
        );
        assert_eq!(
            results[1].column(0).as_primitive::<Int32Type>().values(),
            &[2, 3, 4],
        );
        assert_eq!(
            results[2].column(0).as_primitive::<Int32Type>().values(),
            &[1],
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows_with_row_ids(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..8);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            data_storage_version: Some(data_storage_version),
            enable_move_stable_row_ids: true,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (1, 2, 3, 7)").await.unwrap();

        let indices = &[0, 4, 6, 5];
        let result = dataset
            .take_rows(indices, dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        indices.iter().map(|x| *x as i32)
                    )),
                    Arc::new(StringArray::from_iter_values(
                        indices.iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            result
        );
    }
}
