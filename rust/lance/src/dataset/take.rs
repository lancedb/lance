// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::{collections::BTreeMap, ops::Range, pin::Pin, sync::Arc};

use crate::dataset::rowids::get_row_id_index;
use crate::{Error, Result};
use arrow::{array::as_struct_array, compute::concat_batches, datatypes::UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, StructArray, UInt64Array};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use arrow_select::interleave::interleave;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{Future, Stream, StreamExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::ROW_ADDR;
use snafu::{location, Location};

use super::{fragment::FileFragment, scanner::DatasetRecordBatchStream, Dataset};

pub async fn take(
    dataset: &Dataset,
    row_indices: &[u64],
    projection: &Schema,
) -> Result<RecordBatch> {
    if row_indices.is_empty() {
        let schema = Arc::new(projection.into());
        return Ok(RecordBatch::new_empty(schema));
    }

    let mut sorted_indices: Vec<usize> = (0..row_indices.len()).collect();
    sorted_indices.sort_by_key(|&i| row_indices[i]);

    let fragments = dataset.get_fragments();

    // We will split into sub-requests for each fragment.
    let mut sub_requests: Vec<(&FileFragment, Range<usize>)> = Vec::new();
    // We will remap the row indices to the original row indices, using a pair
    // of (request position, position in request)
    let mut remap_index: Vec<(usize, usize)> = vec![(0, 0); row_indices.len()];
    let mut local_ids_buffer: Vec<u32> = Vec::with_capacity(row_indices.len());

    let mut fragments_iter = fragments.iter();
    let mut current_fragment = fragments_iter.next().ok_or_else(|| Error::InvalidInput {
        source: "Called take on an empty dataset.".to_string().into(),
        location: location!(),
    })?;
    let mut current_fragment_len = current_fragment.count_rows().await?;
    let mut curr_fragment_offset: u64 = 0;
    let mut current_fragment_end = current_fragment_len as u64;
    let mut start = 0;
    let mut end = 0;
    // We want to keep track of the previous row_index to detect duplicates
    // index takes. To start, we pick a value that is guaranteed to be different
    // from the first row_index.
    let mut previous_row_index: u64 = row_indices[sorted_indices[0]] + 1;
    let mut previous_sorted_index: usize = 0;

    for index in sorted_indices {
        // Get the index
        let row_index = row_indices[index];

        if previous_row_index == row_index {
            // If we have a duplicate index request we add a remap_index
            // entry that points to the original index request.
            remap_index[index] = remap_index[previous_sorted_index];
            continue;
        } else {
            previous_sorted_index = index;
            previous_row_index = row_index;
        }

        // If the row index is beyond the current fragment, iterate
        // until we find the fragment that contains it.
        while row_index >= current_fragment_end {
            // If we have a non-empty sub-request, add it to the list
            if end - start > 0 {
                // If we have a non-empty sub-request, add it to the list
                sub_requests.push((current_fragment, start..end));
            }

            start = end;

            current_fragment = fragments_iter.next().ok_or_else(|| Error::InvalidInput {
                source: format!(
                    "Row index {} is beyond the range of the dataset.",
                    row_index
                )
                .into(),
                location: location!(),
            })?;
            curr_fragment_offset += current_fragment_len as u64;
            current_fragment_len = current_fragment.count_rows().await?;
            current_fragment_end = curr_fragment_offset + current_fragment_len as u64;
        }

        // Note that we cast to u32 *after* subtracting the offset,
        // since it is possible for the global index to be larger than
        // u32::MAX.
        let local_index = (row_index - curr_fragment_offset) as u32;
        local_ids_buffer.push(local_index);

        remap_index[index] = (sub_requests.len(), end - start);

        end += 1;
    }

    // flush last batch
    if end - start > 0 {
        sub_requests.push((current_fragment, start..end));
    }

    let take_tasks = sub_requests
        .into_iter()
        .map(|(fragment, indices_range)| {
            let local_ids = &local_ids_buffer[indices_range];
            fragment.take(local_ids, projection)
        })
        .collect::<Vec<_>>();
    let batches = futures::stream::iter(take_tasks)
        .buffered(num_cpus::get() * 4)
        .try_collect::<Vec<RecordBatch>>()
        .await?;

    let struct_arrs: Vec<StructArray> = batches.into_iter().map(StructArray::from).collect();
    let refs: Vec<_> = struct_arrs.iter().map(|x| x as &dyn Array).collect();
    let reordered = interleave(&refs, &remap_index)?;
    Ok(as_struct_array(&reordered).into())
}

/// Take rows by the internal ROW ids.
pub async fn take_rows(
    dataset: &Dataset,
    row_ids: &[u64],
    projection: &Schema,
) -> Result<RecordBatch> {
    if row_ids.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(projection.into())));
    }

    let row_addrs = if let Some(row_id_index) = get_row_id_index(dataset).await? {
        let addresses = row_ids
            .iter()
            .filter_map(|id| row_id_index.get(*id).map(|address| address.into()))
            .collect::<Vec<_>>();
        Cow::Owned(addresses)
    } else {
        Cow::Borrowed(row_ids)
    };

    let projection = Arc::new(projection.clone());
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

    if row_addr_stats.contiguous {
        // Fastest path: Can use `read_range` directly
        let start = row_addrs.first().expect("empty range passed to take_rows");
        let fragment_id = (start >> 32) as usize;
        let range_start = *start as u32 as usize;
        let range_end = *row_addrs.last().expect("empty range passed to take_rows") as u32 as usize;
        let range = range_start..(range_end + 1);

        let fragment = dataset.get_fragment(fragment_id).ok_or_else(|| {
            Error::invalid_input(
                format!("_rowaddr belongs to non-existent fragment: {start}"),
                location!(),
            )
        })?;

        let reader = fragment.open(projection.as_ref(), false, false).await?;
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

            let fragment = dataset.get_fragment(fragment_id as usize).ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "_rowaddr {} belongs to non-existent fragment: {}",
                        row_addrs[range.start], fragment_id
                    ),
                    location!(),
                )
            })?;
            let row_offsets: Vec<u32> = row_addrs[range].iter().map(|x| *x as u32).collect();

            let batch_fut = do_take(fragment, row_offsets, projection.clone(), false);
            batches.push(batch_fut);
        }
        let batches: Vec<RecordBatch> = futures::stream::iter(batches)
            .buffered(4 * num_cpus::get())
            .try_collect()
            .await?;
        Ok(concat_batches(&batches[0].schema(), &batches)?)
    } else {
        let projection_with_row_addr = Schema::merge(
            projection.as_ref(),
            &ArrowSchema::new(vec![ArrowField::new(
                ROW_ADDR,
                arrow::datatypes::DataType::UInt64,
                false,
            )]),
        )?;
        let schema_with_row_addr = Arc::new(ArrowSchema::from(&projection_with_row_addr));

        // Slow case: need to re-map data into expected order
        let mut sorted_row_addrs = Vec::from(row_addrs.clone());
        sorted_row_addrs.sort();
        // Group ROW Ids by the fragment
        let mut row_addrs_per_fragment: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        sorted_row_addrs.iter().for_each(|row_addr| {
            let row_addr = RowAddress::new_from_id(*row_addr);
            let fragment_id = row_addr.fragment_id();
            let offset = row_addr.row_id();
            row_addrs_per_fragment
                .entry(fragment_id)
                .and_modify(|v| v.push(offset))
                .or_insert_with(|| vec![offset]);
        });

        let fragments = dataset.get_fragments();
        let fragment_and_indices = fragments.into_iter().filter_map(|f| {
            let row_offset = row_addrs_per_fragment.remove(&(f.id() as u32))?;
            Some((f, row_offset))
        });

        let mut batches = futures::stream::iter(fragment_and_indices)
            .map(|(fragment, indices)| do_take(fragment, indices, projection.clone(), true))
            .buffered(4 * num_cpus::get())
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

        debug_assert_eq!(remapping_index.len(), one_batch.num_rows());

        // Remove the rowaddr column.
        let keep_indices = (0..one_batch.num_columns() - 1).collect::<Vec<_>>();
        let one_batch = one_batch.project(&keep_indices)?;
        let struct_arr: StructArray = one_batch.into();
        let reordered = arrow_select::take::take(&struct_arr, &remapping_index, None)?;
        Ok(as_struct_array(&reordered).into())
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
                    .take(&row_pos, projection.as_ref())
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

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::DataType;
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
                    i_range.clone().map(|i| format!("str-{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_take(
        #[values(false, true)] use_legacy_format: bool,
        #[values(false, true)] enable_move_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            use_legacy_format,
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
                &projection,
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
    async fn test_take_rows_out_of_bound(#[values(false, true)] use_legacy_format: bool) {
        // a dataset with 1 fragment and 400 rows
        let test_ds = TestVectorDataset::new(use_legacy_format, false)
            .await
            .unwrap();
        let ds = test_ds.dataset;

        // take the last row of first fragment
        // this triggers the contiguous branch
        let indices = &[(1 << 32) - 1];
        let fut = require_send(ds.take_rows(indices, ds.schema()));
        let err = fut.await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid read params"),
            "{}",
            err.to_string()
        );

        // this triggers the sorted branch, but not contiguous
        let indices = &[(1 << 32) - 3, (1 << 32) - 1];
        let err = ds.take_rows(indices, ds.schema()).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );

        // this triggers the catch all branch
        let indices = &[(1 << 32) - 1, (1 << 32) - 3];
        let err = ds.take_rows(indices, ds.schema()).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows(#[values(false, true)] use_legacy_format: bool) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            use_legacy_format,
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
        let values = dataset.take_rows(indices, &projection).await.unwrap();
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
        let values = dataset.take_rows(indices, &projection).await.unwrap();
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
        let values = dataset.take_rows(&[], &projection).await.unwrap();
        assert_eq!(RecordBatch::new_empty(data.schema()), values);
    }

    #[rstest]
    #[tokio::test]
    async fn take_scan_dataset(#[values(false, true)] use_legacy_format: bool) {
        use arrow::datatypes::Int32Type;

        let data = test_batch(1..5);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            use_legacy_format,
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
    async fn test_take_rows_with_row_ids(#[values(false, true)] use_legacy_format: bool) {
        let data = test_batch(0..8);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            use_legacy_format,
            enable_move_stable_row_ids: true,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (1, 2, 3, 7)").await.unwrap();

        let indices = &[0, 4, 6, 5];
        let result = dataset.take_rows(indices, dataset.schema()).await.unwrap();
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
