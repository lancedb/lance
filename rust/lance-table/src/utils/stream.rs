// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{make_array, BooleanArray, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_buffer::NullBuffer;
use futures::{
    future::BoxFuture,
    stream::{BoxStream, FuturesOrdered},
    FutureExt, Stream, StreamExt,
};
use lance_arrow::RecordBatchExt;
use lance_core::{
    utils::{address::RowAddress, deletion::DeletionVector},
    Result, ROW_ID, ROW_ID_FIELD,
};
use lance_io::ReadBatchParams;

use crate::rowids::RowIdSequence;

pub type ReadBatchFut = BoxFuture<'static, Result<RecordBatch>>;
/// A task, emitted by a file reader, that will produce a batch (of the
/// given size)
pub struct ReadBatchTask {
    pub task: ReadBatchFut,
    pub num_rows: u32,
}
pub type ReadBatchTaskStream = BoxStream<'static, ReadBatchTask>;
pub type ReadBatchFutStream = BoxStream<'static, ReadBatchFut>;

struct MergeStream {
    streams: Vec<ReadBatchTaskStream>,
    next_batch: FuturesOrdered<ReadBatchFut>,
    next_num_rows: u32,
    index: usize,
}

impl MergeStream {
    fn emit(&mut self) -> ReadBatchTask {
        let mut iter = std::mem::take(&mut self.next_batch);
        let task = async move {
            let mut batch = iter.next().await.unwrap()?;
            while let Some(next) = iter.next().await {
                let next = next?;
                batch = batch.merge(&next)?;
            }
            Ok(batch)
        }
        .boxed();
        let num_rows = self.next_num_rows;
        self.next_num_rows = 0;
        ReadBatchTask { task, num_rows }
    }
}

impl Stream for MergeStream {
    type Item = ReadBatchTask;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            let index = self.index;
            match self.streams[index].poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(batch_task)) => {
                    if self.index == 0 {
                        self.next_num_rows = batch_task.num_rows;
                    } else {
                        debug_assert_eq!(self.next_num_rows, batch_task.num_rows);
                    }
                    self.next_batch.push_back(batch_task.task);
                    self.index += 1;
                    if self.index == self.streams.len() {
                        self.index = 0;
                        let next_batch = self.emit();
                        return std::task::Poll::Ready(Some(next_batch));
                    }
                }
                std::task::Poll::Ready(None) => {
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

/// Given multiple streams of batch tasks, merge them into a single stream
///
/// This pulls one batch from each stream and then combines the columns from
/// all of the batches into a single batch.  The order of the batches in the
/// streams is maintained and the merged batch columns will be in order from
/// first to last stream.
///
/// This stream ends as soon as any of the input streams ends (we do not
/// verify that the other input streams are finished as well)
///
/// This will panic if any of the input streams return a batch with a different
/// number of rows than the first stream.
pub fn merge_streams(streams: Vec<ReadBatchTaskStream>) -> ReadBatchTaskStream {
    MergeStream {
        streams,
        next_batch: FuturesOrdered::new(),
        next_num_rows: 0,
        index: 0,
    }
    .boxed()
}

/// Apply a mask to the batch, where rows are "deleted" by the _rowid column null.
/// This is used as a performance optimization to avoid copying data.
fn apply_deletions_as_nulls(batch: RecordBatch, mask: &BooleanArray) -> Result<RecordBatch> {
    // Transform mask into null buffer. Null means deleted, though note that
    // null buffers are actually validity buffers, so True means not null
    // and thus not deleted.
    let mask_buffer = NullBuffer::new(mask.values().clone());

    match mask_buffer.null_count() {
        // All rows are deleted
        n if n == mask_buffer.len() => return Ok(RecordBatch::new_empty(batch.schema())),
        // No rows are deleted
        0 => return Ok(batch),
        _ => {}
    }

    // For each column convert to data
    let new_columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, col)| {
            if field.name() == ROW_ID {
                let col_data = col.to_data();
                // If it already has a validity bitmap, then AND it with the mask.
                // Otherwise, use the boolean buffer as the mask.
                let null_buffer = NullBuffer::union(col_data.nulls(), Some(&mask_buffer));

                Ok(col_data
                    .into_builder()
                    .null_bit_buffer(null_buffer.map(|b| b.buffer().clone()))
                    .build()
                    .map(make_array)?)
            } else {
                Ok(col.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        new_columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// Configuration needed to apply row ids and deletions to a batch
pub struct RowIdAndDeletesConfig {
    /// The row ids that were requested
    pub params: ReadBatchParams,
    /// Whether to include the row id column in the final batch
    pub with_row_id: bool,
    /// An optional deletion vector to apply to the batch
    pub deletion_vector: Option<Arc<DeletionVector>>,
    /// Whether to make deleted rows null instead of filtering them out
    pub make_deletions_null: bool,
    /// The total number of rows that will be loaded
    ///
    /// This is needed to convert ReadbatchParams::RangeTo into a valid range
    pub total_num_rows: u32,
}

fn apply_row_id_and_deletes(
    batch: RecordBatch,
    batch_offset: u32,
    fragment_id: u32,
    row_id_sequence: Option<&RowIdSequence>,
    config: &RowIdAndDeletesConfig,
) -> Result<RecordBatch> {
    let mut deletion_vector = config.deletion_vector.as_ref();
    // Convert Some(NoDeletions) into None to simplify logic below
    if let Some(deletion_vector_inner) = deletion_vector {
        if matches!(deletion_vector_inner.as_ref(), DeletionVector::NoDeletions) {
            deletion_vector = None;
        }
    }
    let has_deletions = deletion_vector.is_some();
    debug_assert!(batch.num_columns() > 0 || config.with_row_id || has_deletions);

    let should_fetch_row_id = config.with_row_id || has_deletions;

    let num_rows = batch.num_rows() as u32;

    let row_ids = if should_fetch_row_id {
        if let Some(row_id_sequence) = row_id_sequence {
            let row_ids = row_id_sequence
                .slice(batch_offset as usize, num_rows as usize)
                .iter()
                .collect::<Vec<u64>>();
            Some(row_ids)
        } else {
            let ids_in_batch = config
                .params
                .slice(batch_offset as usize, num_rows as usize)
                .unwrap()
                .to_offsets()
                .unwrap();
            let row_ids: Vec<u64> = ids_in_batch
                .values()
                .iter()
                .map(|row_id| u64::from(RowAddress::new_from_parts(fragment_id, *row_id)))
                .collect();

            Some(row_ids)
        }
    } else {
        None
    };

    // TODO: This is a minor cop out. Pushing deletion vector in to the decoders is hard
    // so I'm going to just leave deletion filter at this layer for now.
    // We should push this down futurther when we get to statistics-based predicate pushdown

    // This function is meant to be IO bound, but we are doing CPU-bound work here
    // We should try to move this to later.
    let span = tracing::span!(tracing::Level::DEBUG, "apply_deletions");
    let _enter = span.enter();
    let deletion_mask =
        deletion_vector.and_then(|v| v.build_predicate(row_ids.as_ref().unwrap().iter()));

    let batch = if config.with_row_id {
        let row_id_arr = Arc::new(UInt64Array::from(row_ids.unwrap()));
        batch.try_with_column(ROW_ID_FIELD.clone(), row_id_arr)?
    } else {
        batch
    };

    match (deletion_mask, config.make_deletions_null) {
        (None, _) => Ok(batch),
        (Some(mask), false) => Ok(arrow::compute::filter_record_batch(&batch, &mask)?),
        (Some(mask), true) => Ok(apply_deletions_as_nulls(batch, &mask)?),
    }
}

/// Given a stream of batch tasks this function will add a row ids column (if requested)
/// and also apply a deletions vector to the batch.
///
/// This converts from BatchTaskStream to BatchFutStream because, if we are applying a
/// deletion vector, it is impossible to know how many output rows we will have.
pub fn wrap_with_row_id_and_delete(
    stream: ReadBatchTaskStream,
    fragment_id: u32,
    row_id_sequence: Option<Arc<RowIdSequence>>,
    config: RowIdAndDeletesConfig,
) -> ReadBatchFutStream {
    let config = Arc::new(config);
    let mut offset = 0;
    stream
        .map(move |batch_task| {
            let config = config.clone();
            let this_offset = offset;
            let num_rows = batch_task.num_rows;
            offset += num_rows;
            let task = batch_task.task;
            let row_id_sequence_ref = row_id_sequence.clone();
            async move {
                let batch = task.await?;
                apply_row_id_and_deletes(
                    batch,
                    this_offset,
                    fragment_id,
                    row_id_sequence_ref.as_deref(),
                    config.as_ref(),
                )
            }
            .boxed()
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{array::AsArray, datatypes::UInt64Type};
    use arrow_array::{types::Int32Type, RecordBatch, UInt32Array};
    use arrow_schema::ArrowError;
    use futures::{stream::BoxStream, FutureExt, StreamExt, TryStreamExt};
    use lance_core::{
        utils::{address::RowAddress, deletion::DeletionVector},
        ROW_ID,
    };
    use lance_datagen::{BatchCount, RowCount};
    use lance_io::{stream::arrow_stream_to_lance_stream, ReadBatchParams};
    use roaring::RoaringBitmap;

    use crate::utils::stream::ReadBatchTask;

    use super::RowIdAndDeletesConfig;

    fn batch_task_stream(
        datagen_stream: BoxStream<'static, std::result::Result<RecordBatch, ArrowError>>,
    ) -> super::ReadBatchTaskStream {
        arrow_stream_to_lance_stream(datagen_stream)
            .map(|batch| ReadBatchTask {
                num_rows: batch.as_ref().unwrap().num_rows() as u32,
                task: std::future::ready(batch).boxed(),
            })
            .boxed()
    }

    #[tokio::test]
    async fn test_basic_zip() {
        let left = batch_task_stream(
            lance_datagen::gen()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_reader_stream(RowCount::from(100), BatchCount::from(10)),
        );
        let right = batch_task_stream(
            lance_datagen::gen()
                .col("y", lance_datagen::array::step::<Int32Type>())
                .into_reader_stream(RowCount::from(100), BatchCount::from(10)),
        );

        let merged = super::merge_streams(vec![left, right])
            .map(|batch_task| batch_task.task)
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let expected = lance_datagen::gen()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .col("y", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(10))
            .collect::<Result<Vec<_>, ArrowError>>()
            .unwrap();
        assert_eq!(merged, expected);
    }

    async fn check_row_id(params: ReadBatchParams, expected: impl IntoIterator<Item = u32>) {
        let expected = Vec::from_iter(expected);

        for has_columns in [false, true] {
            for fragment_id in [0, 10] {
                // 100 rows across 10 batches of 10 rows
                let mut datagen = lance_datagen::gen();
                if has_columns {
                    datagen = datagen.col("x", lance_datagen::array::rand::<Int32Type>());
                }
                let data = batch_task_stream(
                    datagen.into_reader_stream(RowCount::from(10), BatchCount::from(10)),
                );

                let config = RowIdAndDeletesConfig {
                    params: params.clone(),
                    with_row_id: true,
                    deletion_vector: None,
                    make_deletions_null: false,
                    total_num_rows: 100,
                };
                let stream = super::wrap_with_row_id_and_delete(data, fragment_id, None, config);
                let batches = stream.buffered(1).try_collect::<Vec<_>>().await.unwrap();

                let mut offset = 0;
                let expected = expected.clone();
                for batch in batches {
                    let actual_row_ids =
                        batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec();
                    let expected_row_ids = expected[offset..offset + 10]
                        .iter()
                        .map(|row_offset| {
                            RowAddress::new_from_parts(fragment_id, *row_offset).into()
                        })
                        .collect::<Vec<u64>>();
                    assert_eq!(actual_row_ids, expected_row_ids);
                    offset += batch.num_rows();
                }
            }
        }
    }

    #[tokio::test]
    async fn test_row_id() {
        let some_indices = (0..100).rev().collect::<Vec<u32>>();
        let some_indices_arr = UInt32Array::from(some_indices.clone());
        check_row_id(ReadBatchParams::RangeFull, 0..100).await;
        check_row_id(ReadBatchParams::Indices(some_indices_arr), some_indices).await;
        check_row_id(ReadBatchParams::Range(1000..1100), 1000..1100).await;
        check_row_id(
            ReadBatchParams::RangeFrom(std::ops::RangeFrom { start: 1000 }),
            1000..1100,
        )
        .await;
        check_row_id(
            ReadBatchParams::RangeTo(std::ops::RangeTo { end: 1000 }),
            0..100,
        )
        .await;
    }

    #[tokio::test]
    async fn test_deletes() {
        let no_deletes: Option<Arc<DeletionVector>> = None;
        let no_deletes_2 = Some(Arc::new(DeletionVector::NoDeletions));
        let delete_some_bitmap = Some(Arc::new(DeletionVector::Bitmap(RoaringBitmap::from_iter(
            0..35,
        ))));
        let delete_some_set = Some(Arc::new(DeletionVector::Set((0..35).collect())));

        for deletion_vector in [
            no_deletes,
            no_deletes_2,
            delete_some_bitmap,
            delete_some_set,
        ] {
            for has_columns in [false, true] {
                for with_row_id in [false, true] {
                    for make_deletions_null in [false, true] {
                        for frag_id in [0, 1] {
                            let has_deletions = if let Some(dv) = &deletion_vector {
                                !matches!(dv.as_ref(), DeletionVector::NoDeletions)
                            } else {
                                false
                            };
                            if !has_columns && !has_deletions && !with_row_id {
                                // This is an invalid case and should be prevented upstream,
                                // no meaningful work is being done!
                                continue;
                            }
                            if make_deletions_null && !with_row_id {
                                // This is an invalid case and should be prevented upstream
                                // we cannot make the row_id column null if it isn't present
                                continue;
                            }

                            let mut datagen = lance_datagen::gen();
                            if has_columns {
                                datagen =
                                    datagen.col("x", lance_datagen::array::rand::<Int32Type>());
                            }
                            // 100 rows across 10 batches of 10 rows
                            let data = batch_task_stream(
                                datagen
                                    .into_reader_stream(RowCount::from(10), BatchCount::from(10)),
                            );

                            let config = RowIdAndDeletesConfig {
                                params: ReadBatchParams::RangeFull,
                                with_row_id,
                                deletion_vector: deletion_vector.clone(),
                                make_deletions_null,
                                total_num_rows: 100,
                            };
                            let stream =
                                super::wrap_with_row_id_and_delete(data, frag_id, None, config);
                            let batches = stream
                                .buffered(1)
                                .filter_map(|batch| {
                                    std::future::ready(
                                        batch
                                            .map(|batch| {
                                                if batch.num_rows() == 0 {
                                                    None
                                                } else {
                                                    Some(batch)
                                                }
                                            })
                                            .transpose(),
                                    )
                                })
                                .try_collect::<Vec<_>>()
                                .await
                                .unwrap();

                            let total_num_rows =
                                batches.iter().map(|b| b.num_rows()).sum::<usize>();
                            let total_num_nulls = if make_deletions_null {
                                batches
                                    .iter()
                                    .map(|b| b[ROW_ID].null_count())
                                    .sum::<usize>()
                            } else {
                                0
                            };
                            let total_actually_deleted = total_num_nulls + (100 - total_num_rows);

                            let expected_deletions = match &deletion_vector {
                                None => 0,
                                Some(deletion_vector) => match deletion_vector.as_ref() {
                                    DeletionVector::NoDeletions => 0,
                                    DeletionVector::Bitmap(b) => b.len() as usize,
                                    DeletionVector::Set(s) => s.len(),
                                },
                            };
                            assert_eq!(total_actually_deleted, expected_deletions);
                            if expected_deletions > 0 && with_row_id {
                                if make_deletions_null {
                                    assert_eq!(
                                        batches[0][ROW_ID].as_primitive::<UInt64Type>().value(0),
                                        u64::from(RowAddress::new_from_parts(frag_id, 30))
                                    );
                                } else {
                                    assert_eq!(
                                        batches[0][ROW_ID].as_primitive::<UInt64Type>().value(0),
                                        u64::from(RowAddress::new_from_parts(frag_id, 35))
                                    );
                                }
                            }
                            if !with_row_id {
                                assert!(batches[0].column_by_name(ROW_ID).is_none());
                            }
                        }
                    }
                }
            }
        }
    }
}
