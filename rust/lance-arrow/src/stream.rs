// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with streams of [`RecordBatch`].

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, SchemaRef};
use futures::stream::{self, Stream, StreamExt};
use std::pin::Pin;

/// Rechunks a stream of [`RecordBatch`] so that each output batch has
/// approximately `target_bytes` of array data.
///
/// Small input batches are accumulated (by concatenation) until at least
/// `min_bytes` of data has been collected. If the resulting batch exceeds
/// `max_bytes`, it is sliced into roughly equal pieces of ~`max_bytes`
/// (assuming uniform row sizes).
pub fn rechunk_stream_by_size<S, E>(
    input: S,
    input_schema: SchemaRef,
    min_bytes: usize,
    max_bytes: usize,
) -> impl Stream<Item = Result<RecordBatch, E>>
where
    S: Stream<Item = Result<RecordBatch, E>>,
    E: From<ArrowError>,
{
    stream::try_unfold(
        RechunkState {
            input: Box::pin(input),
            accumulated: Vec::new(),
            acc_bytes: 0,
            done: false,
            input_schema,
            min_bytes,
            max_bytes,
        },
        |mut state| async move {
            if state.done && state.accumulated.is_empty() {
                return Ok(None);
            }

            // Pull batches until we reach the byte target or exhaust input.
            while !state.done && state.acc_bytes < state.min_bytes {
                match state.input.next().await {
                    Some(Ok(batch)) => {
                        state.acc_bytes += batch.get_array_memory_size();
                        state.accumulated.push(batch);
                    }
                    Some(Err(e)) => return Err(e),
                    None => {
                        state.done = true;
                    }
                }
            }

            if state.accumulated.is_empty() {
                return Ok(None);
            }

            // Fast path: if the first accumulated batch already meets the
            // byte threshold, deliver it directly instead of concatenating
            // everything together (which would just get sliced back apart).
            if state.accumulated.len() > 1
                && state.accumulated[0].get_array_memory_size() >= state.min_bytes
            {
                let b = state.accumulated.remove(0);
                state.acc_bytes -= b.get_array_memory_size();
                return Ok(Some((b, state)));
            }

            let batch = if state.accumulated.len() == 1 {
                state.accumulated.pop().unwrap()
            } else {
                let b =
                    arrow_select::concat::concat_batches(&state.input_schema, &state.accumulated)
                        .map_err(E::from)?;
                state.accumulated.clear();
                b
            };
            state.acc_bytes = 0;

            // Slice the batch into ~max_bytes pieces assuming uniform row sizes.
            let batch_bytes = batch.get_array_memory_size();
            let num_rows = batch.num_rows();
            if batch_bytes <= state.max_bytes || num_rows <= 1 {
                Ok(Some((batch, state)))
            } else {
                let rows_per_chunk =
                    (state.max_bytes as u64 * num_rows as u64 / batch_bytes as u64).max(1) as usize;
                let mut slices = Vec::new();
                let mut offset = 0;
                while offset < num_rows {
                    let len = rows_per_chunk.min(num_rows - offset);
                    slices.push(batch.slice(offset, len));
                    offset += len;
                }

                let first = slices.remove(0);

                // Stash leftover slices for subsequent iterations.
                for a in &slices {
                    state.acc_bytes += a.get_array_memory_size();
                }
                state.accumulated = slices;

                Ok(Some((first, state)))
            }
        },
    )
}

/// Internal state for [`rechunk_stream`].
///
/// Kept as a named struct so the `try_unfold` closure stays readable.
struct RechunkState<S> {
    input: Pin<Box<S>>,
    accumulated: Vec<RecordBatch>,
    acc_bytes: usize,
    done: bool,
    input_schema: SchemaRef,
    min_bytes: usize,
    max_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use futures::executor::block_on;

    fn make_batch(num_rows: usize) -> RecordBatch {
        let schema = test_schema();
        let values: Vec<i32> = (0..num_rows as i32).collect();
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]))
    }

    fn collect_rechunked(
        batches: Vec<RecordBatch>,
        min_bytes: usize,
        max_bytes: usize,
    ) -> Vec<RecordBatch> {
        let input = stream::iter(batches.into_iter().map(Ok::<_, ArrowError>));
        let rechunked = rechunk_stream_by_size(input, test_schema(), min_bytes, max_bytes);
        block_on(rechunked.collect::<Vec<_>>())
            .into_iter()
            .map(|r| r.unwrap())
            .collect()
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    #[test]
    fn test_empty_stream() {
        let result = collect_rechunked(vec![], 100, 200);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_batch_passthrough() {
        let batch = make_batch(100);
        let bytes = batch.get_array_memory_size();
        // Batch is between min and max — should pass through as-is.
        let result = collect_rechunked(vec![batch], bytes / 2, bytes * 2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 100);
    }

    #[test]
    fn test_small_batches_concatenated() {
        let one_batch_bytes = make_batch(10).get_array_memory_size();
        let batches: Vec<_> = (0..8).map(|_| make_batch(10)).collect();
        // min = 5 batches worth, max = 10 batches worth.
        let result = collect_rechunked(batches, one_batch_bytes * 5, one_batch_bytes * 10);
        assert_eq!(total_rows(&result), 80);
        // Should have been concatenated into fewer batches than the 8 inputs.
        assert!(
            result.len() < 8,
            "expected fewer output batches, got {}",
            result.len()
        );
    }

    #[test]
    fn test_large_batch_sliced() {
        let batch = make_batch(1000);
        let bytes = batch.get_array_memory_size();
        let result = collect_rechunked(vec![batch], bytes / 8, bytes / 4);
        assert_eq!(total_rows(&result), 1000);
        assert!(
            result.len() >= 4,
            "expected at least 4 slices, got {}",
            result.len()
        );
    }

    #[test]
    fn test_sliced_leftovers_are_not_recombined() {
        // Key test for the fast-path optimisation. When a large batch is
        // sliced, leftover slices should be delivered one-at-a-time without
        // being concatenated back together.  We verify this by checking that
        // every output buffer pointer falls inside the original batch's
        // allocation (i.e. they are all zero-copy slices, not fresh copies).
        let batch = make_batch(1000);
        let bytes = batch.get_array_memory_size();
        let orig_data = batch.column(0).to_data();
        let orig_buf = &orig_data.buffers()[0];
        let orig_start = orig_buf.as_ptr() as usize;
        let orig_end = orig_start + orig_buf.len();

        let result = collect_rechunked(vec![batch], bytes / 8, bytes / 4);

        assert_eq!(total_rows(&result), 1000);
        assert!(result.len() >= 4);

        for (i, b) in result.iter().enumerate() {
            let ptr = b.column(0).to_data().buffers()[0].as_ptr() as usize;
            assert!(
                ptr >= orig_start && ptr < orig_end,
                "slice {i} buffer at {ptr:#x} is outside the original allocation \
                 [{orig_start:#x}, {orig_end:#x}) — it was re-concatenated"
            );
        }
    }

    #[test]
    fn test_flush_remainder_on_stream_end() {
        // Data below min_bytes should still be flushed when the stream ends.
        let batch = make_batch(10);
        let bytes = batch.get_array_memory_size();
        let result = collect_rechunked(vec![batch], bytes * 100, bytes * 200);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 10);
    }

    #[test]
    fn test_large_then_small_batches() {
        // After a large batch is fully drained, subsequent small batches
        // should be accumulated normally.
        let large = make_batch(1000);
        let small_bytes = make_batch(10).get_array_memory_size();
        let batches = vec![
            large,
            make_batch(10),
            make_batch(10),
            make_batch(10),
            make_batch(10),
            make_batch(10),
        ];
        let result = collect_rechunked(batches, small_bytes * 3, small_bytes * 100);
        assert_eq!(total_rows(&result), 1050);
        // The large batch should appear (possibly sliced) followed by
        // concatenated small batches, so we should have fewer output batches
        // than the 6 inputs.
        assert!(result.len() < 6);
    }

    #[test]
    fn test_row_preservation_across_slicing() {
        // Verify that every input row appears exactly once in the output
        // and in the correct order after slicing.
        let batch = make_batch(237); // odd count to exercise remainder slice
        let bytes = batch.get_array_memory_size();
        let result = collect_rechunked(vec![batch], bytes / 8, bytes / 5);

        assert_eq!(total_rows(&result), 237);

        let values: Vec<i32> = result
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        let expected: Vec<i32> = (0..237).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn test_error_propagation() {
        let input = stream::iter(vec![
            Ok(make_batch(10)),
            Err(ArrowError::ComputeError("boom".into())),
            Ok(make_batch(10)),
        ]);
        let rechunked = rechunk_stream_by_size(input, test_schema(), 1, usize::MAX);
        let results: Vec<Result<RecordBatch, ArrowError>> = block_on(rechunked.collect());
        assert!(results.iter().any(|r| r.is_err()));
    }
}
