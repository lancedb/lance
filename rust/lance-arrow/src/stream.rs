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
