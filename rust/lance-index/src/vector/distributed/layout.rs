// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Restore row-major quantized code layouts when re-reading merged segments.
//!
//! A merged vector segment stores its codes in the query-optimized layout:
//! packed SIMD blocks for RaBitQ (`packed = true`) and column-major for PQ
//! (`transposed = true`). Re-merging such a segment must invert that layout
//! back to row-major first, per partition and per shard, because the merge
//! concatenates partitions across shards before applying the layout once.
//! Codes in the optimized layout cannot be concatenated, and applying the
//! layout to already-optimized codes corrupts them silently.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};

use crate::vector::PQ_CODE_COLUMN;
use crate::vector::bq::storage::{RABIT_CODE_COLUMN, unpack_codes};
use crate::vector::pq::storage::transpose;

/// Reassemble each partition's batches and restore its code layout.
///
/// Each partition is laid out as an independent unit, so its rows must be
/// reassembled into one batch (a partition can span multiple stream batches)
/// before `restore` can invert the layout.
pub(super) fn restore_partition_layout(
    per_partition_batches: &mut [Vec<RecordBatch>],
    restore: impl Fn(RecordBatch) -> Result<RecordBatch>,
) -> Result<()> {
    for batches in per_partition_batches.iter_mut() {
        if batches.is_empty() {
            continue;
        }
        let schema = batches[0].schema();
        let merged = concat_batches(&schema, batches.iter())?;
        *batches = vec![restore(merged)?];
    }
    Ok(())
}

/// Unpack one partition's RaBitQ binary codes from SIMD blocks to row-major.
pub(super) fn unpack_rq_partition(partition: RecordBatch) -> Result<RecordBatch> {
    let rq_col = partition.column_by_name(RABIT_CODE_COLUMN).ok_or_else(|| {
        Error::index(format!(
            "RQ column {} missing in packed shard",
            RABIT_CODE_COLUMN
        ))
    })?;
    let rq_fsl = rq_col.as_fixed_size_list_opt().ok_or_else(|| {
        Error::index(format!(
            "RQ column {} is not a FixedSizeList in packed shard, got {}",
            RABIT_CODE_COLUMN,
            rq_col.data_type(),
        ))
    })?;
    let unpacked = unpack_codes(rq_fsl);
    Ok(partition.replace_column_by_name(RABIT_CODE_COLUMN, Arc::new(unpacked))?)
}

/// Transpose one partition's PQ codes from column-major back to row-major.
pub(super) fn untranspose_pq_partition(partition: RecordBatch) -> Result<RecordBatch> {
    let num_rows = partition.num_rows();
    if num_rows == 0 {
        return Ok(partition);
    }
    let pq_col = partition.column_by_name(PQ_CODE_COLUMN).ok_or_else(|| {
        Error::index(format!(
            "PQ column {} missing in transposed shard",
            PQ_CODE_COLUMN
        ))
    })?;
    let pq_fsl = pq_col.as_fixed_size_list_opt().ok_or_else(|| {
        Error::index(format!(
            "PQ column {} is not a FixedSizeList in transposed shard, got {}",
            PQ_CODE_COLUMN,
            pq_col.data_type(),
        ))
    })?;
    let num_bytes = pq_fsl.value_length() as usize;
    let values = pq_fsl
        .values()
        .as_primitive_opt::<UInt8Type>()
        .ok_or_else(|| {
            Error::index(format!(
                "PQ column {} values are not u8 in transposed shard, got {}",
                PQ_CODE_COLUMN,
                pq_fsl.values().data_type(),
            ))
        })?;
    let row_major_codes = transpose(values, num_bytes, num_rows);
    let row_major_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
        row_major_codes,
        num_bytes as i32,
    )?);
    Ok(partition.replace_column_by_name(PQ_CODE_COLUMN, row_major_fsl)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, UInt8Array};
    use arrow_schema::{Field, Schema as ArrowSchema};

    use crate::vector::bq::storage::pack_codes;

    fn code_batch(column: &str, codes: FixedSizeListArray) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                column,
                codes.data_type().clone(),
                true,
            )])),
            vec![Arc::new(codes)],
        )
        .unwrap()
    }

    fn u8_codes(values: &[u8], width: i32) -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(UInt8Array::from(values.to_vec()), width).unwrap()
    }

    fn column_values(batch: &RecordBatch, column: &str) -> Vec<u8> {
        batch
            .column_by_name(column)
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .to_vec()
    }

    #[test]
    fn test_unpack_rq_partition_inverts_pack_codes() {
        // 64 rows (a multiple of the 32-row SIMD block) of 16-byte codes so
        // the pack/unpack round trip is exact and non-trivial.
        let num_rows = 64usize;
        let width = 16i32;
        let values: Vec<u8> = (0..num_rows * width as usize)
            .map(|v| (v % 251) as u8)
            .collect();

        let packed = pack_codes(&u8_codes(&values, width));
        let restored = unpack_rq_partition(code_batch(RABIT_CODE_COLUMN, packed)).unwrap();

        assert_eq!(column_values(&restored, RABIT_CODE_COLUMN), values);
    }

    #[test]
    fn test_untranspose_pq_partition_inverts_write_side_transpose() {
        // Rectangular shape (5 rows x 3 bytes) so a wrong transpose argument
        // order cannot produce the original values by accident.
        let num_rows = 5usize;
        let width = 3i32;
        let values: Vec<u8> = (0..(num_rows * width as usize) as u8).collect();

        let row_major = u8_codes(&values, width);
        let transposed = transpose(
            row_major.values().as_primitive::<UInt8Type>(),
            num_rows,
            width as usize,
        );
        let transposed_fsl = FixedSizeListArray::try_new_from_values(transposed, width).unwrap();

        let restored =
            untranspose_pq_partition(code_batch(PQ_CODE_COLUMN, transposed_fsl)).unwrap();

        assert_eq!(column_values(&restored, PQ_CODE_COLUMN), values);
    }

    #[test]
    fn test_untranspose_pq_partition_rejects_non_u8_codes() {
        let codes =
            FixedSizeListArray::try_new_from_values(Int32Array::from(vec![0, 1, 2, 3]), 2).unwrap();

        let result = untranspose_pq_partition(code_batch(PQ_CODE_COLUMN, codes));

        let err = result.unwrap_err().to_string();
        assert!(err.contains("not u8"), "unexpected error: {err}");
    }
}
