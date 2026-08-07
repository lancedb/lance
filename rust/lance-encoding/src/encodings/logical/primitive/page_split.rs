// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, ArrayRef};
use lance_core::{Error, Result};

use super::{
    PrimitivePageData, PrimitivePageStructure, PrimitiveStructuralEncoder, SerializedRepDefs,
};

/// Split dense pages into independently encodable tasks using the unencoded byte size.
///
/// Page sizes are estimates because variable-width values may be uneven and compression changes
/// the final size. Repeated values are left intact because leaf-value slices do not necessarily
/// align with their top-level row boundaries.
pub(super) fn split_dense_pages(
    pages: Vec<PrimitivePageData>,
    arrays: &[ArrayRef],
    max_page_bytes: u64,
    num_rows: u64,
    num_values: u64,
) -> Result<Vec<PrimitivePageData>> {
    if max_page_bytes == 0 {
        return Err(Error::invalid_input(
            "max_page_bytes must be greater than zero for primitive page splitting, got max_page_bytes=0",
        ));
    }
    if num_rows <= 1 || num_values <= 1 {
        return Ok(pages);
    }

    let total_size_bytes = arrays.iter().try_fold(0_u64, |total, array| {
        let array_size = u64::try_from(array.get_array_memory_size()).map_err(|_| {
            Error::internal(format!(
                "Array memory size {} does not fit in u64",
                array.get_array_memory_size()
            ))
        })?;
        total.checked_add(array_size).ok_or_else(|| {
            Error::internal(format!(
                "Total array buffer size overflowed u64 while adding {} bytes",
                array_size
            ))
        })
    })?;
    if total_size_bytes <= max_page_bytes {
        return Ok(pages);
    }

    let desired_pages = total_size_bytes
        .div_ceil(max_page_bytes)
        .min(num_rows)
        .min(num_values);
    // A page cannot split an individual value. Avoid producing one-value pages when every
    // requested partition would still exceed the target, and keep the existing jumbo-value
    // encoding path intact.
    if desired_pages <= 1 || desired_pages == num_values {
        return Ok(pages);
    }
    let target_values_per_page = num_values.div_ceil(desired_pages);

    pages
        .into_iter()
        .map(|page| split_dense_page(page, target_values_per_page))
        .collect::<Result<Vec<_>>>()
        .map(|pages| pages.into_iter().flatten().collect())
}

fn split_dense_page(
    page: PrimitivePageData,
    target_values_per_page: u64,
) -> Result<Vec<PrimitivePageData>> {
    let PrimitivePageData {
        arrays,
        structure,
        row_number,
        num_rows,
    } = page;
    let PrimitivePageStructure::Dense {
        repdef,
        single_row_miniblock_repdef_levels,
    } = structure
    else {
        return Err(Error::internal(
            "Cannot apply dense page splitting to a sparse page",
        ));
    };
    let num_values = arrays.iter().try_fold(0_u64, |total, array| {
        let array_len = u64::try_from(array.len()).map_err(|_| {
            Error::internal(format!("Array length {} does not fit in u64", array.len()))
        })?;
        total.checked_add(array_len).ok_or_else(|| {
            Error::internal(format!(
                "Page value count overflowed u64 while adding {} values",
                array_len
            ))
        })
    })?;

    if num_rows <= 1
        || num_values <= target_values_per_page
        || single_row_miniblock_repdef_levels.is_some()
        || repdef.repetition_levels.is_some()
    {
        return Ok(unsplit_dense_page(
            arrays,
            repdef,
            single_row_miniblock_repdef_levels,
            row_number,
            num_rows,
        ));
    }

    split_non_repeated_page(
        arrays,
        repdef,
        row_number,
        num_rows,
        num_values,
        target_values_per_page,
    )
}

fn split_non_repeated_page(
    arrays: Vec<ArrayRef>,
    repdef: SerializedRepDefs,
    row_number: u64,
    num_rows: u64,
    num_values: u64,
    target_values_per_page: u64,
) -> Result<Vec<PrimitivePageData>> {
    // Without repetition, primitive values are either one-per-row or a fixed number per row.
    // If that invariant does not hold then there is no safe row boundary to infer here.
    if !num_values.is_multiple_of(num_rows) {
        return Ok(unsplit_dense_page(
            arrays, repdef, None, row_number, num_rows,
        ));
    }

    let values_per_row = num_values / num_rows;
    if values_per_row == 0 {
        return Ok(unsplit_dense_page(
            arrays, repdef, None, row_number, num_rows,
        ));
    }
    let rows_per_page = (target_values_per_page / values_per_row).max(1);
    if rows_per_page >= num_rows {
        return Ok(unsplit_dense_page(
            arrays, repdef, None, row_number, num_rows,
        ));
    }

    let page_capacity = usize::try_from(num_rows.div_ceil(rows_per_page))
        .map_err(|_| Error::internal("Page count does not fit in usize"))?;
    let mut pages = Vec::with_capacity(page_capacity);
    let mut row_start = 0_u64;
    while row_start < num_rows {
        let page_num_rows = rows_per_page.min(num_rows - row_start);
        let value_start = row_start.checked_mul(values_per_row).ok_or_else(|| {
            Error::internal(format!(
                "Page value offset overflowed u64 for row {} with {} values per row",
                row_start, values_per_row
            ))
        })?;
        let page_num_values = page_num_rows.checked_mul(values_per_row).ok_or_else(|| {
            Error::internal(format!(
                "Page value count overflowed u64 for {} rows with {} values per row",
                page_num_rows, values_per_row
            ))
        })?;
        let level_start = usize::try_from(value_start).map_err(|_| {
            Error::internal(format!(
                "Page level offset {} does not fit in usize",
                value_start
            ))
        })?;
        let value_end = value_start.checked_add(page_num_values).ok_or_else(|| {
            Error::internal(format!(
                "Page value end overflowed u64 for start {} and count {}",
                value_start, page_num_values
            ))
        })?;
        let level_end = usize::try_from(value_end).map_err(|_| {
            Error::internal(format!(
                "Page level end {} does not fit in usize",
                value_end
            ))
        })?;
        let page_row_number = row_number.checked_add(row_start).ok_or_else(|| {
            Error::internal(format!(
                "Page row number overflowed u64 for start {} and offset {}",
                row_number, row_start
            ))
        })?;

        pages.push(PrimitivePageData {
            arrays: PrimitiveStructuralEncoder::slice_arrays(
                &arrays,
                value_start,
                page_num_values,
            )?,
            structure: PrimitivePageStructure::Dense {
                repdef: PrimitiveStructuralEncoder::slice_repdef(&repdef, level_start..level_end),
                single_row_miniblock_repdef_levels: None,
            },
            row_number: page_row_number,
            num_rows: page_num_rows,
        });
        row_start = row_start.checked_add(page_num_rows).ok_or_else(|| {
            Error::internal("Row offset overflowed u64 while splitting a primitive page")
        })?;
    }
    Ok(pages)
}

fn unsplit_dense_page(
    arrays: Vec<ArrayRef>,
    repdef: SerializedRepDefs,
    single_row_miniblock_repdef_levels: Option<u64>,
    row_number: u64,
    num_rows: u64,
) -> Vec<PrimitivePageData> {
    vec![PrimitivePageData {
        arrays,
        structure: PrimitivePageStructure::Dense {
            repdef,
            single_row_miniblock_repdef_levels,
        },
        row_number,
        num_rows,
    }]
}
