// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset C API: open, close, metadata, schema, take.

use std::ffi::c_char;
use std::sync::Arc;

use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::Schema as ArrowSchema;
use lance::dataset::builder::DatasetBuilder;
use lance::Dataset;
use lance_core::Result;

use crate::error::{ffi_try, set_last_error, LanceErrorCode};
use crate::helpers;
use crate::runtime::block_on;

/// Opaque handle representing an opened Lance dataset.
pub struct LanceDataset {
    pub(crate) inner: Arc<Dataset>,
}

// ---------------------------------------------------------------------------
// Dataset lifecycle
// ---------------------------------------------------------------------------

/// Open a Lance dataset at the given URI.
///
/// - `uri`: Dataset path (file://, s3://, az://, gs://, memory://)
/// - `storage_options`: NULL-terminated key-value pairs `["k1","v1","k2","v2",NULL]`, or NULL.
/// - `version`: Dataset version to open. Pass 0 for latest.
///
/// Returns an opaque `LanceDataset*` on success, or NULL on error.
/// On error, call `lance_last_error_code()` / `lance_last_error_message()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_open(
    uri: *const c_char,
    storage_options: *const *const c_char,
    version: u64,
) -> *mut LanceDataset {
    ffi_try!(
        unsafe { open_dataset_inner(uri, storage_options, version) },
        null
    )
}

unsafe fn open_dataset_inner(
    uri: *const c_char,
    storage_options: *const *const c_char,
    version: u64,
) -> Result<*mut LanceDataset> {
    let uri_str = unsafe { helpers::parse_c_string(uri)? }.ok_or_else(|| {
        lance_core::Error::InvalidInput {
            source: "uri must not be NULL".into(),
            location: snafu::location!(),
        }
    })?;

    let opts = unsafe { helpers::parse_storage_options(storage_options)? };

    let mut builder = DatasetBuilder::from_uri(uri_str);
    if !opts.is_empty() {
        builder = builder.with_storage_options(opts);
    }
    if version != 0 {
        builder = builder.with_version(version);
    }

    let dataset = block_on(builder.load())?;
    let handle = LanceDataset {
        inner: Arc::new(dataset),
    };
    Ok(Box::into_raw(Box::new(handle)))
}

/// Close and free a dataset handle.
/// Safe to call with NULL. Safe to call multiple times (subsequent calls are no-ops).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_close(dataset: *mut LanceDataset) {
    if !dataset.is_null() {
        unsafe {
            let _ = Box::from_raw(dataset);
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata (in-memory, sync only)
// ---------------------------------------------------------------------------

/// Return the version number of this dataset snapshot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_version(dataset: *const LanceDataset) -> u64 {
    if dataset.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "dataset is NULL");
        return 0;
    }
    let ds = unsafe { &*dataset };
    ds.inner.version().version
}

/// Return the number of rows in the dataset.
/// Returns 0 on error — check `lance_last_error_code()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_count_rows(dataset: *const LanceDataset) -> u64 {
    if dataset.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "dataset is NULL");
        return 0;
    }
    let ds = unsafe { &*dataset };
    match block_on(ds.inner.count_rows(None)) {
        Ok(n) => {
            crate::error::clear_last_error();
            n as u64
        }
        Err(err) => {
            crate::error::set_lance_error(&err);
            0
        }
    }
}

/// Return the latest version ID of the dataset.
/// Returns 0 on error — check `lance_last_error_code()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_latest_version(dataset: *const LanceDataset) -> u64 {
    if dataset.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "dataset is NULL");
        return 0;
    }
    let ds = unsafe { &*dataset };
    match block_on(ds.inner.latest_version_id()) {
        Ok(v) => {
            crate::error::clear_last_error();
            v
        }
        Err(err) => {
            crate::error::set_lance_error(&err);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Schema (Arrow C Data Interface)
// ---------------------------------------------------------------------------

/// Export the dataset schema as an Arrow C Data Interface `ArrowSchema`.
///
/// The caller must provide a pointer to a stack-allocated `ArrowSchema` struct.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_schema(
    dataset: *const LanceDataset,
    out: *mut FFI_ArrowSchema,
) -> i32 {
    ffi_try!(unsafe { dataset_schema_inner(dataset, out) }, neg)
}

unsafe fn dataset_schema_inner(
    dataset: *const LanceDataset,
    out: *mut FFI_ArrowSchema,
) -> Result<i32> {
    if dataset.is_null() || out.is_null() {
        return Err(lance_core::Error::InvalidInput {
            source: "dataset and out must not be NULL".into(),
            location: snafu::location!(),
        });
    }
    let ds = unsafe { &*dataset };
    let lance_schema = ds.inner.schema();
    let arrow_schema: ArrowSchema = lance_schema.into();
    let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
    unsafe {
        std::ptr::write_unaligned(out, ffi_schema);
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// Random access (take)
// ---------------------------------------------------------------------------

/// Take rows by indices, returning results as an ArrowArrayStream.
///
/// - `indices`: array of row indices (0-based offsets)
/// - `num_indices`: length of the indices array
/// - `columns`: NULL-terminated column name array, or NULL for all columns
/// - `out`: pointer to a stack-allocated `ArrowArrayStream`
///
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_dataset_take(
    dataset: *const LanceDataset,
    indices: *const u64,
    num_indices: usize,
    columns: *const *const c_char,
    out: *mut FFI_ArrowArrayStream,
) -> i32 {
    ffi_try!(
        unsafe { dataset_take_inner(dataset, indices, num_indices, columns, out) },
        neg
    )
}

unsafe fn dataset_take_inner(
    dataset: *const LanceDataset,
    indices: *const u64,
    num_indices: usize,
    columns: *const *const c_char,
    out: *mut FFI_ArrowArrayStream,
) -> Result<i32> {
    if dataset.is_null() || indices.is_null() || out.is_null() {
        return Err(lance_core::Error::InvalidInput {
            source: "dataset, indices, and out must not be NULL".into(),
            location: snafu::location!(),
        });
    }
    let ds = unsafe { &*dataset };
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, num_indices) };
    let col_names = unsafe { helpers::parse_c_string_array(columns)? };

    let projection = match &col_names {
        Some(cols) => {
            lance::dataset::ProjectionRequest::from_columns(cols.iter(), ds.inner.schema())
        }
        None => lance::dataset::ProjectionRequest::from_schema(ds.inner.schema().clone()),
    };

    let batch = block_on(ds.inner.take(idx_slice, projection))?;

    // Wrap the single RecordBatch as a RecordBatchReader, then export as FFI stream.
    let schema = batch.schema();
    let reader = arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)], schema);
    let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
    unsafe {
        std::ptr::write_unaligned(out, ffi_stream);
    }
    Ok(0)
}
