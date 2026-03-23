// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for the Lance C API.
//!
//! These tests call the `extern "C"` functions directly from Rust,
//! validating the C API contract without needing a C compiler.

use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use arrow::ffi::from_ffi;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatchReader;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance_c::*;

/// Helper: create a test dataset in a temp directory and return its path.
fn create_test_dataset() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().join("test_ds").to_str().unwrap().to_string();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "alice", "bob", "carol", "dave", "eve",
            ])),
        ],
    )
    .unwrap();

    // Use lance-c's internal runtime to write the dataset.
    lance_c::runtime::block_on(async {
        Dataset::write(
            arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)], schema),
            &uri,
            None,
        )
        .await
        .unwrap();
    });

    (tmp, uri)
}

fn c_str(s: &str) -> CString {
    CString::new(s).unwrap()
}

// ---------------------------------------------------------------------------
// Dataset tests
// ---------------------------------------------------------------------------

#[test]
fn test_open_close() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);

    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null(), "dataset open should succeed");
    assert_eq!(lance_last_error_code(), LanceErrorCode::Ok);

    unsafe { lance_dataset_close(ds) };

    // Closing NULL is safe.
    unsafe { lance_dataset_close(ptr::null_mut()) };
}

#[test]
fn test_open_nonexistent() {
    let c_uri = c_str("memory://nonexistent_dataset_xyz");
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(
        ds.is_null(),
        "opening nonexistent dataset should return NULL"
    );
    assert_ne!(lance_last_error_code(), LanceErrorCode::Ok);

    let msg = lance_last_error_message();
    assert!(!msg.is_null());
    unsafe { lance_free_string(msg) };
}

#[test]
fn test_version() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let version = unsafe { lance_dataset_version(ds) };
    assert!(version >= 1, "version should be >= 1, got {version}");

    unsafe { lance_dataset_close(ds) };
}

#[test]
fn test_count_rows() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let count = unsafe { lance_dataset_count_rows(ds) };
    assert_eq!(count, 5);

    unsafe { lance_dataset_close(ds) };
}

#[test]
fn test_schema_export() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let mut ffi_schema = FFI_ArrowSchema::empty();
    let rc = unsafe { lance_dataset_schema(ds, &mut ffi_schema) };
    assert_eq!(rc, 0);

    // Import the schema back and verify fields.
    let schema = Schema::try_from(&ffi_schema).unwrap();
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "name");

    unsafe { lance_dataset_close(ds) };
}

// ---------------------------------------------------------------------------
// Scanner tests
// ---------------------------------------------------------------------------

#[test]
fn test_scanner_full_scan() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    // Create scanner (all columns, no filter).
    let scanner = unsafe { lance_scanner_new(ds, ptr::null(), ptr::null()) };
    assert!(!scanner.is_null());

    // Iterate via lance_scanner_next.
    let mut total_rows = 0u64;
    loop {
        let mut batch: *mut LanceBatch = ptr::null_mut();
        let rc = unsafe { lance_scanner_next(scanner, &mut batch) };
        match rc {
            0 => {
                assert!(!batch.is_null());
                // Export to Arrow and count rows.
                let mut ffi_array = arrow::ffi::FFI_ArrowArray::empty();
                let mut ffi_schema = FFI_ArrowSchema::empty();
                let rc2 = unsafe { lance_batch_to_arrow(batch, &mut ffi_array, &mut ffi_schema) };
                assert_eq!(rc2, 0);
                let data = unsafe { from_ffi(ffi_array, &ffi_schema) }.unwrap();
                total_rows += data.len() as u64;
                unsafe { lance_batch_free(batch) };
            }
            1 => break, // end of stream
            _ => panic!("scanner_next returned error: {rc}"),
        }
    }
    assert_eq!(total_rows, 5);

    unsafe { lance_scanner_close(scanner) };
    unsafe { lance_dataset_close(ds) };
}

#[test]
fn test_scanner_to_arrow_stream() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let scanner = unsafe { lance_scanner_new(ds, ptr::null(), ptr::null()) };
    assert!(!scanner.is_null());

    let mut ffi_stream = FFI_ArrowArrayStream::empty();
    let rc = unsafe { lance_scanner_to_arrow_stream(scanner, &mut ffi_stream) };
    assert_eq!(rc, 0);

    // Read via Arrow's standard stream reader.
    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut ffi_stream) }.unwrap();
    let batches: Vec<RecordBatch> = reader.map(|r| r.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 5);

    unsafe { lance_scanner_close(scanner) };
    unsafe { lance_dataset_close(ds) };
}

#[test]
fn test_scanner_with_filter() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let filter = c_str("id > 3");
    let scanner = unsafe { lance_scanner_new(ds, ptr::null(), filter.as_ptr()) };
    assert!(!scanner.is_null());

    let mut ffi_stream = FFI_ArrowArrayStream::empty();
    let rc = unsafe { lance_scanner_to_arrow_stream(scanner, &mut ffi_stream) };
    assert_eq!(rc, 0);

    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut ffi_stream) }.unwrap();
    let total_rows: usize = reader.map(|r| r.unwrap().num_rows()).sum();
    assert_eq!(total_rows, 2); // id=4 and id=5

    unsafe { lance_scanner_close(scanner) };
    unsafe { lance_dataset_close(ds) };
}

#[test]
fn test_scanner_with_projection() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    // Project only "name" column.
    let col = c_str("name");
    let columns: [*const i8; 2] = [col.as_ptr(), ptr::null()];
    let scanner = unsafe { lance_scanner_new(ds, columns.as_ptr(), ptr::null()) };
    assert!(!scanner.is_null());

    let mut ffi_stream = FFI_ArrowArrayStream::empty();
    let rc = unsafe { lance_scanner_to_arrow_stream(scanner, &mut ffi_stream) };
    assert_eq!(rc, 0);

    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut ffi_stream) }.unwrap();
    let schema = reader.schema();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "name");

    unsafe { lance_scanner_close(scanner) };
    unsafe { lance_dataset_close(ds) };
}

#[test]
fn test_scanner_with_limit_offset() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let scanner = unsafe { lance_scanner_new(ds, ptr::null(), ptr::null()) };
    assert!(!scanner.is_null());
    unsafe {
        lance_scanner_set_limit(scanner, 2);
        lance_scanner_set_offset(scanner, 1);
    };

    let mut ffi_stream = FFI_ArrowArrayStream::empty();
    let rc = unsafe { lance_scanner_to_arrow_stream(scanner, &mut ffi_stream) };
    assert_eq!(rc, 0);

    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut ffi_stream) }.unwrap();
    let total_rows: usize = reader.map(|r| r.unwrap().num_rows()).sum();
    assert_eq!(total_rows, 2);

    unsafe { lance_scanner_close(scanner) };
    unsafe { lance_dataset_close(ds) };
}

// ---------------------------------------------------------------------------
// Take test
// ---------------------------------------------------------------------------

#[test]
fn test_dataset_take() {
    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let indices: [u64; 3] = [0, 2, 4];
    let mut ffi_stream = FFI_ArrowArrayStream::empty();
    let rc = unsafe { lance_dataset_take(ds, indices.as_ptr(), 3, ptr::null(), &mut ffi_stream) };
    assert_eq!(rc, 0);

    let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut ffi_stream) }.unwrap();
    let batches: Vec<RecordBatch> = reader.map(|r| r.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);

    // Verify the taken IDs.
    let id_col = batches[0]
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(id_col.values(), &[1, 3, 5]);

    unsafe { lance_dataset_close(ds) };
}

// ---------------------------------------------------------------------------
// Error handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_null_inputs() {
    // NULL dataset in version query.
    let v = unsafe { lance_dataset_version(ptr::null()) };
    assert_eq!(v, 0);
    assert_ne!(lance_last_error_code(), LanceErrorCode::Ok);

    // NULL dataset in scanner creation.
    let scanner = unsafe { lance_scanner_new(ptr::null(), ptr::null(), ptr::null()) };
    assert!(scanner.is_null());
    assert_ne!(lance_last_error_code(), LanceErrorCode::Ok);
}

// ---------------------------------------------------------------------------
// Async scan test
// ---------------------------------------------------------------------------

#[test]
fn test_scanner_scan_async() {
    use std::sync::{Condvar, Mutex};

    let (_tmp, uri) = create_test_dataset();
    let c_uri = c_str(&uri);
    let ds = unsafe { lance_dataset_open(c_uri.as_ptr(), ptr::null(), 0) };
    assert!(!ds.is_null());

    let scanner = unsafe { lance_scanner_new(ds, ptr::null(), ptr::null()) };
    assert!(!scanner.is_null());

    // Synchronization primitive for the async callback.
    struct CallbackResult {
        status: i32,
        stream_ptr: *mut std::ffi::c_void,
    }
    unsafe impl Send for CallbackResult {}

    let pair = Arc::new((Mutex::new(None::<CallbackResult>), Condvar::new()));
    let pair_clone = pair.clone();

    unsafe extern "C" fn on_complete(
        ctx: *mut std::ffi::c_void,
        status: i32,
        result: *mut std::ffi::c_void,
    ) {
        let pair = unsafe { &*(ctx as *const (Mutex<Option<CallbackResult>>, Condvar)) };
        let mut guard = pair.0.lock().unwrap();
        *guard = Some(CallbackResult {
            status,
            stream_ptr: result,
        });
        pair.1.notify_one();
    }

    unsafe {
        lance_scanner_scan_async(
            scanner,
            on_complete,
            Arc::as_ptr(&pair_clone) as *mut std::ffi::c_void,
        );
    }

    // Wait for callback.
    let (lock, cvar) = &*pair;
    let guard = cvar
        .wait_while(lock.lock().unwrap(), |r| r.is_none())
        .unwrap();
    let result = guard.as_ref().unwrap();
    assert_eq!(result.status, 0, "async scan should succeed");
    assert!(!result.stream_ptr.is_null());

    // Read the stream.
    let ffi_stream = unsafe { &mut *(result.stream_ptr as *mut FFI_ArrowArrayStream) };
    let reader = unsafe { ArrowArrayStreamReader::from_raw(ffi_stream) }.unwrap();
    let total_rows: usize = reader.map(|r| r.unwrap().num_rows()).sum();
    assert_eq!(total_rows, 5);

    unsafe { lance_scanner_close(scanner) };
    unsafe { lance_dataset_close(ds) };
}
