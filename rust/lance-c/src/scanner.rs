// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scanner C API: builder, sync iteration, async scan, poll-based iteration.

use std::ffi::{c_char, c_void};
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::SchemaRef;
use futures::{Stream, StreamExt};
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance::Dataset;
use lance_core::Result;
use lance_io::ffi::to_ffi_arrow_array_stream;
use lance_io::stream::RecordBatchStream;

use crate::async_dispatcher::{self, LanceCallback};
use crate::batch::LanceBatch;
use crate::dataset::LanceDataset;
use crate::error::{clear_last_error, ffi_try, set_lance_error, set_last_error, LanceErrorCode};
use crate::helpers;
use crate::runtime::{block_on, RT};

/// Opaque scanner handle. Stores configuration until stream materialization.
pub struct LanceScanner {
    dataset: Arc<Dataset>,
    columns: Option<Vec<String>>,
    filter: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
    batch_size: Option<usize>,
    with_row_id: bool,
    // Materialized on first iteration call
    stream: Option<Pin<Box<DatasetRecordBatchStream>>>,
    #[allow(dead_code)]
    schema: Option<SchemaRef>,
}

/// Poll status for `lance_scanner_poll_next`.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub enum LancePollStatus {
    /// Batch available in `*out`.
    Ready = 0,
    /// I/O in progress; waker will fire when ready.
    Pending = 1,
    /// End of stream.
    Finished = 2,
    /// Error occurred (check `lance_last_error_*`).
    Error = -1,
}

/// Waker callback type for poll-based iteration.
/// Called from a Tokio I/O thread when data becomes available.
/// Must be thread-safe and must NOT call back into `lance_scanner_*`.
pub type LanceWaker = unsafe extern "C" fn(ctx: *mut c_void);

impl LanceScanner {
    fn new(dataset: Arc<Dataset>) -> Self {
        Self {
            dataset,
            columns: None,
            filter: None,
            limit: None,
            offset: None,
            batch_size: None,
            with_row_id: false,
            stream: None,
            schema: None,
        }
    }

    /// Build the underlying Scanner and open a stream.
    fn materialize_stream(&mut self) -> Result<()> {
        let mut scanner = self.dataset.scan();
        if let Some(cols) = &self.columns {
            scanner.project(cols)?;
        }
        if let Some(filter) = &self.filter {
            scanner.filter(filter)?;
        }
        if self.limit.is_some() || self.offset.is_some() {
            scanner.limit(self.limit, self.offset)?;
        }
        if let Some(bs) = self.batch_size {
            scanner.batch_size(bs);
        }
        if self.with_row_id {
            scanner.with_row_id();
        }
        let stream = block_on(scanner.try_into_stream())?;
        self.schema = Some(stream.schema());
        self.stream = Some(Box::pin(stream));
        Ok(())
    }

    /// Build a Scanner (without materializing) and return it.
    fn build_scanner(&self) -> Result<lance::dataset::scanner::Scanner> {
        let mut scanner = self.dataset.scan();
        if let Some(cols) = &self.columns {
            scanner.project(cols)?;
        }
        if let Some(filter) = &self.filter {
            scanner.filter(filter)?;
        }
        if self.limit.is_some() || self.offset.is_some() {
            scanner.limit(self.limit, self.offset)?;
        }
        if let Some(bs) = self.batch_size {
            scanner.batch_size(bs);
        }
        if self.with_row_id {
            scanner.with_row_id();
        }
        Ok(scanner)
    }
}

// ---------------------------------------------------------------------------
// Scanner lifecycle + builder
// ---------------------------------------------------------------------------

/// Create a new scanner for the given dataset.
///
/// - `dataset`: An open `LanceDataset*` (not consumed; remains valid).
/// - `columns`: NULL-terminated column name array, or NULL for all columns.
/// - `filter`: SQL filter expression, or NULL for no filter.
///
/// Returns a `LanceScanner*` on success, or NULL on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_new(
    dataset: *const LanceDataset,
    columns: *const *const c_char,
    filter: *const c_char,
) -> *mut LanceScanner {
    ffi_try!(unsafe { scanner_new_inner(dataset, columns, filter) }, null)
}

unsafe fn scanner_new_inner(
    dataset: *const LanceDataset,
    columns: *const *const c_char,
    filter: *const c_char,
) -> Result<*mut LanceScanner> {
    if dataset.is_null() {
        return Err(lance_core::Error::InvalidInput {
            source: "dataset must not be NULL".into(),
            location: snafu::location!(),
        });
    }
    let ds = unsafe { &*dataset };
    let col_names = unsafe { helpers::parse_c_string_array(columns)? };
    let filter_str = unsafe { helpers::parse_c_string(filter)? }.map(|s| s.to_string());

    let mut scanner = LanceScanner::new(ds.inner.clone());
    scanner.columns = col_names;
    scanner.filter = filter_str;
    Ok(Box::into_raw(Box::new(scanner)))
}

/// Set the row limit on the scanner. Returns 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_set_limit(scanner: *mut LanceScanner, limit: i64) -> i32 {
    if scanner.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "scanner is NULL");
        return -1;
    }
    let s = unsafe { &mut *scanner };
    s.limit = Some(limit);
    clear_last_error();
    0
}

/// Set the row offset on the scanner. Returns 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_set_offset(scanner: *mut LanceScanner, offset: i64) -> i32 {
    if scanner.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "scanner is NULL");
        return -1;
    }
    let s = unsafe { &mut *scanner };
    s.offset = Some(offset);
    clear_last_error();
    0
}

/// Set the batch size on the scanner. Returns 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_set_batch_size(
    scanner: *mut LanceScanner,
    batch_size: i64,
) -> i32 {
    if scanner.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "scanner is NULL");
        return -1;
    }
    let s = unsafe { &mut *scanner };
    s.batch_size = Some(batch_size as usize);
    clear_last_error();
    0
}

/// Enable or disable row ID in scan output. Returns 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_with_row_id(
    scanner: *mut LanceScanner,
    enable: bool,
) -> i32 {
    if scanner.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "scanner is NULL");
        return -1;
    }
    let s = unsafe { &mut *scanner };
    s.with_row_id = enable;
    clear_last_error();
    0
}

/// Close and free a scanner handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_close(scanner: *mut LanceScanner) {
    if !scanner.is_null() {
        unsafe {
            let _ = Box::from_raw(scanner);
        }
    }
}

// ---------------------------------------------------------------------------
// Sync stream: ArrowArrayStream export
// ---------------------------------------------------------------------------

/// Materialize the scan as an Arrow C Data Interface `ArrowArrayStream`.
///
/// This is the preferred API for simple integrations — blocks the calling thread.
/// The scanner is consumed by this call and should not be used afterward (close it).
///
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_to_arrow_stream(
    scanner: *mut LanceScanner,
    out: *mut FFI_ArrowArrayStream,
) -> i32 {
    ffi_try!(unsafe { scanner_to_arrow_stream_inner(scanner, out) }, neg)
}

unsafe fn scanner_to_arrow_stream_inner(
    scanner: *mut LanceScanner,
    out: *mut FFI_ArrowArrayStream,
) -> Result<i32> {
    if scanner.is_null() || out.is_null() {
        return Err(lance_core::Error::InvalidInput {
            source: "scanner and out must not be NULL".into(),
            location: snafu::location!(),
        });
    }
    let s = unsafe { &*scanner };
    let built_scanner = s.build_scanner()?;
    let stream = block_on(built_scanner.try_into_stream())?;
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone())?;
    unsafe {
        ptr::write_unaligned(out, ffi_stream);
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// Sync iteration: blocking batch-at-a-time
// ---------------------------------------------------------------------------

/// Read the next batch from the scanner (blocking).
///
/// Returns:
/// -  `0` — batch available, `*out` is set.
/// -  `1` — end of stream, `*out` is NULL.
/// - `-1` — error (check `lance_last_error_*`), `*out` is NULL.
///
/// The caller must free each returned batch with `lance_batch_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_next(
    scanner: *mut LanceScanner,
    out: *mut *mut LanceBatch,
) -> i32 {
    if scanner.is_null() || out.is_null() {
        set_last_error(
            LanceErrorCode::InvalidArgument,
            "scanner and out must not be NULL",
        );
        return -1;
    }
    let s = unsafe { &mut *scanner };

    // Lazily materialize the stream on first call.
    if s.stream.is_none() {
        if let Err(err) = s.materialize_stream() {
            set_lance_error(&err);
            unsafe { *out = ptr::null_mut() };
            return -1;
        }
    }

    let stream = s.stream.as_mut().unwrap();
    match block_on(stream.next()) {
        Some(Ok(batch)) => {
            clear_last_error();
            let lance_batch = LanceBatch { inner: batch };
            unsafe { *out = Box::into_raw(Box::new(lance_batch)) };
            0
        }
        Some(Err(err)) => {
            set_lance_error(&err);
            unsafe { *out = ptr::null_mut() };
            -1
        }
        None => {
            // End of stream
            clear_last_error();
            unsafe { *out = ptr::null_mut() };
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Async scan: callback-based
// ---------------------------------------------------------------------------

/// Start an async scan. The callback is invoked on a dedicated dispatcher thread
/// when the ArrowArrayStream is ready.
///
/// - `callback`: Called with `(ctx, 0, *mut ArrowArrayStream)` on success,
///   or `(ctx, -1, NULL)` on error (check `lance_last_error_*`).
/// - `callback_ctx`: Opaque pointer passed back to the callback.
///
/// The scanner configuration is captured at call time. The scanner handle
/// can be closed immediately after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_scan_async(
    scanner: *const LanceScanner,
    callback: LanceCallback,
    callback_ctx: *mut c_void,
) {
    if scanner.is_null() {
        set_last_error(LanceErrorCode::InvalidArgument, "scanner is NULL");
        async_dispatcher::dispatch_callback(callback, callback_ctx, -1, ptr::null_mut());
        return;
    }

    let s = unsafe { &*scanner };
    let built_scanner = match s.build_scanner() {
        Ok(sc) => sc,
        Err(err) => {
            set_lance_error(&err);
            async_dispatcher::dispatch_callback(callback, callback_ctx, -1, ptr::null_mut());
            return;
        }
    };

    let handle = RT.handle().clone();

    // Wrap non-Send raw pointers for the async task.
    // Safety: The C caller guarantees callback_ctx remains valid until callback fires.
    struct SendCallback {
        callback: LanceCallback,
        ctx: *mut c_void,
    }
    unsafe impl Send for SendCallback {}

    impl SendCallback {
        fn dispatch(&self, status: i32, result: *mut c_void) {
            async_dispatcher::dispatch_callback(self.callback, self.ctx, status, result);
        }
    }

    let send_cb = SendCallback {
        callback,
        ctx: callback_ctx,
    };

    RT.spawn(async move {
        let result = built_scanner.try_into_stream().await;
        match result {
            Ok(stream) => match to_ffi_arrow_array_stream(stream, handle) {
                Ok(ffi_stream) => {
                    let ptr = Box::into_raw(Box::new(ffi_stream));
                    send_cb.dispatch(0, ptr as *mut c_void);
                }
                Err(err) => {
                    set_lance_error(&err);
                    send_cb.dispatch(-1, std::ptr::null_mut());
                }
            },
            Err(err) => {
                set_lance_error(&err);
                send_cb.dispatch(-1, std::ptr::null_mut());
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Poll-based iteration (for cooperative async runtimes)
// ---------------------------------------------------------------------------

/// Poll for the next batch without blocking.
///
/// - If data is already buffered, returns `LANCE_POLL_READY` immediately.
/// - If I/O is needed, returns `LANCE_POLL_PENDING` and schedules the waker callback.
///   The caller should yield the thread and re-poll after the waker fires.
/// - The waker is single-use: it fires at most once per poll call that returns PENDING.
///
/// The stream is lazily materialized on the first poll call (which will typically
/// return PENDING while the stream opens).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lance_scanner_poll_next(
    scanner: *mut LanceScanner,
    waker: LanceWaker,
    waker_ctx: *mut c_void,
    out: *mut *mut LanceBatch,
) -> LancePollStatus {
    if scanner.is_null() || out.is_null() {
        set_last_error(
            LanceErrorCode::InvalidArgument,
            "scanner and out must not be NULL",
        );
        return LancePollStatus::Error;
    }
    let s = unsafe { &mut *scanner };

    // Lazily materialize the stream.
    if s.stream.is_none() {
        if let Err(err) = s.materialize_stream() {
            set_lance_error(&err);
            unsafe { *out = ptr::null_mut() };
            return LancePollStatus::Error;
        }
    }

    let stream = s.stream.as_mut().unwrap();

    // Construct a std::task::Waker from the C function pointer.
    let raw_waker = make_raw_waker(waker, waker_ctx);
    let waker_obj = unsafe { Waker::from_raw(raw_waker) };
    let mut cx = Context::from_waker(&waker_obj);

    match stream.as_mut().poll_next(&mut cx) {
        Poll::Ready(Some(Ok(batch))) => {
            clear_last_error();
            let lance_batch = LanceBatch { inner: batch };
            unsafe { *out = Box::into_raw(Box::new(lance_batch)) };
            LancePollStatus::Ready
        }
        Poll::Ready(Some(Err(err))) => {
            set_lance_error(&err);
            unsafe { *out = ptr::null_mut() };
            LancePollStatus::Error
        }
        Poll::Ready(None) => {
            clear_last_error();
            unsafe { *out = ptr::null_mut() };
            LancePollStatus::Finished
        }
        Poll::Pending => {
            clear_last_error();
            unsafe { *out = ptr::null_mut() };
            LancePollStatus::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Waker construction from C function pointer
// ---------------------------------------------------------------------------

/// Context for a C waker callback.
struct CWakerContext {
    waker_fn: LanceWaker,
    ctx: *mut c_void,
}

// C function pointers + void* are Send by convention for FFI.
unsafe impl Send for CWakerContext {}
unsafe impl Sync for CWakerContext {}

fn make_raw_waker(waker_fn: LanceWaker, ctx: *mut c_void) -> RawWaker {
    let data = Box::into_raw(Box::new(CWakerContext { waker_fn, ctx })) as *const ();

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        // clone
        |data| {
            let orig = unsafe { &*(data as *const CWakerContext) };
            let cloned = Box::new(CWakerContext {
                waker_fn: orig.waker_fn,
                ctx: orig.ctx,
            });
            RawWaker::new(Box::into_raw(cloned) as *const (), &VTABLE)
        },
        // wake (consumes)
        |data| {
            let ctx = unsafe { Box::from_raw(data as *mut CWakerContext) };
            unsafe { (ctx.waker_fn)(ctx.ctx) };
        },
        // wake_by_ref
        |data| {
            let ctx = unsafe { &*(data as *const CWakerContext) };
            unsafe { (ctx.waker_fn)(ctx.ctx) };
        },
        // drop
        |data| {
            unsafe {
                let _ = Box::from_raw(data as *mut CWakerContext);
            };
        },
    );

    RawWaker::new(data, &VTABLE)
}
