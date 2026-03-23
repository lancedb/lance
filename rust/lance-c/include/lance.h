/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: Copyright The Lance Authors */

/**
 * @file lance.h
 * @brief C API for the Lance columnar data format.
 *
 * All data crosses this boundary via the Arrow C Data Interface
 * (ArrowSchema, ArrowArray, ArrowArrayStream).
 *
 * Error handling uses thread-local storage: after any function returns
 * NULL (pointer) or -1 (int), call lance_last_error_code() and
 * lance_last_error_message() to get details.
 */

#ifndef LANCE_H
#define LANCE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Arrow C Data Interface forward declarations ─── */
/* These match the canonical Arrow spec structs. If you already include
   arrow/c/abi.h, guard with ARROW_C_DATA_INTERFACE. */

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;
    void (*release)(struct ArrowSchema*);
    void* private_data;
};

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;
    void (*release)(struct ArrowArray*);
    void* private_data;
};

struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
    const char* (*get_last_error)(struct ArrowArrayStream*);
    void (*release)(struct ArrowArrayStream*);
    void* private_data;
};

#endif /* ARROW_C_DATA_INTERFACE */

/* ─── Error handling ─── */

typedef enum {
    LANCE_OK = 0,
    LANCE_ERR_INVALID_ARGUMENT = 1,
    LANCE_ERR_IO = 2,
    LANCE_ERR_NOT_FOUND = 3,
    LANCE_ERR_DATASET_ALREADY_EXISTS = 4,
    LANCE_ERR_INDEX = 5,
    LANCE_ERR_INTERNAL = 6,
    LANCE_ERR_NOT_SUPPORTED = 7,
    LANCE_ERR_COMMIT_CONFLICT = 8,
} LanceErrorCode;

/** Return the error code from the last failed operation on this thread. */
LanceErrorCode lance_last_error_code(void);

/** Return the error message. Caller must free with lance_free_string(). */
const char* lance_last_error_message(void);

/** Free a string returned by lance_last_error_message(). */
void lance_free_string(const char* s);

/* ─── Opaque handles ─── */

typedef struct LanceDataset LanceDataset;
typedef struct LanceScanner  LanceScanner;
typedef struct LanceBatch    LanceBatch;

/* ─── Dataset lifecycle ─── */

/**
 * Open a Lance dataset.
 *
 * @param uri           Dataset path (file://, s3://, memory://, etc.)
 * @param storage_opts  NULL-terminated key-value pairs ["k1","v1",NULL], or NULL
 * @param version       Version to open (0 = latest)
 * @return Dataset handle, or NULL on error
 */
LanceDataset* lance_dataset_open(
    const char* uri,
    const char* const* storage_opts,
    uint64_t version
);

/** Close and free a dataset handle. Safe to call with NULL. */
void lance_dataset_close(LanceDataset* dataset);

/* ─── Dataset metadata (sync, in-memory) ─── */

/** Return the version number of this dataset snapshot. */
uint64_t lance_dataset_version(const LanceDataset* dataset);

/** Return the number of rows. Returns 0 on error. */
uint64_t lance_dataset_count_rows(const LanceDataset* dataset);

/** Return the latest version ID (I/O). Returns 0 on error. */
uint64_t lance_dataset_latest_version(const LanceDataset* dataset);

/**
 * Export the dataset schema via Arrow C Data Interface.
 * @param out  Pointer to caller-allocated ArrowSchema struct
 * @return 0 on success, -1 on error
 */
int32_t lance_dataset_schema(
    const LanceDataset* dataset,
    struct ArrowSchema* out
);

/* ─── Random access ─── */

/**
 * Take rows by indices.
 * @param indices      Array of 0-based row offsets
 * @param num_indices  Length of indices array
 * @param columns      NULL-terminated column names, or NULL for all
 * @param out          Pointer to caller-allocated ArrowArrayStream
 * @return 0 on success, -1 on error
 */
int32_t lance_dataset_take(
    const LanceDataset* dataset,
    const uint64_t* indices,
    size_t num_indices,
    const char* const* columns,
    struct ArrowArrayStream* out
);

/* ─── Scanner builder ─── */

/**
 * Create a scanner for the dataset.
 * @param dataset  Open dataset (not consumed)
 * @param columns  NULL-terminated column names, or NULL for all
 * @param filter   SQL filter expression, or NULL
 * @return Scanner handle, or NULL on error
 */
LanceScanner* lance_scanner_new(
    const LanceDataset* dataset,
    const char* const* columns,
    const char* filter
);

int32_t lance_scanner_set_limit(LanceScanner* scanner, int64_t limit);
int32_t lance_scanner_set_offset(LanceScanner* scanner, int64_t offset);
int32_t lance_scanner_set_batch_size(LanceScanner* scanner, int64_t batch_size);
int32_t lance_scanner_with_row_id(LanceScanner* scanner, bool enable);

/** Close and free a scanner handle. */
void lance_scanner_close(LanceScanner* scanner);

/* ─── Sync scan: ArrowArrayStream ─── */

/**
 * Materialize the scan as an ArrowArrayStream (blocking).
 * @return 0 on success, -1 on error
 */
int32_t lance_scanner_to_arrow_stream(
    LanceScanner* scanner,
    struct ArrowArrayStream* out
);

/* ─── Sync scan: batch iteration ─── */

/**
 * Read the next batch (blocking).
 * @param out  Set to a LanceBatch* on success, NULL on end/error
 * @return 0 = batch available, 1 = end of stream, -1 = error
 */
int32_t lance_scanner_next(
    LanceScanner* scanner,
    LanceBatch** out
);

/* ─── Async scan: callback-based ─── */

/**
 * Callback type for async operations.
 * @param ctx     Opaque pointer passed back from the caller
 * @param status  0 = success, -1 = error
 * @param result  Operation-specific result (e.g., ArrowArrayStream*)
 */
typedef void (*LanceCallback)(void* ctx, int32_t status, void* result);

/**
 * Start an async scan. The callback fires on a dedicated dispatcher thread
 * when the ArrowArrayStream is ready.
 */
void lance_scanner_scan_async(
    const LanceScanner* scanner,
    LanceCallback callback,
    void* callback_ctx
);

/* ─── Poll-based scan (for cooperative async runtimes) ─── */

typedef enum {
    LANCE_POLL_READY    =  0,
    LANCE_POLL_PENDING  =  1,
    LANCE_POLL_FINISHED =  2,
    LANCE_POLL_ERROR    = -1,
} LancePollStatus;

/** Waker callback: called from a Tokio thread when data is ready. */
typedef void (*LanceWaker)(void* ctx);

/**
 * Poll for the next batch without blocking.
 * See RFC for usage pattern.
 */
LancePollStatus lance_scanner_poll_next(
    LanceScanner* scanner,
    LanceWaker waker,
    void* waker_ctx,
    LanceBatch** out
);

/* ─── Batch (Arrow C Data Interface) ─── */

/**
 * Export a batch as Arrow C Data Interface structs.
 * @return 0 on success, -1 on error
 */
int32_t lance_batch_to_arrow(
    const LanceBatch* batch,
    struct ArrowArray* out_array,
    struct ArrowSchema* out_schema
);

/** Free a batch handle. */
void lance_batch_free(LanceBatch* batch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LANCE_H */
