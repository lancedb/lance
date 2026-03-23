/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: Copyright The Lance Authors */

/**
 * @file lance.hpp
 * @brief C++ RAII wrappers for the Lance C API.
 *
 * Header-only library providing:
 *   - lance::Error exception class
 *   - lance::Dataset RAII handle with builder-pattern Scanner
 *   - lance::Scanner fluent API
 *   - All data exchange via Arrow C Data Interface
 */

#ifndef LANCE_HPP
#define LANCE_HPP

#include "lance.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lance {

// ─── Error ───────────────────────────────────────────────────────────────────

class Error : public std::runtime_error {
public:
    LanceErrorCode code;

    Error(LanceErrorCode code, std::string msg)
        : std::runtime_error(std::move(msg)), code(code) {}
};

/// Check thread-local error and throw if non-OK.
inline void check_error() {
    LanceErrorCode code = lance_last_error_code();
    if (code != LANCE_OK) {
        const char* msg = lance_last_error_message();
        std::string owned(msg ? msg : "Unknown error");
        if (msg) lance_free_string(msg);
        throw Error(code, std::move(owned));
    }
}

// ─── RAII Handle Template ────────────────────────────────────────────────────

template <typename T, void (*Deleter)(T*)>
class Handle {
    T* ptr_;

public:
    explicit Handle(T* ptr = nullptr) : ptr_(ptr) {}
    ~Handle() {
        if (ptr_) Deleter(ptr_);
    }

    Handle(Handle&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) {
            if (ptr_) Deleter(ptr_);
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    T* get() const { return ptr_; }
    T* release() {
        auto p = ptr_;
        ptr_ = nullptr;
        return p;
    }
    explicit operator bool() const { return ptr_ != nullptr; }
};

// ─── Forward Declarations ────────────────────────────────────────────────────

class Scanner;

// ─── Dataset ─────────────────────────────────────────────────────────────────

class Dataset {
    Handle<LanceDataset, lance_dataset_close> handle_;

public:
    /// Open a dataset at the given URI.
    static Dataset open(
        const std::string& uri,
        const std::vector<std::pair<std::string, std::string>>& storage_opts = {},
        uint64_t version = 0) {

        // Build NULL-terminated key-value array for storage options.
        std::vector<const char*> kv;
        for (auto& [k, v] : storage_opts) {
            kv.push_back(k.c_str());
            kv.push_back(v.c_str());
        }
        kv.push_back(nullptr);

        const char* const* opts_ptr =
            storage_opts.empty() ? nullptr : kv.data();

        auto* ds = lance_dataset_open(uri.c_str(), opts_ptr, version);
        if (!ds) check_error();
        return Dataset(ds);
    }

    /// Number of rows in the dataset.
    uint64_t count_rows() const {
        uint64_t n = lance_dataset_count_rows(handle_.get());
        if (lance_last_error_code() != LANCE_OK) check_error();
        return n;
    }

    /// Version of this dataset snapshot.
    uint64_t version() const {
        return lance_dataset_version(handle_.get());
    }

    /// Latest version ID (queries object store).
    uint64_t latest_version() const {
        uint64_t v = lance_dataset_latest_version(handle_.get());
        if (lance_last_error_code() != LANCE_OK) check_error();
        return v;
    }

    /// Export the schema as an Arrow C Data Interface struct.
    void schema(ArrowSchema* out) const {
        if (lance_dataset_schema(handle_.get(), out) != 0) {
            check_error();
        }
    }

    /// Take rows by indices. Results exported as ArrowArrayStream.
    void take(const uint64_t* indices, size_t num_indices,
              const std::vector<std::string>& columns,
              ArrowArrayStream* out) const {
        std::vector<const char*> col_ptrs;
        for (auto& c : columns) col_ptrs.push_back(c.c_str());
        col_ptrs.push_back(nullptr);
        const char* const* cols_ptr = columns.empty() ? nullptr : col_ptrs.data();

        if (lance_dataset_take(handle_.get(), indices, num_indices, cols_ptr, out) != 0) {
            check_error();
        }
    }

    /// Take all columns.
    void take(const uint64_t* indices, size_t num_indices,
              ArrowArrayStream* out) const {
        if (lance_dataset_take(handle_.get(), indices, num_indices, nullptr, out) != 0) {
            check_error();
        }
    }

    /// Create a Scanner builder for this dataset.
    Scanner scan() const;

    /// Access the underlying C handle (does not transfer ownership).
    const LanceDataset* c_handle() const { return handle_.get(); }

private:
    explicit Dataset(LanceDataset* ptr) : handle_(ptr) {}
};

// ─── Scanner ─────────────────────────────────────────────────────────────────

class Scanner {
    Handle<LanceScanner, lance_scanner_close> handle_;

public:
    explicit Scanner(LanceScanner* s) : handle_(s) {}

    /// Set the row limit.
    Scanner& limit(int64_t n) {
        if (lance_scanner_set_limit(handle_.get(), n) != 0)
            check_error();
        return *this;
    }

    /// Set the row offset.
    Scanner& offset(int64_t n) {
        if (lance_scanner_set_offset(handle_.get(), n) != 0)
            check_error();
        return *this;
    }

    /// Set the batch size.
    Scanner& batch_size(int64_t n) {
        if (lance_scanner_set_batch_size(handle_.get(), n) != 0)
            check_error();
        return *this;
    }

    /// Enable/disable row ID in output.
    Scanner& with_row_id(bool enable = true) {
        if (lance_scanner_with_row_id(handle_.get(), enable) != 0)
            check_error();
        return *this;
    }

    /// Materialize the scan as an ArrowArrayStream (blocking).
    void to_arrow_stream(ArrowArrayStream* out) {
        if (lance_scanner_to_arrow_stream(handle_.get(), out) != 0)
            check_error();
    }

    /// Start an async scan. Callback fires when ArrowArrayStream is ready.
    void scan_async(LanceCallback callback, void* ctx) const {
        lance_scanner_scan_async(handle_.get(), callback, ctx);
    }

    /// Access the underlying C handle.
    LanceScanner* c_handle() { return handle_.get(); }
};

inline Scanner Dataset::scan() const {
    auto* s = lance_scanner_new(handle_.get(), nullptr, nullptr);
    if (!s) check_error();
    return Scanner(s);
}

// ─── Batch ───────────────────────────────────────────────────────────────────

class Batch {
    Handle<LanceBatch, lance_batch_free> handle_;

public:
    explicit Batch(LanceBatch* b) : handle_(b) {}

    /// Export as Arrow C Data Interface structs.
    void to_arrow(ArrowArray* out_array, ArrowSchema* out_schema) const {
        if (lance_batch_to_arrow(handle_.get(), out_array, out_schema) != 0)
            check_error();
    }
};

} // namespace lance

#endif /* LANCE_HPP */
