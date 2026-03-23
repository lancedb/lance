/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: Copyright The Lance Authors */

/**
 * @file test_cpp_api.cpp
 * @brief C++ compilation and functional test for lance.hpp
 *
 * Tests the RAII wrappers, exception handling, and builder pattern.
 *
 * Usage: test_cpp_api <dataset_uri>
 */

#include "lance.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define TEST(name) printf("  %s... ", #name)
#define PASS()     printf("OK\n")

static void test_dataset_open(const std::string& uri) {
    TEST(test_dataset_open);

    auto ds = lance::Dataset::open(uri);
    assert(ds.version() >= 1);
    assert(ds.count_rows() > 0);

    printf("version=%llu, rows=%llu... ",
           (unsigned long long)ds.version(),
           (unsigned long long)ds.count_rows());

    PASS();
}

static void test_dataset_schema(const std::string& uri) {
    TEST(test_dataset_schema);

    auto ds = lance::Dataset::open(uri);

    ArrowSchema schema;
    memset(&schema, 0, sizeof(schema));
    ds.schema(&schema);

    assert(schema.n_children > 0);
    printf("fields=%lld... ", (long long)schema.n_children);

    // Print field names
    for (int64_t i = 0; i < schema.n_children; i++) {
        if (i > 0) printf(", ");
        printf("%s", schema.children[i]->name);
    }
    printf("... ");

    if (schema.release) schema.release(&schema);

    PASS();
}

static void test_scanner_fluent(const std::string& uri) {
    TEST(test_scanner_fluent);

    auto ds = lance::Dataset::open(uri);

    // Fluent builder pattern.
    auto scanner = ds.scan();
    scanner.limit(5).offset(0).batch_size(2);

    ArrowArrayStream stream;
    memset(&stream, 0, sizeof(stream));
    scanner.to_arrow_stream(&stream);

    // Count rows from stream.
    uint64_t total = 0;
    while (true) {
        ArrowArray arr;
        memset(&arr, 0, sizeof(arr));
        int rc = stream.get_next(&stream, &arr);
        assert(rc == 0);
        if (!arr.release) break;
        total += (uint64_t)arr.length;
        arr.release(&arr);
    }

    assert(total == 5);
    printf("rows=%llu... ", (unsigned long long)total);

    if (stream.release) stream.release(&stream);
    PASS();
}

static void test_dataset_take(const std::string& uri) {
    TEST(test_dataset_take);

    auto ds = lance::Dataset::open(uri);

    uint64_t indices[] = {0, 1, 2};
    ArrowArrayStream stream;
    memset(&stream, 0, sizeof(stream));
    ds.take(indices, 3, &stream);

    uint64_t total = 0;
    while (true) {
        ArrowArray arr;
        memset(&arr, 0, sizeof(arr));
        int rc = stream.get_next(&stream, &arr);
        assert(rc == 0);
        if (!arr.release) break;
        total += (uint64_t)arr.length;
        arr.release(&arr);
    }

    assert(total == 3);
    printf("rows=%llu... ", (unsigned long long)total);

    if (stream.release) stream.release(&stream);
    PASS();
}

static void test_raii_cleanup(const std::string& uri) {
    TEST(test_raii_cleanup);

    // Dataset and Scanner should clean up automatically.
    {
        auto ds = lance::Dataset::open(uri);
        auto scanner = ds.scan();
        scanner.limit(1);
        // Goes out of scope — RAII cleanup.
    }

    // Move semantics.
    {
        auto ds1 = lance::Dataset::open(uri);
        auto ds2 = std::move(ds1);
        assert(ds2.count_rows() > 0);
    }

    PASS();
}

static void test_error_exception(const std::string& /*uri*/) {
    TEST(test_error_exception);

    bool caught = false;
    try {
        lance::Dataset::open("file:///nonexistent/path/xyz");
    } catch (const lance::Error& e) {
        caught = true;
        assert(e.code != LANCE_OK);
        assert(strlen(e.what()) > 0);
        printf("caught: %s... ", e.what());
    }
    assert(caught);

    PASS();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dataset_uri>\n", argv[0]);
        return 1;
    }

    std::string uri(argv[1]);
    printf("Running C++ API tests with dataset: %s\n", uri.c_str());

    test_dataset_open(uri);
    test_dataset_schema(uri);
    test_scanner_fluent(uri);
    test_dataset_take(uri);
    test_raii_cleanup(uri);
    test_error_exception(uri);

    printf("All C++ tests passed!\n");
    return 0;
}
