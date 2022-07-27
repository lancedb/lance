//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <arrow/table.h>
#include <fmt/format.h>
#include <parquet/arrow/reader.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <random>
#include <string>

#include "bench_utils.h"
#include "lance/arrow/reader.h"

std::string uri;

void BenchmarkPointQueryOnParquet(const std::string& uri) {
  auto f = OpenUri(uri);
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto status = parquet::arrow::OpenFile(f, ::arrow::default_memory_pool(), &reader);
  INFO("Open parquet file: " << status.message());
  CHECK(status.ok());
  fmt::print("Open parquet file: {}\n", uri);
  fmt::print("Number of row groups: {}\n", reader->num_row_groups());

  auto num_row_groups = reader->num_row_groups();
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> dist(0, num_row_groups - 1);

  auto read = [&]() {
    auto idx = dist(mt);
    auto row_group = reader->RowGroup(idx);
    std::shared_ptr<::arrow::Table> tbl;
    CHECK(row_group->ReadTable(&tbl).ok());
    auto num_rows = tbl->num_rows();
    std::uniform_int_distribution<int64_t> row_dist(0, num_rows);
    auto row = tbl->Slice(row_dist(mt), 1);
  };

  BENCHMARK("Single Thread") { return read(); };
}

void BenchmarkPointQueryLance(const std::string& uri) {
  auto f = OpenUri(uri);
  auto reader = ::lance::arrow::FileReader::Make(f).ValueOrDie();
  auto length = reader->length();

  fmt::print("Open Lance File: {}\n", uri);
  fmt::print("Number of Pages: {}, Rows={}\n", reader->num_batches(), length);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> dist(0, length - 1);

  auto read = [&]() {
    auto idx = dist(mt);
    auto row = reader->Get(idx);
    CHECK(row.ok());
  };

  BENCHMARK("Single Thread") { return read(); };
}

TEST_CASE("Random Access Over One File") {
  CHECK(!uri.empty());

  if (uri.ends_with(".parquet") || uri.ends_with(".parq")) {
    BenchmarkPointQueryOnParquet(uri);
  } else if (uri.ends_with(".lance")) {
    BenchmarkPointQueryLance(uri);
  } else {
    FAIL("Unsupported file: " << uri);
  }
}

int main(int argc, char** argv) {
  Catch::Session session;
  using namespace Catch::Clara;

  auto cli = session.cli() | Opt(uri, "uri")["--uri"]("Input file URI");
  session.cli(cli);

  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0) {
    return ret;
  }
  return session.run();
}
