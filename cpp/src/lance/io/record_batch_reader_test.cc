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

#include "lance/io/record_batch_reader.h"

#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/table.h>

#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

#include "lance/arrow/scanner.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using lance::io::RecordBatchReader;

TEST_CASE("Test Read with batch size") {
  const int kBatchSize = 5;
  std::vector<int32_t> values(100);
  std::iota(values.begin(), values.end(), 1);
  auto arr = lance::arrow::ToArray(values).ValueOrDie();

  auto schema = arrow::schema({arrow::field("values", arrow::int32())});
  auto t = arrow::Table::Make(schema, {arr});
  auto file_reader = lance::testing::MakeReader(t).ValueOrDie();
  auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(t);

  auto builder = lance::arrow::ScannerBuilder(dataset);
  builder.BatchSize(kBatchSize);
  auto scanner = builder.Finish().ValueOrDie();

  auto record_batch_reader = RecordBatchReader(file_reader, scanner->options());
  while (true) {
    auto fut = record_batch_reader();
    CHECK(fut.Wait(5));
    auto result = fut.result();
    CHECK(result.ok());
    auto batch = result.ValueOrDie();
    if (!batch) {
      break;
    }
    CHECK(batch->num_rows() == kBatchSize);
  }
}