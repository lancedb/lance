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

#include "lance/io/exec/scan.h"

#include <arrow/table.h>

#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

#include "lance/arrow/stl.h"
#include "lance/format/schema.h"
#include "lance/io/exec/base.h"
#include "lance/testing/io.h"

using lance::format::Schema;
using lance::io::exec::Scan;
using lance::testing::MakeReader;

TEST_CASE("Test Scan::Next") {
  std::vector<int32_t> ints(20);
  std::iota(ints.begin(), ints.end(), 0);
  auto schema = ::arrow::schema({
      ::arrow::field("ints", ::arrow::int32()),
  });
  auto chunked_arrs = ::arrow::ChunkedArray::Make({lance::arrow::ToArray(ints).ValueOrDie(),
                                                   lance::arrow::ToArray(ints).ValueOrDie()})
                          .ValueOrDie();
  auto tab = ::arrow::Table::Make(schema, {chunked_arrs});

  CHECK(tab->column(0)->num_chunks() == 2);
  auto reader = MakeReader(tab).ValueOrDie();

  const int32_t kBatchSize = 8;
  auto scan =
      Scan::Make(reader, std::make_shared<Schema>(reader->schema()), kBatchSize).ValueOrDie();
  auto batch = scan->Next().ValueOrDie();
  CHECK(batch.batch_id == 0);
  CHECK(batch.batch->num_rows() == kBatchSize);
  batch = scan->Next().ValueOrDie();
  CHECK(batch.batch_id == 0);
  CHECK(batch.batch->num_rows() == kBatchSize);
  batch = scan->Next().ValueOrDie();
  CHECK(batch.batch_id == 0);
  CHECK(batch.batch->num_rows() == 20 - 2 * kBatchSize);

  // Second batch
  batch = scan->Next().ValueOrDie();
  CHECK(batch.batch_id == 1);
  CHECK(batch.batch->num_rows() == kBatchSize);
  batch = scan->Next().ValueOrDie();
  CHECK(batch.batch_id == 1);
  CHECK(batch.batch->num_rows() == kBatchSize);
  batch = scan->Next().ValueOrDie();
  CHECK(batch.batch_id == 1);
  CHECK(batch.batch->num_rows() == 20 - 2 * kBatchSize);

  // We should stop now
  batch = scan->Next().ValueOrDie();
  CHECK(!batch.batch);
}