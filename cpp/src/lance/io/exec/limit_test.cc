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

#include "lance/io/exec/limit.h"

#include <arrow/io/api.h>
#include <arrow/table.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numeric>

#include "lance/arrow/stl.h"
#include "lance/arrow/writer.h"
#include "lance/format/schema.h"
#include "lance/io/exec/counter.h"
#include "lance/io/exec/scan.h"
#include "lance/io/reader.h"
#include "lance/testing/io.h"

using lance::format::Schema;
using lance::io::exec::Counter;
using lance::io::exec::Limit;
using lance::io::exec::Scan;
using lance::testing::TableScan;

TEST_CASE("Read limit multiple times") {
  auto values = std::vector<int>(50);
  std::iota(std::begin(values), std::end(values), 1);
  auto array = lance::arrow::ToArray(values).ValueOrDie();

  auto schema = ::arrow::schema({::arrow::field("values", ::arrow::int32())});
  auto table = ::arrow::Table::Make(schema, {array});
  auto sink = ::arrow::io::BufferOutputStream::Create().ValueOrDie();
  CHECK(lance::arrow::WriteTable(*table, sink).ok());

  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  auto reader = std::make_shared<lance::io::FileReader>(infile);
  CHECK(reader->Open().ok());
  auto scan =
      lance::io::exec::Scan::Make({{reader, std::make_shared<Schema>(reader->schema())}}, 100)
          .ValueOrDie();
  auto counter = std::make_shared<Counter>(5, 10);
  auto limit = Limit(counter, std::move(scan));
  auto batch = limit.Next().ValueOrDie();
  INFO("Actual: " << batch.batch->column(0)->ToString());
  CHECK(batch.batch->column(0)->Equals(lance::arrow::ToArray({11, 12, 13, 14, 15}).ValueOrDie()));

  // Already read all the data. Crashed on GH-74.
  batch = limit.Next().ValueOrDie();
  CHECK(batch.eof());
}