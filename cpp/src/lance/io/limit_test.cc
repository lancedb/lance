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

#include "lance/io/limit.h"

#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numeric>

#include "lance/arrow/stl.h"
#include "lance/arrow/writer.h"
#include "lance/io/reader.h"

TEST_CASE("LIMIT 100") {
  auto limit = lance::io::Limit(100);
  CHECK(limit.Apply(10).value() == std::make_tuple(0, 10));
  CHECK(limit.Apply(80).value() == std::make_tuple(0, 80));
  CHECK(limit.Apply(20).value() == std::make_tuple(0, 10));
  // Limit already reached.
  CHECK(limit.Apply(30) == std::nullopt);
}

TEST_CASE("LIMIT 10 OFFSET 20") {
  auto limit = lance::io::Limit(10, 20);
  CHECK(limit.Apply(10).value() == std::make_tuple(0, 0));
  CHECK(limit.Apply(5).value() == std::make_tuple(0, 0));
  auto val = limit.Apply(20).value();
  INFO("Limit::Apply(20): offset=" << std::get<0>(val) << " len=" << std::get<1>(val));
  CHECK(val == std::make_tuple(5, 10));
  CHECK(limit.Apply(5) == std::nullopt);
}

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
  auto limit = lance::io::Limit(5, 10);
  auto batch = limit.ReadBatch(reader, reader->schema()).ValueOrDie();
  INFO("Actual: " << batch->column(0)->ToString());
  CHECK(batch->column(0)->Equals(lance::arrow::ToArray({11, 12, 13, 14, 15}).ValueOrDie()));

  // Already read all the data. Crashed on GH-74.
  batch = limit.ReadBatch(reader, reader->schema()).ValueOrDie();
  CHECK(!batch);
}