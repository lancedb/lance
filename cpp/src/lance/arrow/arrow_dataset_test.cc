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

#include <arrow/dataset/discovery.h>
#include <arrow/type.h>

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <numeric>
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/arrow/writer.h"
#include "lance/testing/io.h"
#include "lance/testing/json.h"

namespace fs = std::filesystem;

using lance::arrow::ToArray;
using lance::testing::MakeDataset;
using lance::testing::TableFromJSON;

TEST_CASE("FileSystemFactory Test") {
  auto tmpdir = fs::temp_directory_path();
  auto path = tmpdir / "test.lance";
  auto uri = std::string("file://") + path.string();

  auto schema = arrow::schema({arrow::field("key", arrow::utf8())});
  auto table =
      TableFromJSON(schema, R"([{"key": "one"}, {"key": "two"}, {"key": "three"}])").ValueOrDie();
  auto fs = arrow::fs::FileSystemFromUriOrPath(path).ValueOrDie();

  {
    auto sink = fs->OpenOutputStream(path).ValueOrDie();
    CHECK(lance::arrow::WriteTable(*table, sink).ok());
  }

  auto factory =
      arrow::dataset::FileSystemDatasetFactory::Make(
          uri,
          std::shared_ptr<arrow::dataset::FileFormat>(new lance::arrow::LanceFileFormat()),
          arrow::dataset::FileSystemFactoryOptions())
          .ValueOrDie();
  auto dataset = factory->Finish().ValueOrDie();
  CHECK(dataset->schema()->Equals(schema));

  auto scanner_builder = dataset->NewScan().ValueOrDie();
  auto scanner = scanner_builder->Finish().ValueOrDie();
  CHECK(scanner->CountRows().ValueOrDie() == 3);
  auto actual_table = scanner->ToTable().ValueOrDie();
  INFO("Expect table: " << table->ToString() << " Actual table: " << actual_table->ToString());
  CHECK(table->Equals(*actual_table));
}

TEST_CASE("Test CountRows fast path") {
  auto tmpdir = fs::temp_directory_path();
  auto path = tmpdir / "test.lance";
  auto uri = std::string("file://") + path.string();

  auto schema = arrow::schema({arrow::field("key", arrow::utf8())});
  std::vector<int32_t> values(1000);
  std::iota(std::begin(values), std::end(values), 1);
  auto table = ::arrow::Table::Make(schema, {ToArray(values).ValueOrDie()});

  auto dataset = MakeDataset(table, {}, 32).ValueOrDie();
  auto scanner = dataset->NewScan().ValueOrDie()->Finish().ValueOrDie();
  CHECK(scanner->CountRows().ValueOrDie() == 1000);
}