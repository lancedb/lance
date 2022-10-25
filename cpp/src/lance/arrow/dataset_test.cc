//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/arrow/dataset.h"

#include <arrow/table.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;

std::shared_ptr<::arrow::Table> ReadTable(const std::string& uri, std::optional<int32_t> version) {
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(uri, &path).ValueOrDie();
  auto actual_dataset = lance::arrow::LanceDataset::Make(fs, uri, version).ValueOrDie();
  CHECK(actual_dataset != nullptr);
  return actual_dataset->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();
}

TEST_CASE("Create new dataset") {
  auto ids = ToArray({1, 2, 3, 4, 5, 6, 8}).ValueOrDie();
  auto values = ToArray({"a", "b", "c", "d", "e", "f", "g"}).ValueOrDie();
  auto table1 = ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                                      ::arrow::field("value", ::arrow::utf8())}),
                                     {ids, values});

  auto dataset = lance::testing::MakeDataset(table1).ValueOrDie();

  auto base_uri = lance::testing::MakeTemporaryDir().ValueOrDie() + "/testdata";
  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(base_uri, &path).ValueOrDie();
  write_options.filesystem = fs;
  write_options.base_dir = path;
  write_options.file_write_options = format->DefaultWriteOptions();

  auto status = lance::arrow::LanceDataset::Write(
      write_options, dataset->NewScan().ValueOrDie()->Finish().ValueOrDie());
  CHECK(status.ok());

  ids = ToArray({100, 101, 102}).ValueOrDie();
  values = ToArray({"aaa", "bbb", "ccc"}).ValueOrDie();
  auto table2 = ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                                      ::arrow::field("value", ::arrow::utf8())}),
                                     {ids, values});
  dataset = lance::testing::MakeDataset(table2).ValueOrDie();
  status = lance::arrow::LanceDataset::Write(
      write_options, dataset->NewScan().ValueOrDie()->Finish().ValueOrDie());
  INFO("Write dataset: " << status.message());
  CHECK(status.ok());

  auto table_v1 = ReadTable(base_uri, 1);
  CHECK(table_v1->Equals(*table1));

  auto table_v2 = ReadTable(base_uri, 2);
  auto combined_table = ::arrow::ConcatenateTables({table1, table2}).ValueOrDie();
  CHECK(table_v2->Equals(*combined_table));
}