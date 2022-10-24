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
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;

TEST_CASE("Create new dataset") {
  auto ids = ToArray({1, 2, 3, 4, 5, 6, 8}).ValueOrDie();
  auto values = ToArray({"a", "b", "c", "d", "e", "f", "g"}).ValueOrDie();
  auto table = ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                                     ::arrow::field("value", ::arrow::utf8())}),
                                    {ids, values});

  auto dataset = lance::testing::MakeDataset(table).ValueOrDie();

  auto base_uri = lance::testing::MakeTemporaryDir().ValueOrDie() + "/testdata";
  auto format = lance::arrow::LanceFileFormat::Make();
  fmt::print("Base uri: {}\n", base_uri);
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(base_uri, &path).ValueOrDie();
  write_options.filesystem = fs;
  write_options.base_dir = path;
  write_options.file_write_options = format->DefaultWriteOptions();
  write_options.basename_template = "part{i}.lance";

  auto status = lance::arrow::LanceDataset::Write(
      write_options, dataset->NewScan().ValueOrDie()->Finish().ValueOrDie());
  INFO("Write dataset: " << status.message());
  CHECK(status.ok());
}