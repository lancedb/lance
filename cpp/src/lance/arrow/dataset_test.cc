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
#include <range/v3/view.hpp>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;
using namespace ranges::views;


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

  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie())
            .ok());

  ids = ToArray({100, 101, 102}).ValueOrDie();
  values = ToArray({"aaa", "bbb", "ccc"}).ValueOrDie();
  auto table2 = ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                                      ::arrow::field("value", ::arrow::utf8())}),
                                     {ids, values});

  // Version 2 is appending.
  dataset = lance::testing::MakeDataset(table2).ValueOrDie();
  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kAppend)
            .ok());

  // Version 3 is overwriting.
  dataset = lance::testing::MakeDataset(table2).ValueOrDie();
  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kOverwrite)
            .ok());

  auto table_v1 = ReadTable(base_uri, 1);
  CHECK(table_v1->Equals(*table1));

  auto table_v2 = ReadTable(base_uri, 2);
  auto combined_table = ::arrow::ConcatenateTables({table1, table2}).ValueOrDie();
  CHECK(table_v2->Equals(*combined_table));

  // Only read the overwritten dataset.
  auto table_v3 = ReadTable(base_uri, 3);
  CHECK(table_v3->Equals(*table2));

  // Read dataset versions
  auto lance_dataset = lance::arrow::LanceDataset::Make(fs, path).ValueOrDie();
  auto versions = lance_dataset->versions().ValueOrDie();
  CHECK(versions.size() == 3);
  std::vector<uint64_t> expected_version({1, 2, 3});
  for (auto [v, data_version] : zip(expected_version, versions)) {
    CHECK(v == data_version.version());
  }

  auto latest = lance_dataset->latest_version().ValueOrDie();
  CHECK(latest.version() == 3);
}

TEST_CASE("Create new dataset over existing dataset") {
  auto ids = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32())}), {ids});
  auto dataset = lance::testing::MakeDataset(table).ValueOrDie();

  auto base_uri = lance::testing::MakeTemporaryDir().ValueOrDie() + "/testdata";
  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(base_uri, &path).ValueOrDie();
  write_options.filesystem = fs;
  write_options.base_dir = path;
  write_options.file_write_options = format->DefaultWriteOptions();

  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kCreate)
            .ok());
  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kCreate)
            .IsAlreadyExists());
}

TEST_CASE("Dataset append error cases") {
  auto ids = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32())}), {ids});
  auto dataset = lance::testing::MakeDataset(table).ValueOrDie();

  auto base_uri = lance::testing::MakeTemporaryDir().ValueOrDie() + "/testdata";
  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(base_uri, &path).ValueOrDie();
  write_options.filesystem = fs;
  write_options.base_dir = path;
  write_options.file_write_options = format->DefaultWriteOptions();

  SECTION("Append to non-existed path") {
    CHECK(lance::arrow::LanceDataset::Write(write_options,
                                            dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                            lance::arrow::LanceDataset::kAppend)
              .IsIOError());
  }

  SECTION("Append with different schema") {
    write_options.base_dir = path + "_1";
    CHECK(lance::arrow::LanceDataset::Write(write_options,
                                            dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                            lance::arrow::LanceDataset::kCreate)
              .ok());
    auto values = ToArray({"one", "two", "three", "four", "five"}).ValueOrDie();
    table = ::arrow::Table::Make(::arrow::schema({::arrow::field("values", ::arrow::utf8())}),
                                 {values});
    dataset = lance::testing::MakeDataset(table).ValueOrDie();
    auto status =
        lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kAppend);
    CHECK(status.IsIOError());
  }
}

TEST_CASE("Dataset overwrite error cases") {
  auto ids = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32())}), {ids});
  auto dataset = lance::testing::MakeDataset(table).ValueOrDie();

  auto base_uri = lance::testing::MakeTemporaryDir().ValueOrDie() + "/testdata";
  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(base_uri, &path).ValueOrDie();
  write_options.filesystem = fs;
  write_options.base_dir = path;
  write_options.file_write_options = format->DefaultWriteOptions();

  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kOverwrite)
            .ok());

  auto values = ToArray({"one", "two", "three", "four", "five"}).ValueOrDie();
  table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("values", ::arrow::utf8())}), {values});
  dataset = lance::testing::MakeDataset(table).ValueOrDie();
  auto status =
      lance::arrow::LanceDataset::Write(write_options,
                                        dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                        lance::arrow::LanceDataset::kOverwrite);
  INFO("Status: " << status.message() << " is ok: " << status.ok());
  CHECK(status.IsIOError());
}