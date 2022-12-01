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
#include <chrono>
#include <memory>
#include <range/v3/view.hpp>
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;
using namespace ranges::views;
using namespace std::chrono_literals;

std::shared_ptr<::arrow::Table> ReadTable(const std::string& uri, std::optional<int32_t> version) {
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(uri, &path).ValueOrDie();
  auto actual_dataset = lance::arrow::LanceDataset::Make(fs, uri, version).ValueOrDie();
  CHECK(actual_dataset != nullptr);
  return actual_dataset->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();
}

// Write table as dataset.
std::string WriteTable(const std::shared_ptr<::arrow::Table>& table) {
  auto base_uri = lance::testing::MakeTemporaryDir().ValueOrDie() + "/testdata";
  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(base_uri, &path).ValueOrDie();
  write_options.filesystem = fs;
  write_options.base_dir = path;
  write_options.file_write_options = format->DefaultWriteOptions();

  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie())
            .ok());
  return base_uri;
}

TEST_CASE("Create new dataset") {
  auto ids = ToArray({1, 2, 3, 4, 5, 6, 8}).ValueOrDie();
  auto values = ToArray({"a", "b", "c", "d", "e", "f", "g"}).ValueOrDie();
  auto table1 = ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                                      ::arrow::field("value", ::arrow::utf8())}),
                                     {ids, values});

  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table1);

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
  dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table2);
  CHECK(lance::arrow::LanceDataset::Write(write_options,
                                          dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                          lance::arrow::LanceDataset::kAppend)
            .ok());

  // Version 3 is overwriting.
  dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table2);
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
  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);

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
  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);

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
    auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
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
  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);

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
  dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  auto status =
      lance::arrow::LanceDataset::Write(write_options,
                                        dataset->NewScan().ValueOrDie()->Finish().ValueOrDie(),
                                        lance::arrow::LanceDataset::kOverwrite);
  INFO("Status: " << status.message() << " is ok: " << status.ok());
  CHECK(status.IsIOError());
}

TEST_CASE("Dataset write dictionary array") {
  auto dict_values = ToArray({"a", "b", "c"}).ValueOrDie();
  auto dict_indices = ToArray({0, 1, 1, 2, 2, 0}).ValueOrDie();
  auto data_type = ::arrow::dictionary(::arrow::int32(), ::arrow::utf8());
  auto dict_arr =
      ::arrow::DictionaryArray::FromArrays(data_type, dict_indices, dict_values).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("dict", data_type)}), {dict_arr});

  auto base_uri = WriteTable(table);
  fmt::print("Base URI: {}\n", base_uri);

  auto actual = ReadTable(base_uri, 1);
  CHECK(actual->Equals(*table));
}

TEST_CASE("Dataset add column with a constant value") {
  auto ids = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32())}), {ids});
  auto base_uri = WriteTable(table);
  auto actual = ReadTable(base_uri, 1);

  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  auto dataset = lance::arrow::LanceDataset::Make(fs, base_uri).ValueOrDie();
  CHECK(dataset->version().timestamp() > std::chrono::system_clock::now() - 30s);

  auto dataset2 =
      dataset
          ->AddColumn(::arrow::field("doubles", ::arrow::float64()), ::arrow::compute::literal(0.5))
          .ValueOrDie();
  CHECK(dataset2->version().version() == 2);
  CHECK(dataset2->version().timestamp() > dataset->version().timestamp());

  auto table2 = dataset2->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();
  auto doubles = ToArray<double>({0.5, 0.5, 0.5, 0.5, 0.5}).ValueOrDie();
  auto expected_table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                            ::arrow::field("doubles", ::arrow::float64())}),
                           {ids, doubles});
  CHECK(table2->Equals(*expected_table));
}

TEST_CASE("Dataset add column with a function call") {
  auto ids = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32())}), {ids});
  auto base_uri = WriteTable(table);
  auto actual = ReadTable(base_uri, 1);

  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  auto dataset = lance::arrow::LanceDataset::Make(fs, base_uri).ValueOrDie();
  CHECK(dataset->version().timestamp() > std::chrono::system_clock::now() - 30s);

  auto dataset2 =
      dataset
          ->AddColumn(
              ::arrow::field("doubles", ::arrow::float64()),
              ::arrow::compute::call(
                  "add", {::arrow::compute::field_ref("id"), ::arrow::compute::literal(0.5)}))
          .ValueOrDie();
  CHECK(dataset2->version().version() == 2);
  CHECK(dataset2->version().timestamp() > dataset->version().timestamp());
  auto table2 = dataset2->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();
  auto doubles = ToArray<double>({1.5, 2.5, 3.5, 4.5, 5.5}).ValueOrDie();
  auto expected_table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                            ::arrow::field("doubles", ::arrow::float64())}),
                           {ids, doubles});
  CHECK(table2->Equals(*expected_table));
}

TEST_CASE("Dataset add columns with a table") {
  auto ids = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto values = ToArray({"one", "two", "three", "four", "five"}).ValueOrDie();
  auto schema = ::arrow::schema(
      {::arrow::field("id", ::arrow::int32()), ::arrow::field("value", ::arrow::utf8())});
  auto table = ::arrow::Table::Make(schema, {ids, values});
  auto base_uri = WriteTable(table);

  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  auto dataset = lance::arrow::LanceDataset::Make(fs, base_uri).ValueOrDie();
  CHECK(dataset->version().version() == 1);

  auto added_ids = ToArray({5, 4, 3, 10, 12, 1}).ValueOrDie();
  auto added_values = ToArray({50, 40, 30, 100, 120, 10}).ValueOrDie();
  auto added_table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                            ::arrow::field("new_value", ::arrow::int32())}),
                           {added_ids, added_values});
  auto new_dataset = dataset->Merge(added_table, "id").ValueOrDie();
  CHECK(new_dataset->version().version() == 2);
  auto new_table =
      new_dataset->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();

  // TODO: Plain array does not support null yet, so arr[1] = 0 instead of Null.
  auto new_values = ToArray({10, 0, 30, 40, 50}).ValueOrDie();
  auto expected_table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("id", ::arrow::int32()),
                                            ::arrow::field("value", ::arrow::utf8()),
                                            ::arrow::field("new_value", ::arrow::int32())}),
                           {ids, values, new_values});
  CHECK(new_table->Equals(*expected_table));
}