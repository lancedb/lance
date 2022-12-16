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

#include "lance/arrow/updater.h"

#include <arrow/chunked_array.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <range/v3/all.hpp>
#include <unordered_map>
#include <vector>

#include "lance/arrow/dataset.h"
#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using namespace ranges;
using lance::arrow::LanceDataset;
using lance::arrow::ToArray;
namespace fs = std::filesystem;

std::shared_ptr<LanceDataset> TestDataset(
    std::unordered_map<std::string, std::string> metadata = {}) {
  auto ints = views::iota(0, 100) | to<std::vector<int>>();
  auto ints_arr = ToArray(ints).ValueOrDie();
  auto strs = ints | views::transform([](auto v) { return fmt::format("{}", v); }) |
              to<std::vector<std::string>>;
  auto strs_arr = ToArray(strs).ValueOrDie();

  std::shared_ptr<arrow::KeyValueMetadata> kv_metadata;
  if (!metadata.empty()) {
    kv_metadata = std::make_shared<arrow::KeyValueMetadata>(metadata);
  }
  auto schema = arrow::schema(
      {arrow::field("ints", arrow::int32()), ::arrow::field("strs", arrow::utf8())}, kv_metadata);
  auto table = arrow::Table::Make(schema, {ints_arr, strs_arr});

  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);

  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  auto dataset_uri = fs::path(lance::testing::MakeTemporaryDir().ValueOrDie()) / "data";

  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  write_options.filesystem = std::make_shared<::arrow::fs::LocalFileSystem>();
  write_options.base_dir = dataset_uri;
  write_options.max_rows_per_group = 10;
  write_options.file_write_options = format->DefaultWriteOptions();
  CHECK(LanceDataset::Write(write_options, dataset->NewScan().ValueOrDie()->Finish().ValueOrDie())
            .ok());
  return LanceDataset::Make(fs, dataset_uri).ValueOrDie();
}

TEST_CASE("Use updater to update one column") {
  auto lance_dataset = TestDataset();
  CHECK(lance_dataset->version().ValueOrDie().version() == 1);
  auto table = lance_dataset->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();

  auto updater = lance_dataset->NewUpdate(::arrow::field("values", arrow::utf8()))
                     .ValueOrDie()
                     ->Finish()
                     .ValueOrDie();
  int cnt = 0;
  while (true) {
    auto batch = updater->Next().ValueOrDie();
    if (!batch) {
      break;
    }
    cnt++;
    CHECK(batch->schema()->Equals(*lance_dataset->schema()));
    auto input_arr = batch->GetColumnByName("ints");
    auto datum = ::arrow::compute::Cast(input_arr, ::arrow::utf8()).ValueOrDie();
    auto output_arr = datum.make_array();
    auto status = updater->UpdateBatch(output_arr);
    CHECK(status.ok());
  }
  CHECK(cnt == 10);

  auto new_dataset = updater->Finish().ValueOrDie();
  auto actual = new_dataset->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();
  auto expected_strs_arr =
      ToArray(views::iota(0, 100) | views::transform([](auto i) { return fmt::format("{}", i); }) |
              to<std::vector<std::string>>)
          .ValueOrDie();
  auto expected = arrow::Table::Make(::arrow::schema({arrow::field("ints", arrow::int32()),
                                                      arrow::field("strs", arrow::utf8()),
                                                      arrow::field("values", arrow::utf8())}),
                                     {table->GetColumnByName("ints"),
                                      table->GetColumnByName("strs"),
                                      std::make_shared<::arrow::ChunkedArray>(expected_strs_arr)});
  CHECK(expected->Equals(*actual));
}

TEST_CASE("Batch must be consumed before the next iteration") {
  auto dataset = TestDataset();
  auto updater = dataset->NewUpdate(::arrow::field("new_col", arrow::boolean()))
                     .ValueOrDie()
                     ->Finish()
                     .ValueOrDie();
  auto batch = updater->Next().ValueOrDie();
  CHECK(batch);
  auto result = updater->Next();
  CHECK(!result.ok());
}

TEST_CASE("Test update with projection") {
  auto dataset = TestDataset();
  auto builder = dataset->NewUpdate(::arrow::field("new_col", arrow::utf8())).ValueOrDie();
  builder->Project({"ints"});
  auto updater = builder->Finish().ValueOrDie();
  while (true) {
    auto batch = updater->Next().ValueOrDie();
    if (!batch) {
      break;
    }
    CHECK(batch->schema()->Equals(*::arrow::schema({::arrow::field("ints", arrow::int32())})));
    auto input_arr = batch->GetColumnByName("ints");
    auto datum = ::arrow::compute::Cast(input_arr, ::arrow::utf8()).ValueOrDie();
    auto output_arr = datum.make_array();
    CHECK(updater->UpdateBatch(output_arr).ok());
  }
}

TEST_CASE("Test data file stores the relative path to the data dir") {
  auto dataset = TestDataset();

  // Re-open with relative path
  auto local_fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  auto parent_path = fs::path(dataset->uri()).parent_path();
  auto dataset_name = fs::path(dataset->uri()).filename().string();
  fs::current_path(parent_path);
  dataset = LanceDataset::Make(local_fs, dataset_name).ValueOrDie();

  auto updater = dataset->NewUpdate(::arrow::field("col", ::arrow::int32()))
                     .ValueOrDie()
                     ->Finish()
                     .ValueOrDie();
  while (true) {
    auto batch = updater->Next().ValueOrDie();
    if (!batch) {
      break;
    }
    auto output = ::arrow::compute::Add(batch->GetColumnByName("ints"), ::arrow::Datum(2))
                      .ValueOrDie()
                      .make_array();
    CHECK(updater->UpdateBatch(output).ok());
  }
  updater->Finish().ValueOrDie();

  dataset = LanceDataset::Make(local_fs, fs::path(dataset->uri()).filename().string()).ValueOrDie();
  CHECK(dataset->version().ValueOrDie().version() == 2);
  auto table = dataset->NewScan().ValueOrDie()->Finish().ValueOrDie()->ToTable().ValueOrDie();
  CHECK(table->schema()->num_fields() == 3);

  for (auto fragment_result : dataset->GetFragments().ValueOrDie()) {
    auto fragment = fragment_result.ValueOrDie();
    auto lance_fragment = std::dynamic_pointer_cast<lance::arrow::LanceFragment>(fragment);
    for (auto& data_file : lance_fragment->data_fragment()->data_files()) {
      CHECK(data_file.path().find("/") == std::string::npos);
    }
  }
}

TEST_CASE("Update schema with metadata") {
  auto dataset = TestDataset({{"k1", "v1"}});

  CHECK(dataset->schema()->metadata()->keys() == std::vector<std::string>({"k1"}));
  CHECK(dataset->schema()->metadata()->values() == std::vector<std::string>({"v1"}));

  // Do not change metadata
  auto updater = dataset->NewUpdate(::arrow::field("col1", ::arrow::int32()))
                     .ValueOrDie()
                     ->Finish()
                     .ValueOrDie();
  while (true) {
    auto batch = updater->Next().ValueOrDie();
    if (!batch) {
      break;
    }
    auto output = ::arrow::compute::Add(batch->GetColumnByName("ints"), ::arrow::Datum(2))
                      .ValueOrDie()
                      .make_array();
    CHECK(updater->UpdateBatch(output).ok());
  }
  dataset = updater->Finish().ValueOrDie();
  INFO("Expect the metadata is preserved if not overwrite");
  CHECK(dataset->version().ValueOrDie().version() == 2);
  CHECK(dataset->schema()->metadata()->keys() == std::vector<std::string>({"k1"}));
  CHECK(dataset->schema()->metadata()->values() == std::vector<std::string>({"v1"}));

  auto updater_builder = dataset->NewUpdate(::arrow::field("col", ::arrow::int32())).ValueOrDie();
  std::unordered_map<std::string, std::string> new_metadata{{"k2", "v2"}, {"k3", "v3"}};
  updater_builder->Metadata(new_metadata);
  updater = updater_builder->Finish().ValueOrDie();

  while (true) {
    auto batch = updater->Next().ValueOrDie();
    if (!batch) {
      break;
    }
    auto output = ::arrow::compute::Add(batch->GetColumnByName("ints"), ::arrow::Datum(2))
                      .ValueOrDie()
                      .make_array();
    CHECK(updater->UpdateBatch(output).ok());
  }
  dataset = updater->Finish().ValueOrDie();

  CHECK(dataset->version().ValueOrDie().version() == 3);
  CHECK(dataset->version()->metadata() == new_metadata);
}