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

#include <arrow/compute/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/table.h>

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <range/v3/all.hpp>
#include <vector>

#include "lance/arrow/dataset.h"
#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using namespace ranges;
using lance::arrow::LanceDataset;
using lance::arrow::ToArray;
namespace fs = std::filesystem;

TEST_CASE("Use updater to update one column") {
  auto ints = views::iota(0, 100) | to<std::vector<int>>();
  auto ints_arr = ToArray(ints).ValueOrDie();
  auto schema = arrow::schema({arrow::field("ints", arrow::int32())});
  auto table = arrow::Table::Make(schema, {ints_arr});

  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);

  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  auto dataset_uri = fs::path(lance::testing::MakeTemporaryDir().ValueOrDie()) / "update";

  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  write_options.filesystem = std::make_shared<::arrow::fs::LocalFileSystem>();
  write_options.base_dir = dataset_uri;
  write_options.max_rows_per_group = 10;
  write_options.file_write_options = format->DefaultWriteOptions();
  CHECK(LanceDataset::Write(write_options, dataset->NewScan().ValueOrDie()->Finish().ValueOrDie())
            .ok());
  fmt::print("Dataset URI: {}\n", dataset_uri.string());

  auto lance_dataset = LanceDataset::Make(fs, dataset_uri).ValueOrDie();
  CHECK(lance_dataset->version().version() == 1);

  auto updater = lance_dataset->NewUpdate(::arrow::field("strs", arrow::utf8()))
                     .ValueOrDie()
                     .Finish()
                     .ValueOrDie();
  int cnt = 0;
  while (true) {
    auto batch = updater->Next().ValueOrDie();
    if (!batch) {
      break;
    }
    cnt++;
    CHECK(batch->schema()->Equals(*table->schema()));
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
  auto expected = arrow::Table::Make(
      ::arrow::schema({arrow::field("ints", arrow::int32()), arrow::field("strs", arrow::utf8())}),
      {ints_arr, expected_strs_arr});
  CHECK(expected->Equals(*actual));
}