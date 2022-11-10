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

#include "lance/arrow/fragment.h"

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/table.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <memory>
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/arrow/utils.h"
#include "lance/io/writer.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;
namespace fs = std::filesystem;

/// Make one lance format from the table.
std::shared_ptr<lance::arrow::LanceFragment> MakeFragment(
    const std::shared_ptr<::arrow::Table>& table) {
  auto data_dir = lance::testing::MakeTemporaryDir().ValueOrDie();
  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  std::string filename(fs::path(data_dir) / "12345-67890.lance");
  auto output = fs->OpenOutputStream(filename).ValueOrDie();

  {
    auto write_options = lance::arrow::LanceFileFormat().DefaultWriteOptions();
    auto writer = lance::io::FileWriter(table->schema(), write_options, output);
    auto batch_reader = ::arrow::TableBatchReader(table);
    std::shared_ptr<::arrow::RecordBatch> batch;
    while (true) {
      CHECK(batch_reader.ReadNext(&batch).ok());
      if (batch == nullptr) {
        break;
      }
      CHECK(writer.Write(batch).ok());
    }
    writer.Finish().Wait();
  }

  auto schema = std::make_shared<lance::format::Schema>(table->schema());
  auto data_file = lance::format::DataFile(filename, schema->GetFieldIds());
  auto fragment = std::make_shared<lance::format::DataFragment>(data_file);
  return std::make_shared<lance::arrow::LanceFragment>(
      fs, data_dir, std::move(fragment), std::move(schema));
};

TEST_CASE("Read fragment with one file") {
  auto arr = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto t = arrow::Table::Make(::arrow::schema({::arrow::field("int", arr->type())}), {arr});

  auto fragment = MakeFragment(t);

  auto scan_options = std::make_shared<::arrow::dataset::ScanOptions>();
  scan_options->dataset_schema = t->schema();
  scan_options->projected_schema = t->schema();
  auto generator = fragment->ScanBatchesAsync(scan_options).ValueOrDie();
  auto batch = generator().result().ValueOrDie();

  CHECK(batch->schema()->Equals(*t->schema()));
  CHECK(arr->Equals(batch->GetColumnByName("int")));
}
