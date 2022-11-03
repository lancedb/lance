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

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/stl.h"
#include "lance/io/writer.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;
using lance::testing::MakeFragment;
namespace fs = std::filesystem;

TEST_CASE("Read fragment with one file") {
  auto arr = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto t = arrow::Table::Make(::arrow::schema({::arrow::field("int", arr->type())}), {arr});

  auto fragment = MakeFragment(t).ValueOrDie();

  auto scan_options = std::make_shared<::arrow::dataset::ScanOptions>();
  scan_options->dataset_schema = t->schema();
  scan_options->projected_schema = t->schema();
  auto generator = fragment->ScanBatchesAsync(scan_options).ValueOrDie();
  auto batch = generator().result().ValueOrDie();

  CHECK(batch->schema()->Equals(*t->schema()));
  CHECK(arr->Equals(batch->GetColumnByName("int")));
}

TEST_CASE("Test add column with an array") {
  auto arr = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto t = arrow::Table::Make(::arrow::schema({::arrow::field("int", arr->type())}), {arr});
  auto fragment = MakeFragment(t).ValueOrDie();

  auto new_arr = ToArray({"1", "2", "3", "4", "5"}).ValueOrDie();
  auto new_chunked_arr = std::make_shared<::arrow::ChunkedArray>(new_arr);
  auto full_schema = std::make_shared<lance::format::Schema>(t->schema());
  auto new_field = std::make_shared<lance::format::Field>(::arrow::field("str", ::arrow::utf8()));
  full_schema->AddField(new_field);
  auto new_schema = full_schema->Project({"str"}).ValueOrDie();
  auto new_fragment = fragment->AddColumn(full_schema, new_schema, new_chunked_arr).ValueOrDie();

  auto expected_table = arrow::Table::Make(full_schema->ToArrow(), {arr, new_arr});
  auto scan_options = std::make_shared<::arrow::dataset::ScanOptions>();
  scan_options->dataset_schema = expected_table->schema();
  scan_options->projected_schema = expected_table->schema();

  auto generator = new_fragment->ScanBatchesAsync(scan_options).ValueOrDie();
  auto batch = generator().result().ValueOrDie();
  INFO("Batch schema: " << batch->schema()->ToString()
                        << " != expected table schema: " << expected_table->schema()->ToString());
  CHECK(batch->schema()->Equals(*expected_table->schema()));
  //  CHECK(arr->Equals(batch->GetColumnByName("int")));
}

TEST_CASE("Test add column without data") {}

TEST_CASE("Add one ") {}
