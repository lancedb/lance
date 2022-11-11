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
