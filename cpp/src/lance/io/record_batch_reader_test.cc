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

#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using lance::arrow::ToArray;

TEST_CASE("Scan partitioned dataset") {
  auto value_arr = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto split_arr = ToArray({"train", "train", "eval", "test", "train"}).ValueOrDie();

  auto schema = ::arrow::schema(
      {::arrow::field("value", ::arrow::int32()), ::arrow::field("split", ::arrow::utf8())});
  auto t = ::arrow::Table::Make(schema, {value_arr, split_arr});

  auto dataset = lance::testing::MakeDataset(t, {"split"}).ValueOrDie();
  auto scanner = dataset->NewScan().ValueOrDie()->Finish().ValueOrDie();
  auto actual = scanner->ToTable().ValueOrDie()->CombineChunks().ValueOrDie();
  auto indices = ::arrow::compute::SortIndices(*actual->GetColumnByName("value")).ValueOrDie();
  auto new_datum = ::arrow::compute::Take(actual, indices).ValueOrDie();
  auto sorted_table = new_datum.table();
  INFO("Expected table: " << t->ToString() << " \nActual table: " << sorted_table->ToString());
  CHECK(t->Equals(*sorted_table));
}

TEST_CASE("Scan partitioned dataset with nonexistent column") {
  auto value_arr = ToArray({1, 2, 3, 4, 5}).ValueOrDie();
  auto split_arr = ToArray({"train", "train", "eval", "test", "train"}).ValueOrDie();

  auto schema = ::arrow::schema(
      {::arrow::field("value", ::arrow::int32()), ::arrow::field("split", ::arrow::utf8())});
  auto t = ::arrow::Table::Make(schema, {value_arr, split_arr});
  auto dataset = lance::testing::MakeDataset(t, {"split"}).ValueOrDie();
  auto scan_builder = dataset->NewScan().ValueOrDie();
  // Woo column does not exist in the dataset, split column does not exist in the lance file.
  CHECK(!scan_builder->Project({"value", "split", "woo"}).ok());
}
