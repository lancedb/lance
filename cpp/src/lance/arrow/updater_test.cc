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

#include <arrow/table.h>

#include <catch2/catch_test_macros.hpp>
#include <range/v3/all.hpp>
#include <ranges>
#include <vector>

#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using namespace ranges;
using lance::arrow::ToArray;
using lance::testing::MakeDataset;

TEST_CASE("Use updater to update one column") {
  auto ints = views::iota(0, 100) | to<std::vector<int>>();
  auto ints_arr = ToArray(ints).ValueOrDie();
  auto schema = arrow::schema({arrow::field("ints", arrow::int32())});
  auto table = arrow::Table::Make(schema, {ints_arr});

  const int kMaxRowsPerGroup = 10;
  auto dataset = MakeDataset(table, {}, kMaxRowsPerGroup).ValueOrDie();
}