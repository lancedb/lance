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

#include "lance/arrow/utils.h"

#include <arrow/type.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"
#include "lance/arrow/type.h"

TEST_CASE("Merge simple structs") {
  auto a = lance::arrow::ToArray<int32_t>({1, 2, 3}).ValueOrDie();
  auto left = ::arrow::StructArray::Make({a}, {"a"}).ValueOrDie();

  auto b = lance::arrow::ToArray({"One", "Two", "Three"}).ValueOrDie();
  auto right = ::arrow::StructArray::Make({b}, {"b"}).ValueOrDie();

  auto merged = lance::arrow::Merge(left, right).ValueOrDie();
  auto expected =
      ::arrow::StructArray::Make({a, b}, std::vector<std::string>({"a", "b"})).ValueOrDie();
  CHECK(merged->Equals(expected));
}