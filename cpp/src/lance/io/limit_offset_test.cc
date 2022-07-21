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

#include "lance/io/limit_offset.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

#include "lance/arrow/stl.h"

TEST_CASE("Test offsets") {
  auto offset = lance::io::Offset(100);
  CHECK(!offset.Execute(20).has_value());
  CHECK(!offset.Execute(70).has_value());
  CHECK(offset.Execute(30) == 10);
  /// After reaching the offset, it always returns zeros, the start of the chunk.
  CHECK(offset.Execute(15) == 0);
  CHECK(offset.Execute(200) == 0);
}

TEST_CASE("Test apply offset over arrays") {
  auto offset = lance::io::Offset(40);

  std::vector<int32_t> nums(20);
  std::iota(std::begin(nums), std::end(nums), 0);
  fmt::print("Numbers: {}\n", nums);
  auto arr = lance::arrow::ToArray(nums).ValueOrDie();

  CHECK(!offset.Execute(nullptr).operator bool());

  // [0, 19]
  CHECK(offset.Execute(arr)->length() == 0);
  // [20, 34]
  CHECK(offset.Execute(arr->Slice(0, 15))->length() == 0);
  // [35, 54]
  CHECK(offset.Execute(arr)->length() == 15);
  // [55, 75]
  CHECK(offset.Execute(arr)->length() == 20);
}