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

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test offsets") {
  auto offset = lance::io::Offset(100);
  CHECK(!offset.Execute(20).has_value());
  CHECK(!offset.Execute(70).has_value());
  CHECK(offset.Execute(30) == 10);
  /// After reaching the offset, it always returns zeros, the start of the chunk.
  CHECK(offset.Execute(15) == 0);
  CHECK(offset.Execute(200) == 0);
}