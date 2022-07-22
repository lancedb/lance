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

#include "lance/io/limit.h"

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <numeric>

TEST_CASE("LIMIT 100") {
  auto limit = lance::io::Limit(100);
  CHECK(limit.Apply(10).value() == std::make_tuple(0, 10));
  CHECK(limit.Apply(80).value() == std::make_tuple(0, 80));
  CHECK(limit.Apply(20).value() == std::make_tuple(0, 10));
  // Limit already reached.
  CHECK(limit.Apply(30) == std::nullopt);
}

TEST_CASE("LIMIT 10 OFFSET 20") {
  auto limit = lance::io::Limit(10, 20);
  CHECK(limit.Apply(10).value() == std::make_tuple(0, 0));
  CHECK(limit.Apply(5).value() == std::make_tuple(0, 0));
  auto val = limit.Apply(20).value();
  INFO("Limit::Apply(20): offset=" << std::get<0>(val) << " len=" << std::get<1>(val));
  CHECK(val == std::make_tuple(5, 10));
  CHECK(limit.Apply(5) == std::nullopt);
}