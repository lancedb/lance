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

#include "lance/arrow/type.h"

#include <arrow/type.h>

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;

TEST_CASE("Parse dictionary type") {
  auto dict_type = arrow::dictionary(arrow::uint16(), arrow::utf8(), false);
  auto logical_type = lance::arrow::ToLogicalType(dict_type).ValueOrDie();
  CHECK(logical_type == "dict:string:uint16:false");

  auto actual = lance::arrow::FromLogicalType(logical_type).ValueOrDie();
  CHECK(dict_type->Equals(actual));

  dict_type = arrow::dictionary(arrow::int32(), arrow::utf8(), true);
  logical_type = lance::arrow::ToLogicalType(dict_type).ValueOrDie();
  CHECK(logical_type == "dict:string:int32:true");

  actual = lance::arrow::FromLogicalType(logical_type).ValueOrDie();
  CHECK(dict_type->Equals(actual));
}