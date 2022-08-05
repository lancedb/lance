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
#include <memory>
#include <string>
#include <tuple>
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

// Type reference: https://arrow.apache.org/docs/cpp/api/datatype.html
TEST_CASE("Logical type coverage") {
  const auto kArrayTypeMap =
      std::vector<std::tuple<std::shared_ptr<::arrow::DataType>, std::string>>({
          // Primitive types
          {::arrow::null(), "null"},
          {::arrow::boolean(), "bool"},
          {::arrow::int8(), "int8"},
          {::arrow::uint8(), "uint8"},
          {::arrow::int16(), "int16"},
          {::arrow::uint16(), "uint16"},
          {::arrow::int32(), "int32"},
          {::arrow::uint32(), "uint32"},
          {::arrow::int64(), "int64"},
          {::arrow::uint64(), "uint64"},
          {::arrow::float16(), "halffloat"},
          {::arrow::float32(), "float"},
          {::arrow::float64(), "double"},
          {::arrow::utf8(), "string"},
          {::arrow::binary(), "binary"},
          {::arrow::large_utf8(), "large_string"},
          {::arrow::large_binary(), "large_binary"},
          {::arrow::fixed_size_binary(1234), "fixed_size_binary:1234"},
          // Time
          {::arrow::date32(), "date32:day"},
          {::arrow::date64(), "date64:ms"},
          {::arrow::timestamp(::arrow::TimestampType::Unit::SECOND), "timestamp:s"},
          {::arrow::timestamp(::arrow::TimestampType::Unit::MILLI), "timestamp:ms"},
          {::arrow::timestamp(::arrow::TimestampType::Unit::MICRO), "timestamp:us"},
          {::arrow::timestamp(::arrow::TimestampType::Unit::NANO), "timestamp:ns"},
          {::arrow::time32(::arrow::TimeUnit::SECOND), "time32:s"},
          {::arrow::time32(::arrow::TimeUnit::MILLI), "time32:ms"},
          {::arrow::time64(::arrow::TimeUnit::MICRO), "time64:us"},
          {::arrow::time64(::arrow::TimeUnit::NANO), "time64:ns"},
      });

  for (auto& [arrow_type, type_str] : kArrayTypeMap) {
    INFO("Data type: " << arrow_type->ToString() << " type string: " << type_str);
    CHECK(lance::arrow::ToLogicalType(arrow_type).ValueOrDie() == type_str);
    CHECK(lance::arrow::FromLogicalType(type_str).ValueOrDie()->Equals(arrow_type));
  }
}