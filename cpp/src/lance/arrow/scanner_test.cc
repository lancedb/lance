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

#include "lance/arrow/scanner.h"

#include <arrow/builder.h>
#include <arrow/type.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/type.h"

auto nested_schema = ::arrow::schema({::arrow::field("objects",
                                                     ::arrow::list(::arrow::struct_({
                                                         ::arrow::field("val", ::arrow::int64()),
                                                     })))});

TEST_CASE("Project nested columns") {
  auto schema = ::arrow::schema({::arrow::field("objects",
                                                ::arrow::list(::arrow::struct_({
                                                    ::arrow::field("val", ::arrow::int64()),
                                                })))});
  auto fields = schema->GetFieldByName("objects");
  fmt::print("Fields: {} {}\n", fields, fields->type()->field(0)->type()->field(0));

  auto ref = ::arrow::FieldRef("objects", 0, "val");
  auto f = ref.FindOne(*schema).ValueOrDie();
  fmt::print("FindAll: {}\n", f.ToString());
  CHECK(!f.empty());
}

TEST_CASE("Build Scanner with nested struct") {
  //  auto scanner_builder = lance::arrow::ScannerBuilder();
}