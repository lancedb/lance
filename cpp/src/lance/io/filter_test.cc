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

#include "lance/io/filter.h"

#include <arrow/array.h>
#include <arrow/compute/exec/expression.h>
#include <arrow/record_batch.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"
#include "lance/arrow/type.h"
#include "lance/format/schema.h"

using ::arrow::compute::equal;
using ::arrow::compute::field_ref;
using ::arrow::compute::literal;

const auto kSchema = lance::format::Schema(::arrow::schema(
    {::arrow::field("pk", ::arrow::int32()), ::arrow::field("value", ::arrow::int32())}));

TEST_CASE("Test without filter") {
  auto empty_filter =
      lance::io::Filter::Make(kSchema, ::arrow::compute::literal(true)).ValueOrDie();
  CHECK(empty_filter == nullptr);
}

TEST_CASE("Test with one condition") {
  auto filter = lance::io::Filter::Make(kSchema, equal(field_ref("value"), literal("32")));
  INFO("filter " << filter.status().message());
  CHECK(filter.ok());

  filter = lance::io::Filter::Make(kSchema, equal(literal("32"), field_ref("value")));
  INFO("filter " << filter.status().message());
  CHECK(filter.ok());
}

TEST_CASE("Test apply filter to array") {
  auto expr = equal(literal(32), field_ref("value"));
  auto bar = lance::arrow::ToArray({1, 2, 32, 0, 32}).ValueOrDie();
  auto struct_arr =
      ::arrow::StructArray::Make({bar}, {::arrow::field("value", ::arrow::int32())}).ValueOrDie();
  auto batch = ::arrow::RecordBatch::FromStructArray(struct_arr).ValueOrDie();

  auto filter = lance::io::Filter::Make(kSchema, expr).ValueOrDie();
  auto [mask, output] = filter->Exec(batch).ValueOrDie();
  fmt::print("mask is: {}, Output is: {}\n", mask, output);
}