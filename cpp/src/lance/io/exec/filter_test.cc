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

#include "lance/io/exec/filter.h"

#include <arrow/array.h>
#include <arrow/compute/exec/expression.h>
#include <arrow/record_batch.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"
#include "lance/testing/io.h"

using ::arrow::compute::equal;
using ::arrow::compute::field_ref;
using ::arrow::compute::literal;
using ::arrow::compute::or_;
using lance::io::exec::Filter;
using lance::testing::TableScan;

TEST_CASE("Test with one condition") {
  auto filter = Filter::Make(equal(field_ref("value"), literal("32")), nullptr);
  INFO("filter " << filter.status().message());
  CHECK(filter.ok());

  filter = Filter::Make(equal(literal("32"), field_ref("value")), nullptr);
  INFO("filter " << filter.status().message());
  CHECK(filter.ok());
}

TEST_CASE("value = 32") {
  auto expr = equal(literal(32), field_ref("value"));
  auto bar = lance::arrow::ToArray({1, 2, 32, 0, 32}).ValueOrDie();
  auto struct_arr =
      ::arrow::StructArray::Make({bar}, {::arrow::field("value", ::arrow::int32())}).ValueOrDie();
  auto schema = ::arrow::schema({::arrow::field("value", ::arrow::int32())});
  auto table = ::arrow::Table::Make(schema, {bar});
  auto batch = ::arrow::RecordBatch::FromStructArray(struct_arr).ValueOrDie();

  auto filter = Filter::Make(expr, TableScan::Make(*table)).ValueOrDie();
  auto filtered_batch = filter->Next().ValueOrDie();
  auto output = filtered_batch.batch->GetColumnByName("values");
  CHECK(filtered_batch.indices->Equals(lance::arrow::ToArray({2, 4}).ValueOrDie()));

  bar = lance::arrow::ToArray({32, 32}).ValueOrDie();
  auto expected = ::arrow::RecordBatch::Make(schema, 2, {bar});
  CHECK(filtered_batch.batch->Equals(*expected));
}

TEST_CASE("label = cat or label = dog") {
  auto expr =
      or_(equal(field_ref("label"), literal("cat")), equal(field_ref("label"), literal("dog")));
  auto labels =
      lance::arrow::ToArray({"person", "dog", "cat", "car", "cat", "food", "hotdog"}).ValueOrDie();
  auto schema = ::arrow::schema({::arrow::field("label", ::arrow::utf8())});
  auto table = ::arrow::Table::Make(schema, {labels});

  auto filter = Filter::Make(expr, TableScan::Make(*table)).ValueOrDie();
  auto filtered_batch = filter->Next().ValueOrDie();
  CHECK(filtered_batch.indices->Equals(lance::arrow::ToArray({1, 2, 4}).ValueOrDie()));

  labels = lance::arrow::ToArray({"dog", "cat", "cat"}).ValueOrDie();
  auto expected = ::arrow::RecordBatch::Make(schema, 3, {labels});
  CHECK(filtered_batch.batch->Equals(*expected));
}