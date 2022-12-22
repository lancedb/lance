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

#include "lance/arrow/hash_merger.h"

#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "lance/arrow/stl.h"
#include "lance/arrow/type.h"
#include "lance/testing/json.h"

using lance::arrow::HashMerger;

template <ArrowType T>
std::shared_ptr<::arrow::Table> MakeTable() {
  std::vector<typename ::arrow::TypeTraits<T>::CType> keys;
  typename ::arrow::TypeTraits<T>::BuilderType keys_builder;
  typename ::arrow::StringBuilder value_builder;
  ::arrow::ArrayVector key_arrs, value_arrs;
  for (int chunk = 0; chunk < 5; chunk++) {
    for (int i = 0; i < 10; i++) {
      typename ::arrow::TypeTraits<T>::CType value = chunk * 10 + i;
      CHECK(keys_builder.Append(value).ok());
      CHECK(value_builder.Append(fmt::format("{}", value)).ok());
    }
    auto keys_arr = keys_builder.Finish().ValueOrDie();
    auto values_arr = value_builder.Finish().ValueOrDie();
    key_arrs.emplace_back(keys_arr);
    value_arrs.emplace_back(values_arr);
  }
  auto keys_chunked_arr = std::make_shared<::arrow::ChunkedArray>(key_arrs);
  auto values_chunked_arr = std::make_shared<::arrow::ChunkedArray>(value_arrs);
  return ::arrow::Table::Make(::arrow::schema({::arrow::field("keys", std::make_shared<T>()),
                                               ::arrow::field("values", ::arrow::utf8())}),
                              {keys_chunked_arr, values_chunked_arr});
}

template <ArrowType T>
void TestMergeOnPrimitiveType() {
  auto table = MakeTable<T>();

  HashMerger merger(table, "keys");
  CHECK(merger.Init().ok());

  auto pk_arr =
      lance::arrow::ToArray<typename ::arrow::TypeTraits<T>::CType>({10, 20, 0, 5, 120, 32, 88})
          .ValueOrDie();

  ::arrow::StringBuilder values_builder;
  typename ::arrow::TypeTraits<T>::BuilderType key_builder;

  CHECK(values_builder.AppendValues({"10", "20", "0", "5"}).ok());
  CHECK(values_builder.AppendNull().ok());
  CHECK(values_builder.Append("32").ok());
  CHECK(values_builder.AppendNull().ok());
  auto values_arr = values_builder.Finish().ValueOrDie();

  auto result_batch = merger.Collect(pk_arr).ValueOrDie();
  auto expected =
      ::arrow::RecordBatch::Make(::arrow::schema({::arrow::field("values", ::arrow::utf8())}),
                                 values_arr->length(),
                                 {values_arr});
  CHECK(result_batch->Equals(*expected));
}

TEST_CASE("Hash merge with primitive keys") {
  TestMergeOnPrimitiveType<::arrow::UInt8Type>();
  TestMergeOnPrimitiveType<::arrow::Int8Type>();
  TestMergeOnPrimitiveType<::arrow::UInt16Type>();
  TestMergeOnPrimitiveType<::arrow::Int16Type>();
  TestMergeOnPrimitiveType<::arrow::Int32Type>();
  TestMergeOnPrimitiveType<::arrow::UInt32Type>();
  TestMergeOnPrimitiveType<::arrow::UInt64Type>();
  TestMergeOnPrimitiveType<::arrow::Int64Type>();
}

template <ArrowType T>
void TestMergeOnFloatType() {
  auto table =
      lance::testing::TableFromJSON(::arrow::schema({::arrow::field("a", std::make_shared<T>())}),
                                    R"([{"a": 1.0}, {"a": 2.0}])")
          .ValueOrDie();
  HashMerger merger(table, "a");
  CHECK(!merger.Init().ok());
}

TEST_CASE("Float keys are not supported") {
  TestMergeOnFloatType<::arrow::FloatType>();
  TestMergeOnFloatType<::arrow::DoubleType>();
};

TEST_CASE("Hash merge with string keys") {
  auto keys = lance::arrow::ToArray({"a", "b", "c", "d"}).ValueOrDie();
  auto values = lance::arrow::ToArray({1, 2, 3, 4}).ValueOrDie();
  auto schema = ::arrow::schema(
      {::arrow::field("keys", ::arrow::utf8()), ::arrow::field("values", ::arrow::int32())});
  auto table = ::arrow::Table::Make(schema, {keys, values});

  HashMerger merger(table, "keys");
  CHECK(merger.Init().ok());

  auto pk_arr = lance::arrow::ToArray({"c", "d", "e", "f", "a"}).ValueOrDie();
  auto batch = merger.Collect(pk_arr).ValueOrDie();
  CHECK(batch->num_columns() == 1);
  ::arrow::Int32Builder builder;
  CHECK(builder.AppendValues({3, 4}).ok());
  CHECK(builder.AppendNulls(2).ok());
  CHECK(builder.Append(1).ok());
  auto expected_values = builder.Finish().ValueOrDie();
  auto expected = ::arrow::RecordBatch::Make(
      ::arrow::schema({::arrow::field("values", ::arrow::int32())}), 5, {expected_values});
  CHECK(batch->Equals(*expected));
}