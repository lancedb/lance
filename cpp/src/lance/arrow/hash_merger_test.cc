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

using lance::arrow::HashMerger;

template <ArrowType T>
std::shared_ptr<::arrow::Table> MakeTable() {
  std::vector<typename ::arrow::TypeTraits<T>::CType> keys;
  typename ::arrow::TypeTraits<T>::BuilderType keys_builder;
  typename ::arrow::StringBuilder value_builder;
  ::arrow::ArrayVector key_arrs, value_arrs;
  ///
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
void TestBuildHashMap() {
  auto table = MakeTable<T>();

  HashMerger merger(table, "keys");
  CHECK(merger.Init().ok());

  auto pk_arr =
      lance::arrow::ToArray<typename ::arrow::TypeTraits<T>::CType>({0, 3, 5, 10, 20}).ValueOrDie();
  auto result_batch = merger.Collect(pk_arr).ValueOrDie();
  fmt::print("Result: {}\n", result_batch->ToString());
}

TEST_CASE("Hash merge with primitive keys") {
  TestBuildHashMap<::arrow::UInt8Type>();
  TestBuildHashMap<::arrow::Int32Type>();
  TestBuildHashMap<::arrow::UInt64Type>();
  TestBuildHashMap<::arrow::FloatType>();
  TestBuildHashMap<::arrow::DoubleType>();
}

TEST_CASE("Hash merge with string keys") {}