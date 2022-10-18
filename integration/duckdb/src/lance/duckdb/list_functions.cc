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

#include "lance/duckdb/list_functions.h"

#include <cstdint>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <memory>

namespace lance::duckdb {

template <typename T>
void ListArgMax(::duckdb::DataChunk &args,
                ::duckdb::ExpressionState &state,
                ::duckdb::Vector &result) {
  result.SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
  for (::duckdb::idx_t i = 0; i < args.size(); i++) {
    // TODO: vectorize argmax.
    auto values = ::duckdb::ListValue::GetChildren(args.data[0].GetValue(i));
    auto max_iter = std::max_element(std::begin(values), std::end(values), [](auto a, auto b) {
      return a.template GetValue<T>() < b.template GetValue<T>();
    });
    auto idx_max = std::distance(std::begin(values), max_iter);
    result.SetValue(i, idx_max);
  }
}

std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> GetListFunctions() {
  std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> functions;

  ::duckdb::ScalarFunctionSet list_argmax("list_argmax");
  list_argmax.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::BIGINT)},
                               ::duckdb::LogicalType::INTEGER,
                               ListArgMax<int64_t>));
  list_argmax.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::INTEGER)},
                               ::duckdb::LogicalType::INTEGER,
                               ListArgMax<int>));
  list_argmax.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::FLOAT)},
                               ::duckdb::LogicalType::INTEGER,
                               ListArgMax<float>));
  list_argmax.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::DOUBLE)},
                               ::duckdb::LogicalType::INTEGER,
                               ListArgMax<double>));
  functions.emplace_back(std::make_unique<::duckdb::CreateScalarFunctionInfo>(list_argmax));

  return functions;
}

}  // namespace lance::duckdb
