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

#include "lance/duckdb/vector_functions.h"

#include <cstdint>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <memory>

namespace lance::duckdb {

template <typename T>
void L2Distance(::duckdb::DataChunk &args,
                ::duckdb::ExpressionState &state,
                ::duckdb::Vector &result) {
  result.SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
  for (::duckdb::idx_t i = 0; i < args.size(); i++) {
    auto first = ::duckdb::ListValue::GetChildren(args.data[0].GetValue(i));
    auto second = ::duckdb::ListValue::GetChildren(args.data[1].GetValue(i));
    assert(first.size() == second.size());
    // TODO: vectorize this computation later
    // To vectorize the compute, we need either have arrow Functions `Pow(list[float],
    // list[float])`, `Sub(list[float], list[float])`, or the equivalent in DuckDB.
    float sum = 0;
    for (size_t j = 0; j < first.size(); j++) {
      sum += pow((first[j].GetValue<T>() - second[j].GetValue<T>()), 2);
    }
    result.SetValue(i, sum);
  }
}

void IsInRectangle(::duckdb::DataChunk &args,
                   ::duckdb::ExpressionState &state,
                   ::duckdb::Vector &result) {
  result.SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
  for (::duckdb::idx_t i = 0; i < args.size(); i++) {
    auto point = ::duckdb::ListValue::GetChildren(args.data[0].GetValue(i));
    auto x = point[0];
    auto y = point[1];

    auto second = ::duckdb::ListValue::GetChildren(args.data[1].GetValue(i));
    auto upper_left = ::duckdb::ListValue::GetChildren(second[0]);
    auto lower_right = ::duckdb::ListValue::GetChildren(second[1]);

    result.SetValue(i,
                    ((upper_left[0] <= x) && (x <= lower_right[0]) && (upper_left[1] <= y) &&
                     (y <= lower_right[1])));
  }
}

std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> GetVectorFunctions() {
  std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> functions;

  ::duckdb::ScalarFunctionSet l2_distance("l2_distance");
  l2_distance.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::INTEGER),
                                ::duckdb::LogicalType::LIST(::duckdb::LogicalType::INTEGER)},
                               ::duckdb::LogicalType::INTEGER,
                               L2Distance<int>));
  l2_distance.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::BIGINT),
                                ::duckdb::LogicalType::LIST(::duckdb::LogicalType::BIGINT)},
                               ::duckdb::LogicalType::BIGINT,
                               L2Distance<int64_t>));
  l2_distance.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::FLOAT),
                                ::duckdb::LogicalType::LIST(::duckdb::LogicalType::FLOAT)},
                               ::duckdb::LogicalType::FLOAT,
                               L2Distance<float>));
  l2_distance.AddFunction(
      ::duckdb::ScalarFunction({::duckdb::LogicalType::LIST(::duckdb::LogicalType::DOUBLE),
                                ::duckdb::LogicalType::LIST(::duckdb::LogicalType::DOUBLE)},
                               ::duckdb::LogicalType::DOUBLE,
                               L2Distance<double>));
  functions.emplace_back(std::make_unique<::duckdb::CreateScalarFunctionInfo>(l2_distance));

  ::duckdb::ScalarFunctionSet in_rectangle("in_rectangle");
  in_rectangle.AddFunction(::duckdb::ScalarFunction(
      {::duckdb::LogicalType::LIST(::duckdb::LogicalType::INTEGER),
       ::duckdb::LogicalType::LIST(::duckdb::LogicalType::LIST(::duckdb::LogicalType::INTEGER))},
      ::duckdb::LogicalType::BOOLEAN,
      IsInRectangle));
  in_rectangle.AddFunction(::duckdb::ScalarFunction(
      {::duckdb::LogicalType::LIST(::duckdb::LogicalType::BIGINT),
       ::duckdb::LogicalType::LIST(::duckdb::LogicalType::LIST(::duckdb::LogicalType::BIGINT))},
      ::duckdb::LogicalType::BOOLEAN,
      IsInRectangle));
  in_rectangle.AddFunction(::duckdb::ScalarFunction(
      {::duckdb::LogicalType::LIST(::duckdb::LogicalType::FLOAT),
       ::duckdb::LogicalType::LIST(::duckdb::LogicalType::LIST(::duckdb::LogicalType::FLOAT))},
      ::duckdb::LogicalType::BOOLEAN,
      IsInRectangle));
  in_rectangle.AddFunction(::duckdb::ScalarFunction(
      {::duckdb::LogicalType::LIST(::duckdb::LogicalType::DOUBLE),
       ::duckdb::LogicalType::LIST(::duckdb::LogicalType::LIST(::duckdb::LogicalType::DOUBLE))},
      ::duckdb::LogicalType::BOOLEAN,
      IsInRectangle));
  functions.emplace_back(std::make_unique<::duckdb::CreateScalarFunctionInfo>(in_rectangle));

  return functions;
}

}  // namespace lance::duckdb