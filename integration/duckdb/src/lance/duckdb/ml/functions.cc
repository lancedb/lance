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

#include "lance/duckdb/ml/functions.h"

#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <iostream>
#include <memory>
#include <vector>

#include "lance/duckdb/ml/pytorch.h"

namespace lance::duckdb::ml {

struct CreateModelFunctionData : public ::duckdb::TableFunctionData {
  CreateModelFunctionData() = default;

  std::string name;
  std::string uri;
  bool finished = false;
};

std::unique_ptr<::duckdb::FunctionData> CreateModelBind(
    ::duckdb::ClientContext& context,
    ::duckdb::TableFunctionBindInput& input,
    std::vector<::duckdb::LogicalType>& return_types,
    std::vector<std::string>& names) {
  auto result = std::make_unique<CreateModelFunctionData>();
  result->name = input.inputs[0].GetValue<std::string>();
  result->uri = input.inputs[1].GetValue<std::string>();

  return_types.push_back(::duckdb::LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

void CreateModelFunction(::duckdb::ClientContext& context,
                         ::duckdb::TableFunctionInput& data_p,
                         ::duckdb::DataChunk& output) {
  auto& data = (CreateModelFunctionData&)*data_p.bind_data;
  if (data.finished) {
    return;
  }

  auto catalog = ModelCatalog::Get();
  assert(catalog != nullptr);
  catalog->Load(data.name, data.uri);
  data.finished = true;
}

struct ShowModelsFunctionData : public ::duckdb::TableFunctionData {
  bool finished = false;
};

void ShowModelsFunction(::duckdb::ClientContext& context,
                        ::duckdb::TableFunctionInput& data_p,
                        ::duckdb::DataChunk& output) {
  auto data = const_cast<ShowModelsFunctionData*>(
      dynamic_cast<const ShowModelsFunctionData*>(data_p.bind_data));
  if (data->finished) {
    return;
  }

  auto catalog = ModelCatalog::Get();
  assert(catalog != nullptr);
  int i = 0;
  output.SetCardinality(catalog->models().size());
  for (const auto& [_, model] : catalog->models()) {
    output.SetValue(0, i, model->name());
    output.SetValue(1, i, model->uri());
    output.SetValue(2, i, model->type());
    i++;
  }
  data->finished = true;
}

std::unique_ptr<::duckdb::FunctionData> ShowModelsBind(
    ::duckdb::ClientContext& context,
    ::duckdb::TableFunctionBindInput& input,
    std::vector<::duckdb::LogicalType>& return_types,
    std::vector<std::string>& names) {
  auto result = std::make_unique<ShowModelsFunctionData>();

  for (const auto& col : {"name", "uri", "type"}) {
    return_types.push_back(::duckdb::LogicalType::VARCHAR);
    names.emplace_back(col);
  }
  return std::move(result);
}

struct DropModelFunctionData : public ::duckdb::TableFunctionData {
  std::string name;
  bool finished = false;
};

std::unique_ptr<::duckdb::FunctionData> DropModelBind(
    ::duckdb::ClientContext& context,
    ::duckdb::TableFunctionBindInput& input,
    std::vector<::duckdb::LogicalType>& return_types,
    std::vector<std::string>& names) {
  auto result = std::make_unique<DropModelFunctionData>();
  result->name = input.inputs[0].GetValue<std::string>();

  return_types.push_back(::duckdb::LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

void DropModelFunction(::duckdb::ClientContext& context,
                       ::duckdb::TableFunctionInput& data_p,
                       ::duckdb::DataChunk& output) {
  auto data = const_cast<DropModelFunctionData*>(
      dynamic_cast<const DropModelFunctionData*>(data_p.bind_data));
  //  if (data->finished) {
  //    return;
  //  }
  auto catalog = ModelCatalog::Get();
  assert(catalog != nullptr);
  catalog->Drop(data->name);
}

std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> GetMLFunctions() {
  std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> functions;

  for (auto& func : GetPyTorchFunctions()) {
    functions.emplace_back(std::move(func));
  }
  return functions;
}

std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> GetMLTableFunctions() {
  std::vector<std::unique_ptr<::duckdb::CreateTableFunctionInfo>> functions;

  ::duckdb::TableFunction create_model_func(
      "create_pytorch_model",
      {::duckdb::LogicalType::VARCHAR, ::duckdb::LogicalType::VARCHAR},
      CreateModelFunction,
      CreateModelBind);
  functions.emplace_back(std::make_unique<::duckdb::CreateTableFunctionInfo>(create_model_func));

  ::duckdb::TableFunction show_ml_models("ml_models", {}, ShowModelsFunction, ShowModelsBind);
  functions.emplace_back(std::make_unique<::duckdb::CreateTableFunctionInfo>(show_ml_models));

  ::duckdb::TableFunction drop_model(
      "drop_model", {::duckdb::LogicalType::VARCHAR}, DropModelFunction, DropModelBind);
  functions.emplace_back(std::make_unique<::duckdb::CreateTableFunctionInfo>(drop_model));
  return functions;
}

}  // namespace lance::duckdb::ml