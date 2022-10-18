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

#pragma once

#include <torch/torch.h>

#include <duckdb.hpp>
#include <duckdb/parser/parsed_data/create_function_info.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <memory>
#include <vector>

#include "lance/duckdb/ml/catalog.h"

namespace lance::duckdb::ml {

/// PyTorch / TorchScript model entry
class PyTorchModelEntry : ModelEntry {
 public:
  static std::unique_ptr<ModelEntry> Make(const std::string &name, const std::string &uri);

  std::string type() const override { return "torchscript"; }

  void Execute(::duckdb::DataChunk &args,
               ::duckdb::ExpressionState &state,
               ::duckdb::Vector &result) override;

 private:
  PyTorchModelEntry(std::string name, std::string uri, torch::jit::script::Module module)
      : ModelEntry(name, uri), module_(std::move(module)) {
    module_.eval();
  }

  std::string name_;
  std::string uri_;
  torch::jit::script::Module module_;
};

/// Get PyTorch functions
std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> GetPyTorchFunctions();


}  // namespace lance::duckdb::ml