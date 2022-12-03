// Copyright 2022 Lance Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "lance/duckdb/lance_reader.h"

#include <arrow/filesystem/api.h>
#include <lance/arrow/dataset.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "lance/duckdb/lance.h"

namespace lance::duckdb {

struct GlobalScanState : public ::duckdb::GlobalTableFunctionState {
  std::shared_ptr<lance::arrow::LanceDataset> dataset;
};

struct LocalScanState : public ::duckdb::LocalTableFunctionState {};

/// BindData for Lance Scan
struct ScanBindData : public ::duckdb::TableFunctionData {
  std::shared_ptr<lance::arrow::LanceDataset> dataset;
};

std::unique_ptr<::duckdb::FunctionData> LanceScanBind(
    ::duckdb::ClientContext &context,
    ::duckdb::TableFunctionBindInput &input,
    std::vector<::duckdb::LogicalType> &return_types,
    std::vector<std::string> &names) {
  auto dataset_uri = input.inputs[0].GetValue<std::string>();
  printf("Dataset uri is: %s\n", dataset_uri.c_str());

  std::string path;
  auto fs = GetResult(::arrow::fs::FileSystemFromUriOrPath(dataset_uri, &path));
  auto dataset = GetResult(lance::arrow::LanceDataset::Make(std::move(fs), path));
  auto schema = dataset->schema();
  auto bind_data = std::make_unique<ScanBindData>();
  for (int i = 0; i < schema->fields().size(); ++i) {
    const auto& field = schema->field(i);
    names.emplace_back(field->name());
    return_types.emplace_back(ToLogicalType(*field->type()));
    bind_data->column_ids.emplace_back(i);
  }
  return std::move(bind_data);
}

void LanceScan(::duckdb::ClientContext &context,
               ::duckdb::TableFunctionInput &input,
               ::duckdb::DataChunk &output) {}

::duckdb::TableFunctionSet GetLanceReaderFunction() {
  ::duckdb::TableFunctionSet func_set("lance_scan");

  ::duckdb::TableFunction table_function(
      {::duckdb::LogicalType::VARCHAR}, LanceScan, LanceScanBind);
  table_function.projection_pushdown = true;
  table_function.filter_pushdown = true;
  table_function.filter_prune = true;

  func_set.AddFunction(table_function);
  return func_set;
}

}  // namespace lance::duckdb