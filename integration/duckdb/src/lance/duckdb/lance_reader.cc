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

#include <arrow/array.h>
#include <arrow/dataset/scanner.h>
#include <arrow/filesystem/api.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <lance/arrow/dataset.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "lance/duckdb/lance.h"

namespace lance::duckdb {

namespace {

struct GlobalScanState : public ::duckdb::GlobalTableFunctionState {
  std::shared_ptr<lance::arrow::LanceDataset> dataset;
  ::arrow::dataset::TaggedRecordBatchGenerator batch_generator;
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
  std::string path;
  auto fs = GetResult(::arrow::fs::FileSystemFromUriOrPath(dataset_uri, &path));
  auto dataset = GetResult(lance::arrow::LanceDataset::Make(std::move(fs), path));
  auto schema = dataset->schema();
  auto bind_data = std::make_unique<ScanBindData>();
  bind_data->dataset = std::move(dataset);
  for (int i = 0; i < schema->fields().size(); ++i) {
    const auto &field = schema->field(i);
    names.emplace_back(field->name());
    return_types.emplace_back(ToLogicalType(*field->type()));
    bind_data->column_ids.emplace_back(i);
  }
  return std::move(bind_data);
}

std::unique_ptr<::duckdb::GlobalTableFunctionState> InitGlobal(
    ::duckdb::ClientContext &context, ::duckdb::TableFunctionInitInput &input) {
  auto bind_data = dynamic_cast<const ScanBindData *>(input.bind_data);
  assert(bind_data != nullptr);

  auto state = std::make_unique<GlobalScanState>();
  state->dataset = bind_data->dataset;

  auto schema = state->dataset->schema();
  std::vector<std::string> columns;
  for (auto &column_id : input.column_ids) {
    columns.emplace_back(schema->field(column_id)->name());
  }

  auto builder = GetResult(state->dataset->NewScan());
  CheckStatus(builder->Project(columns));
  auto scanner = GetResult(builder->Finish());
  state->batch_generator = GetResult(scanner->ScanBatchesAsync());
  fmt::print("Columns ids: {}\n", input.column_ids);
  return state;
}

/// Convert numeric array to duckdb vector.
template <typename ArrowType>
void NumericArrayToVector(const std::shared_ptr<::arrow::Array> &arr, ::duckdb::Vector *out) {
  // TODO: dynamic_pointer_cast does not work here, IDK why.
  auto array = std::static_pointer_cast<typename ::arrow::TypeTraits<ArrowType>::ArrayType>(arr);
  assert(array != nullptr);
  // TODO: How to use zero copy to move data from arrow to duckdb.
  for (int i = 0; i < array->length(); ++i) {
    out->SetValue(i, ::duckdb::Value::CreateValue(array->Value(i)));
  }
  out->SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
}

/// Convert a `arrow::Array` to `duckdb::Vector`.
void ArrowArrayToVector(const std::shared_ptr<::arrow::Array> &arr, ::duckdb::Vector *out) {
  switch (arr->type_id()) {
    case ::arrow::Type::UINT8:
      NumericArrayToVector<::arrow::UInt8Type>(arr, out);
      break;
    case ::arrow::Type::INT8:
      NumericArrayToVector<::arrow::Int8Type>(arr, out);
      break;
    case ::arrow::Type::UINT16:
      NumericArrayToVector<::arrow::UInt16Type>(arr, out);
      break;
    case ::arrow::Type::INT16:
      NumericArrayToVector<::arrow::Int16Type>(arr, out);
      break;
    case ::arrow::Type::UINT32:
      NumericArrayToVector<::arrow::UInt32Type>(arr, out);
      break;
    case ::arrow::Type::INT32:
      NumericArrayToVector<::arrow::Int32Type>(arr, out);
      break;
    case ::arrow::Type::UINT64:
      NumericArrayToVector<::arrow::UInt64Type>(arr, out);
      break;
    case ::arrow::Type::INT64:
      NumericArrayToVector<::arrow::Int64Type>(arr, out);
      break;
    case ::arrow::Type::FLOAT:
      NumericArrayToVector<::arrow::FloatType>(arr, out);
      break;
    case ::arrow::Type::DOUBLE:
      NumericArrayToVector<::arrow::FloatType>(arr, out);
      break;
    default:
      throw ::duckdb::IOException("Unsupported type: " + arr->type()->ToString());
  }
}

void LanceScan(::duckdb::ClientContext &context,
               ::duckdb::TableFunctionInput &input,
               ::duckdb::DataChunk &output) {
  auto global_state = dynamic_cast<const GlobalScanState *>(input.global_state);
  auto fut = global_state->batch_generator();
  auto batch = GetResult(fut.MoveResult());
  if (batch.record_batch == nullptr) {
    return;
  }
  output.SetCardinality(batch.record_batch->num_rows());
  for (int i = 0; i < output.data.size(); ++i) {
    auto col = batch.record_batch->column(i);
    ArrowArrayToVector(col, &output.data[i]);
  }
}

}  // namespace

::duckdb::TableFunctionSet GetLanceReaderFunction() {
  ::duckdb::TableFunctionSet func_set("lance_scan");

  ::duckdb::TableFunction table_function(
      {::duckdb::LogicalType::VARCHAR}, LanceScan, LanceScanBind, InitGlobal);
  table_function.projection_pushdown = true;
  table_function.filter_pushdown = true;
  table_function.filter_prune = true;

  func_set.AddFunction(table_function);
  return func_set;
}

}  // namespace lance::duckdb