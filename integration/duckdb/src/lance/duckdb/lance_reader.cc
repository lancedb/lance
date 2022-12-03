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

template <typename ArrowType>
void NumericArrayToVector(const std::shared_ptr<::arrow::Array> &arr, ::duckdb::Vector *out) {
  fmt::print("Numberic to vector, out={} arr->type={} template={} arr={}\n",
             fmt::ptr(out),
             arr->type()->ToString(),
             ArrowType().ToString(),
             fmt::ptr(arr));
  auto array = std::dynamic_pointer_cast<const typename ::arrow::TypeTraits<ArrowType>::ArrayType>(arr);
  assert(array != nullptr);
  // TODO: Use zero copy
  //  out->SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
  fmt::print("Add data: out={} length={}\n", fmt::ptr(out), fmt::ptr(array));
  for (int i = 0; i < array->length(); ++i) {
    out->SetValue(i, ::duckdb::Value::CreateValue(array->Value(i)));
  }
}

/// Convert a `arrow::Array` to `duckdb::Vector`.
::duckdb::Vector ArrowArrayToVector(const std::shared_ptr<::arrow::Array> &arr) {
  // TODO: optimize it for zero copy
  auto logical_type = ToLogicalType(*arr->type());
  ::duckdb::Vector result(logical_type);
  switch (arr->type_id()) {
    case ::arrow::Type::UINT8:
      NumericArrayToVector<::arrow::UInt8Type>(arr, &result);
      break;
    case ::arrow::Type::INT8:
      NumericArrayToVector<::arrow::Int8Type>(arr, &result);
      break;
    case ::arrow::Type::UINT16:
      NumericArrayToVector<::arrow::UInt16Type>(arr, &result);
      break;
    case ::arrow::Type::INT16:
      NumericArrayToVector<::arrow::Int16Type>(arr, &result);
      break;
    case ::arrow::Type::UINT32:
      NumericArrayToVector<::arrow::UInt32Type>(arr, &result);
      break;
    case ::arrow::Type::INT32:
      NumericArrayToVector<::arrow::Int32Type>(arr, &result);
      break;
    case ::arrow::Type::UINT64:
      NumericArrayToVector<::arrow::UInt64Type>(arr, &result);
      break;
    case ::arrow::Type::INT64:
      NumericArrayToVector<::arrow::Int64Type>(arr, &result);
      break;
    case ::arrow::Type::FLOAT:
      NumericArrayToVector<::arrow::FloatType>(arr, &result);
      break;
    case ::arrow::Type::DOUBLE:
      NumericArrayToVector<::arrow::FloatType>(arr, &result);
      break;
    default:
      throw ::duckdb::IOException("Unsupported type: " + arr->type()->ToString());
  }
  return std::move(result);
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
  fmt::print("Batch: {}\n", batch.record_batch->ToString());
  output.SetCapacity(batch.record_batch->num_rows());
  for (auto &col : batch.record_batch->columns()) {
    //    auto vec = ArrowArrayToVector(col);
    fmt::print("Convert to array: {}\n", col->ToString());
    output.data.emplace_back(ArrowArrayToVector(col));
    fmt::print("After convert\n");
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