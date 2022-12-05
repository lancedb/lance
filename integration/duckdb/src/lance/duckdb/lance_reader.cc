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
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <memory>
#include <string>
#include <vector>

#include "lance/arrow/type.h"
#include "lance/duckdb/lance.h"

namespace lance::duckdb {

namespace {

// Forward declaration
void ArrowArrayToVector(const std::shared_ptr<::arrow::Array> &arr, ::duckdb::Vector *out);

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
  return state;
}

/// Convert numeric array to duckdb vector.
template <ArrowType T>
void ToVector(const std::shared_ptr<::arrow::Array> &arr, ::duckdb::Vector *out) {
  // TODO: dynamic_pointer_cast does not work here, IDK why.
  auto array = std::static_pointer_cast<typename ::arrow::TypeTraits<T>::ArrayType>(arr);
  assert(array != nullptr);
  // TODO: How to use zero copy to move data from arrow to duckdb.
  for (int i = 0; i < array->length(); ++i) {
    out->SetValue(i, ::duckdb::Value::CreateValue(array->Value(i)));
  }
  out->SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
}

/// Convert a String array into duckdb vector.
template <>
void ToVector<::arrow::StringType>(const std::shared_ptr<::arrow::Array> &arr,
                                   ::duckdb::Vector *out) {
  auto array = std::static_pointer_cast<::arrow::StringArray>(arr);
  assert(array != nullptr);
  // TODO: How to use zero copy to move data from arrow to duckdb.
  for (int i = 0; i < array->length(); ++i) {
    out->SetValue(i, std::string(array->Value(i)));
  }
  out->SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
}

/// Convert a Binary array into duckdb vector.
template <>
void ToVector<::arrow::BinaryType>(const std::shared_ptr<::arrow::Array> &arr,
                                   ::duckdb::Vector *out) {
  auto array = std::static_pointer_cast<::arrow::BinaryArray>(arr);
  assert(array != nullptr);
  // TODO: How to use zero copy to move data from arrow to duckdb.
  for (int i = 0; i < array->length(); ++i) {
    auto val = array->Value(i);
    out->SetValue(i, ::duckdb::Value::BLOB((::duckdb::data_ptr_t)val.data(), val.size()));
  }
  out->SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
}

template <>
void ToVector<::arrow::DictionaryType>(const std::shared_ptr<::arrow::Array> &arr,
                                       ::duckdb::Vector *out) {
  auto array = std::static_pointer_cast<::arrow::DictionaryArray>(arr);
  // TODO: zero copy
  out->SetVectorType(::duckdb::VectorType::FLAT_VECTOR);
  auto dict_arr = std::dynamic_pointer_cast<::arrow::StringArray>(array->dictionary());
  auto indices_arr = std::static_pointer_cast<::arrow::Int8Array>(array->indices());
  for (int i = 0; i < indices_arr->length(); ++i) {
    auto idx = indices_arr->Value(i);
    out->SetValue(i, std::string(dict_arr->Value(idx)));
  }
}

/// Convert `arrow::Array` to duckdb Struct Vector.
template <>
void ToVector<::arrow::StructType>(const std::shared_ptr<::arrow::Array> &arr,
                                   ::duckdb::Vector *out) {
  assert(arr->type_id() == ::arrow::Type::STRUCT);
  auto struct_arr = std::static_pointer_cast<::arrow::StructArray>(arr);
  auto &vector_children = ::duckdb::StructVector::GetEntries(*out);

  // Sanity checks
  if (struct_arr->num_fields() != vector_children.size()) {
    throw ::duckdb::InvalidInputException("Struct fields are not expected: %lu != %lu",
                                          struct_arr->num_fields(),
                                          vector_children.size());
  }

  for (int i = 0; i < struct_arr->num_fields(); i++) {
    ArrowArrayToVector(struct_arr->field(i), vector_children[i].get());
  }
}

template <>
void ToVector<::arrow::ListType>(const std::shared_ptr<::arrow::Array> &arr,
                                 ::duckdb::Vector *out) {
  /// TODO: zero copy vector construction.
  assert(arr->type_id() == ::arrow::Type::LIST);
  auto list_arr = std::static_pointer_cast<::arrow::ListArray>(arr);
  for (int i = 0; i < list_arr->length(); ++i) {
    auto scalar = GetResult(list_arr->GetScalar(i));
    auto list_scalar = std::static_pointer_cast<::arrow::ListScalar>(scalar);
    ::duckdb::Vector elem_vector(ToLogicalType(*list_scalar->value->type()));
    ArrowArrayToVector(list_scalar->value, &elem_vector);
  }
}

template <>
void ToVector<::arrow::FixedSizeListType>(const std::shared_ptr<::arrow::Array> &arr,
                                          ::duckdb::Vector *out) {
  /// TODO: zero copy vector construction.
  assert(arr->type_id() == ::arrow::Type::FIXED_SIZE_LIST);
  auto list_arr = std::static_pointer_cast<::arrow::FixedSizeListArray>(arr);
  for (int i = 0; i < list_arr->length(); ++i) {
    auto scalar = GetResult(list_arr->GetScalar(i));
    auto list_scalar = std::static_pointer_cast<::arrow::FixedSizeListScalar>(scalar);
    ::duckdb::Vector elem_vector(ToLogicalType(*list_scalar->value->type()));
    ArrowArrayToVector(list_scalar->value, &elem_vector);
  }
}

/// Convert a `arrow::Array` to `duckdb::Vector`.
void ArrowArrayToVector(const std::shared_ptr<::arrow::Array> &arr, ::duckdb::Vector *out) {
  switch (arr->type_id()) {
    case ::arrow::Type::BOOL:
      ToVector<::arrow::BooleanType>(arr, out);
      break;
    case ::arrow::Type::UINT8:
      ToVector<::arrow::UInt8Type>(arr, out);
      break;
    case ::arrow::Type::INT8:
      ToVector<::arrow::Int8Type>(arr, out);
      break;
    case ::arrow::Type::UINT16:
      ToVector<::arrow::UInt16Type>(arr, out);
      break;
    case ::arrow::Type::INT16:
      ToVector<::arrow::Int16Type>(arr, out);
      break;
    case ::arrow::Type::UINT32:
      ToVector<::arrow::UInt32Type>(arr, out);
      break;
    case ::arrow::Type::INT32:
      ToVector<::arrow::Int32Type>(arr, out);
      break;
    case ::arrow::Type::UINT64:
      ToVector<::arrow::UInt64Type>(arr, out);
      break;
    case ::arrow::Type::INT64:
      ToVector<::arrow::Int64Type>(arr, out);
      break;
    case ::arrow::Type::FLOAT:
      ToVector<::arrow::FloatType>(arr, out);
      break;
    case ::arrow::Type::DOUBLE:
      ToVector<::arrow::FloatType>(arr, out);
      break;
    case ::arrow::Type::STRING:
      ToVector<::arrow::StringType>(arr, out);
      break;
    case ::arrow::Type::BINARY:
      ToVector<::arrow::BinaryType>(arr, out);
      break;
    case ::arrow::Type::DICTIONARY:
      ToVector<::arrow::DictionaryType>(arr, out);
      break;
    case ::arrow::Type::STRUCT:
      ToVector<::arrow::StructType>(arr, out);
      break;
    case ::arrow::Type::LIST:
      ToVector<::arrow::ListType>(arr, out);
      break;
    case ::arrow::Type::FIXED_SIZE_LIST:
      ToVector<::arrow::FixedSizeListType>(arr, out);
      break;
    default:
      throw ::duckdb::IOException("Unsupported Arrow Type: " + arr->type()->ToString());
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

std::unique_ptr<::duckdb::TableFunctionRef> LanceScanReplacement(
    ::duckdb::ClientContext &context,
    const ::std::string &table_name,
    ::duckdb::ReplacementScanData *data) {
  auto lower_name = ::duckdb::StringUtil::Lower(table_name);
  if (!::duckdb::StringUtil::EndsWith(lower_name, ".lance")) {
    return nullptr;
  }
  auto table_function = ::duckdb::make_unique<::duckdb::TableFunctionRef>();
  ::std::vector<::std::unique_ptr<::duckdb::ParsedExpression>> children;
  children.emplace_back(
      ::std::make_unique<::duckdb::ConstantExpression>(::duckdb::Value(table_name)));
  table_function->function =
      ::std::make_unique<::duckdb::FunctionExpression>("lance_scan", ::std::move(children));
  return table_function;
}

}  // namespace lance::duckdb