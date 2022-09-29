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

#include "lance/testing/json.h"

#include <arrow/ipc/json_simple.h>
#include <arrow/type.h>

#include <vector>

namespace lance::testing {

::arrow::Result<std::shared_ptr<::arrow::Array>> ArrayFromJSON(
    const std::shared_ptr<::arrow::DataType>& type, const std::string& json) {
  return ::arrow::ipc::internal::json::ArrayFromJSON(type, json);
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> RecordBatchFromJSON(
    const std::shared_ptr<::arrow::Schema>& schema, const std::string& json) {
  // Parse as a StructArray
  auto struct_type = ::arrow::struct_(schema->fields());
  ARROW_ASSIGN_OR_RAISE(auto struct_array, ArrayFromJSON(struct_type, json));

  // Convert StructArray to RecordBatch
  return ::arrow::RecordBatch::FromStructArray(struct_array);
}

::arrow::Result<std::shared_ptr<::arrow::Table>> TableFromJSON(
    const std::shared_ptr<::arrow::Schema>& schema, const std::vector<std::string>& json) {
  std::vector<std::shared_ptr<::arrow::RecordBatch>> batches;
  for (auto& array_json : json) {
    ARROW_ASSIGN_OR_RAISE(auto batch, RecordBatchFromJSON(schema, array_json));
    batches.emplace_back(batch);
  }
  return ::arrow::Table::FromRecordBatches(batches);
}

::arrow::Result<std::shared_ptr<::arrow::Table>> TableFromJSON(
    const std::shared_ptr<::arrow::Schema>& schema, const std::string& json) {
  return TableFromJSON(schema, std::vector<std::string>({json}));
}

}  // namespace lance::testing