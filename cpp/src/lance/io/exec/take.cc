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

#include "lance/io/exec/take.h"

#include "lance/arrow/utils.h"

namespace lance::io::exec {

Take::Take(std::shared_ptr<FileReader> reader,
           std::shared_ptr<lance::format::Schema> schema,
           std::unique_ptr<ExecNode> child)
    : reader_(std::move(reader)), schema_(std::move(schema)), child_(std::move(child)) {
  assert(child_);
}

::arrow::Result<std::unique_ptr<Take>> Take::Make(std::shared_ptr<FileReader> reader,
                                                  std::shared_ptr<lance::format::Schema> schema,
                                                  std::unique_ptr<ExecNode> child) {
  if (!child) {
    return ::arrow::Status::Invalid("Take::Make: child can not be null");
  }
  return std::unique_ptr<Take>(new Take(reader, schema, std::move(child)));
}

::arrow::Result<ScanBatch> Take::Next() {
  ARROW_ASSIGN_OR_RAISE(auto filtered, child_->Next());
  if (filtered.eof()) {
    return ScanBatch{};
  }
  auto indices = filtered.batch->GetColumnByName("indices");
  auto vals = filtered.batch->GetColumnByName("values");
  if (!indices || indices->type_id() != ::arrow::Type::INT32 || !vals ||
      vals->type_id() != ::arrow::Type::STRUCT) {
    return ::arrow::Status::Invalid("Invalid data from filter node: batch=",
                                    filtered.batch->ToString());
  }
  auto values = std::reinterpret_pointer_cast<::arrow::StructArray>(vals);
  if (!schema_ || schema_->fields().empty()) {
    return ScanBatch{::arrow::RecordBatch::FromStructArray(vals).ValueOrDie(), filtered.batch_id};
  } else {
    auto& batch_id = filtered.batch_id;
    auto int32_indices = std::dynamic_pointer_cast<::arrow::Int32Array>(indices);
    ARROW_ASSIGN_OR_RAISE(auto filtered_record_batch, ::arrow::RecordBatch::FromStructArray(vals));
    ARROW_ASSIGN_OR_RAISE(auto batch, reader_->ReadBatch(*schema_, batch_id, int32_indices));
    assert(filtered_record_batch->num_rows() == batch->num_rows());
    fmt::print("Merge scan results: filtered={} extra={}\n",
               filtered_record_batch->ToString(),
               batch->schema()->ToString());
    ARROW_ASSIGN_OR_RAISE(auto merged,
                          lance::arrow::MergeRecordBatches(filtered_record_batch, batch));
    return ScanBatch{merged, filtered.batch_id};
  }
}

std::string Take::ToString() const { return "Take"; }

}  // namespace lance::io::exec