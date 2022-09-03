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
    return ScanBatch::Null();
  }
  assert(filtered.indices);
  const auto batch_id = filtered.batch_id;
  if (!schema_ || schema_->fields().empty()) {
    return ScanBatch(filtered.batch, batch_id);
  } else {
    ARROW_ASSIGN_OR_RAISE(auto rest_columns,
                          reader_->ReadBatch(*schema_, batch_id, filtered.indices));
    assert(filtered.batch->num_rows() == rest_columns->num_rows());
    ARROW_ASSIGN_OR_RAISE(auto merged_batch,
                          lance::arrow::MergeRecordBatches(filtered.batch, rest_columns));
    return ScanBatch(merged_batch, filtered.batch_id);
  }
}

std::string Take::ToString() const { return "Take"; }

}  // namespace lance::io::exec