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

#include <arrow/result.h>

#include <memory>

#include "lance/arrow/type.h"
#include "lance/arrow/utils.h"
#include "lance/io/exec/scan.h"

namespace lance::io::exec {

Take::Take(std::unique_ptr<ExecNode> child, std::unique_ptr<Scan> scan)
    : child_(std::move(child)), scan_(std::move(scan)) {
  assert(child_);
}

::arrow::Result<ScanBatch> Take::Next() {
  ARROW_ASSIGN_OR_RAISE(auto filtered, child_->Next());
  if (filtered.eof()) {
    return ScanBatch::Null();
  }
  assert(filtered.indices);
  const auto batch_id = filtered.batch_id;
  auto offset = filtered.offset;
  if (!scan_) {
    return ScanBatch(filtered.batch, batch_id, offset);
  }

  auto offset_datum = ::arrow::Datum(offset);
  ARROW_ASSIGN_OR_RAISE(auto adjusted_offsets,
                        ::arrow::compute::Add(filtered.indices, offset_datum));
  auto adjusted_offsets_arr =
      std::dynamic_pointer_cast<::arrow::Int32Array>(adjusted_offsets.make_array());
  ARROW_ASSIGN_OR_RAISE(auto rest_columns, scan_->Take(batch_id, adjusted_offsets_arr));
  assert(filtered.batch->num_rows() == rest_columns->num_rows());
  ARROW_ASSIGN_OR_RAISE(auto merged_batch,
                        lance::arrow::MergeRecordBatches(filtered.batch, rest_columns));
  return ScanBatch(merged_batch, filtered.batch_id, offset);
}

std::string Take::ToString() const { return "Take"; }

}  // namespace lance::io::exec