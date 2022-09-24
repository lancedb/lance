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

#include "lance/io/exec/base.h"

#include <arrow/array.h>

#include <memory>

namespace lance::io::exec {

ScanBatch ScanBatch::Null() { return ScanBatch(nullptr, -1); }

ScanBatch::ScanBatch(std::shared_ptr<::arrow::RecordBatch> records,
                     int32_t bid,
                     std::shared_ptr<::arrow::Int32Array> idx)
    : batch(std::move(records)), batch_id(bid), indices(std::move(idx)) {}

ScanBatch ScanBatch::Slice(int64_t offset, int64_t length) const {
  auto sliced_batch = batch->Slice(offset, length);
  decltype(indices) sliced_indices;
  if (indices) {
    sliced_indices = std::dynamic_pointer_cast<::arrow::Int32Array>(indices->Slice(offset, length));
  }
  return ScanBatch(sliced_batch, batch_id, sliced_indices);
}

int64_t ScanBatch::length() const {
  if (!batch) {
    return 0;
  }
  return batch->num_rows();
}

}  // namespace lance::io::exec