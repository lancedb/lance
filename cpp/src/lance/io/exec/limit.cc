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

#include "lance/io/exec/limit.h"

#include <arrow/record_batch.h>
#include <fmt/format.h>

#include <algorithm>

#include "lance/io/exec/scan.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

::arrow::Result<std::unique_ptr<ExecNode>> Limit::Make(int64_t limit,
                                                       int64_t offset,
                                                       std::unique_ptr<ExecNode> child) noexcept {
  auto limit_node = std::make_unique<Limit>(limit, offset, std::move(child));
  if (limit_node->child_->type() == kScan) {
    auto scan = dynamic_cast<Scan*>(limit_node->child_.get());
    ARROW_RETURN_NOT_OK(scan->Seek(offset));
    limit_node->seen_ = offset;
  }
  return std::unique_ptr<ExecNode>(limit_node.release());
}

Limit::Limit(int64_t limit, int64_t offset, std::unique_ptr<ExecNode> child) noexcept
    : limit_(limit), offset_(offset), child_(std::move(child)) {
  assert(offset >= 0);
  assert(limit >= 0);
}

::arrow::Result<ScanBatch> Limit::Next() {
  if (seen_ >= offset_ + limit_) {
    return ScanBatch{};
  }
  ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
  if (batch.eof()) {
    return batch;
  }
  // Find intersection of two ranges (offset, offset + limit) and (seen, seen + batch_size).
  auto batch_size = batch.batch->num_rows();
  auto left = std::max(offset_, seen_);
  auto right = std::min(seen_ + batch_size, offset_ + limit_);
  std::shared_ptr<::arrow::RecordBatch> record_batch;
  if (left < right) {
    record_batch = batch.batch->Slice(left - seen_, right - left);
  } else {
    /// No intersection, skip the whole batch.
    ARROW_ASSIGN_OR_RAISE(record_batch, ::arrow::RecordBatch::MakeEmpty(batch.batch->schema()));
  }
  seen_ += batch_size;
  return ScanBatch{record_batch, batch.batch_id};
}

std::string Limit::ToString() const {
  return fmt::format("Limit(n={}, offset={})", limit_, offset_);
}

}  // namespace lance::io::exec