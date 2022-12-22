//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/format/metadata.h"

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include <memory>
#include <tuple>

#include "lance/format/format.pb.h"
#include "lance/format/manifest.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;
using std::shared_ptr;

namespace lance::format {

Metadata::Metadata(std::vector<int32_t> batch_offsets,
                   int64_t page_length_position,
                   int64_t manifest_position)
    : batch_offsets_(std::move(batch_offsets)),
      page_table_position_(page_length_position),
      manifest_position_(manifest_position) {}

Result<shared_ptr<Metadata>> Metadata::Make(const shared_ptr<::arrow::Buffer>& buffer) {
  auto meta = std::make_unique<Metadata>();
  ARROW_ASSIGN_OR_RAISE(auto msg, io::ParseProto<pb::Metadata>(buffer));
  std::vector<int32_t> offsets(msg.batch_offsets().begin(), msg.batch_offsets().end());
  return std::make_unique<Metadata>(offsets, msg.page_table_position(), msg.manifest_position());
}

pb::Metadata Metadata::ToProto() const {
  lance::format::pb::Metadata proto;
  proto.set_page_table_position(page_table_position_);
  proto.set_manifest_position(manifest_position_);
  for (auto offset : batch_offsets_) {
    proto.add_batch_offsets(offset);
  }
  return proto;
}

int32_t Metadata::num_batches() const { return batch_offsets_.size() - 1; }

int64_t Metadata::length() const {
  if (batch_offsets_.empty()) {
    return 0;
  }
  return *batch_offsets_.rbegin();
}

void Metadata::AddBatchLength(int32_t batch_length) {
  assert(batch_length > 0);
  if (batch_offsets_.empty()) {
    batch_offsets_.emplace_back(0);
  }
  batch_offsets_.emplace_back(length() + batch_length);
}

int32_t Metadata::GetBatchLength(int32_t batch_id) const {
  assert(static_cast<std::size_t>(batch_id) < batch_offsets_.size() - 1);
  return batch_offsets_[batch_id + 1] - batch_offsets_[batch_id];
}

::arrow::Result<std::tuple<int32_t, int32_t>> Metadata::LocateBatch(int32_t row_index) const {
  int64_t len = length();
  if (len == 0) {
    return ::arrow::Status::IndexError("The offsets table is empty");
  }

  if (row_index < 0 || row_index >= len) {
    return ::arrow::Status::IndexError(
        fmt::format("Row index out of range: {} of {}", row_index, len - 1));
  }
  auto it = std::upper_bound(batch_offsets_.begin(), batch_offsets_.end(), row_index);
  if (it == batch_offsets_.end()) {
    return ::arrow::Status::IndexError("Row index out of range {} of {}", row_index, len);
  }
  int32_t bound_idx = std::distance(batch_offsets_.begin(), it) - 1;
  // Offset within the batch.
  int32_t offset = row_index - batch_offsets_[bound_idx];
  return std::tuple(bound_idx, offset);
}

void Metadata::SetManifestPosition(int64_t position) { manifest_position_ = position; }

int64_t Metadata::page_table_position() const { return page_table_position_; }

void Metadata::SetPageTablePosition(int64_t position) { page_table_position_ = position; }

int64_t Metadata::manifest_position() const { return manifest_position_; }

}  // namespace lance::format