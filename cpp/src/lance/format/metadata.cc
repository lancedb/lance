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

#include "lance/format/format.h"
#include "lance/format/manifest.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;
using std::shared_ptr;

namespace lance::format {

Result<shared_ptr<Metadata>> Metadata::Make(const shared_ptr<::arrow::Buffer>& buffer) {
  auto meta = std::make_unique<Metadata>();
  auto msg = io::ParseProto<pb::Metadata>(buffer);
  if (!msg.ok()) {
    return msg.status();
  }
  meta->pb_ = std::move(*msg);
  return meta;
}

::arrow::Result<int64_t> Metadata::Write(const std::shared_ptr<::arrow::io::OutputStream>& out) {
  return io::WriteProto(out, pb_);
}

int32_t Metadata::num_batches() const { return pb_.batch_offsets_size() - 1; }

int64_t Metadata::length() const {
  if (pb_.batch_offsets_size() == 0) {
    return 0;
  }
  return pb_.batch_offsets(pb_.batch_offsets_size() - 1);
}

void Metadata::AddBatchLength(int32_t batch_length) {
  if (pb_.batch_offsets_size() == 0) {
    pb_.add_batch_offsets(0);
  }
  pb_.add_batch_offsets(length() + batch_length);
}

int32_t Metadata::GetBatchLength(int32_t batch_id) const {
  assert(batch_id <= pb_.batch_offsets_size());
  return pb_.batch_offsets(batch_id + 1) - pb_.batch_offsets(batch_id);
}

::arrow::Result<std::tuple<int32_t, int32_t>> Metadata::LocateBatch(int32_t row_index) const {
  int64_t len = length();
  if (len == 0) {
    return ::arrow::Status::IndexError("The offsets table is empty");
  }

  if (row_index < 0 || row_index >= len) {
    return ::arrow::Status::IndexError(fmt::format("Row index out of range: {} of {}", row_index, len - 1));
  }
  auto it = std::upper_bound(pb_.batch_offsets().begin(), pb_.batch_offsets().end(), row_index);
  if (it == pb_.batch_offsets().end()) {
    return ::arrow::Status::IndexError("Row index out of range {} of {}", row_index, len);
  }
  int32_t bound_idx = std::distance(pb_.batch_offsets().begin(), it) - 1;
  // Offset within the batch.
  int32_t offset = row_index - pb_.batch_offsets(bound_idx);
  return std::tuple(bound_idx, offset);
}

::arrow::Result<std::shared_ptr<Manifest>> Metadata::GetManifest(
    std::shared_ptr<::arrow::io::RandomAccessFile> in) {
  // TODO: change to read buffer instead of read file again.
  if (pb_.manifest_position() == 0) {
    return Status::IOError("Can not find manifest within the file");
  }
  return Manifest::Parse(in, pb_.manifest_position());
}

void Metadata::SetManifestPosition(int64_t position) { pb_.set_manifest_position(position); }

int64_t Metadata::page_table_position() const { return pb_.page_table_position(); }

void Metadata::SetPageTablePosition(int64_t position) { pb_.set_page_table_position(position); }

}  // namespace lance::format