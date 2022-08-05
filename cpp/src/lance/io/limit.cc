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

#include "lance/io/limit.h"

#include <arrow/record_batch.h>
#include <fmt/format.h>

#include <algorithm>

#include "lance/io/reader.h"

namespace lance::io {

Limit::Limit(int64_t limit, int64_t offset) noexcept : limit_(limit), offset_(offset) {
  assert(offset >= 0);
  assert(limit >= 0);
}

std::optional<std::tuple<int64_t, int64_t>> Limit::Apply(int64_t length) {
  if (seen_ >= limit_ + offset_) {
    /// Already read all the data.
    return std::nullopt;
  }
  auto read_to = std::min(length, offset_ + limit_ - seen_);
  auto offset = std::max(static_cast<int64_t>(0), offset_ - seen_);
  seen_ += length;
  if (seen_ < offset_) {
    /// No data to read.
    return std::make_tuple(0, 0);
  }
  assert(read_to >= offset);
  return std::make_tuple(offset, read_to - offset);
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Limit::ReadBatch(
    const std::shared_ptr<FileReader>& reader, const lance::format::Schema& schema) {
  if (seen_ >= limit_) {
    return nullptr;
  }
  ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadAt(schema, offset_, limit_ - seen_));
  seen_ += batch->num_rows();
  return batch;
}

std::string Limit::ToString() const {
  return fmt::format("Limit(n={}, offset={})", limit_, offset_);
}

}  // namespace lance::io