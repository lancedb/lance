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

#include "lance/format/lookup_table.h"

#include <arrow/builder.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <memory>
#include <vector>

#include "lance/format/format.pb.h"

namespace lance::format {

void LookupTable::AddOffset(int32_t column, int32_t chunk, int64_t offset) {
  offsets_[column][chunk] = offset;
}

void LookupTable::AddPageLength(int32_t column, int32_t chunk, int64_t length) {
  lengths_[column][chunk] = length;
}

::arrow::Result<int64_t> LookupTable::GetPageLength(int32_t column_id, int32_t chunk_id) const {
  auto it = lengths_.find(column_id);
  if (it == lengths_.end()) {
    return ::arrow::Status::IndexError(fmt::format(
        "LookupTable::GetPageLength:: column={} chunk={} does not exist.", column_id, chunk_id));
  }
  auto chunk_iter = it->second.find(chunk_id);
  if (chunk_iter == it->second.end()) {
    return ::arrow::Status::IndexError(fmt::format(
        "LookupTable::GetPageLength:: column={} chunk={} does not exist.", column_id, chunk_id));
  }
  return chunk_iter->second;
}

::arrow::Result<int64_t> LookupTable::Write(std::shared_ptr<::arrow::io::OutputStream> out) {
  ::arrow::Int64Builder builder;

  auto columns = offsets_.rbegin()->first + 1;
  int chunks = 0;
  for (auto& [k, m] : offsets_) {
    chunks = std::max(chunks, m.rbegin()->first + 1);
  }

  ARROW_RETURN_NOT_OK(builder.Reserve(columns * chunks));
  for (int col = 0; col < columns; ++col) {
    for (int chk = 0; chk < chunks; ++chk) {
      ARROW_RETURN_NOT_OK(builder.Append(GetOffset(col, chk).value_or(-1)));
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto chunk_positions, builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto pos, out->Tell());
  ARROW_RETURN_NOT_OK(out->Write(&columns, sizeof(columns)));
  ARROW_RETURN_NOT_OK(out->Write(&chunks, sizeof(chunks)));
  ARROW_RETURN_NOT_OK(
      out->Write(std::static_pointer_cast<::arrow::Int64Array>(chunk_positions)->values()));
  return pos;
}

::arrow::Result<std::shared_ptr<LookupTable>> LookupTable::Read(
    const std::shared_ptr<::arrow::io::RandomAccessFile>& in,
    int64_t position,
    const pb::Metadata& pb) {
  int columns, chunks;
  ARROW_RETURN_NOT_OK(in->ReadAt(position, sizeof(columns), &columns));
  ARROW_RETURN_NOT_OK(in->ReadAt(position + sizeof(columns), sizeof(chunks), &chunks));

  ARROW_ASSIGN_OR_RAISE(auto buf,
                        in->ReadAt(position + sizeof(columns) + sizeof(chunks),
                                   (columns * chunks * sizeof(int64_t))));

  auto arr = ::arrow::Int64Array(columns * chunks, buf);

  auto lt = std::make_shared<LookupTable>();
  for (int col = 0; col < columns; col++) {
    for (int ch = 0; ch < chunks; ch++) {
      auto idx = col * chunks + ch;
      lt->offsets_[col][ch] = arr.Value(idx);
      lt->lengths_[col][ch] = pb.page_lengths(idx);
    }
  }
  return lt;
}

std::optional<int64_t> LookupTable::GetOffset(int32_t column_id, int32_t chunk_id) const {
  auto it = offsets_.find(column_id);
  if (it == offsets_.end()) {
    return std::nullopt;
  }
  auto chunk_iter = it->second.find(chunk_id);
  if (chunk_iter == it->second.end()) {
    return std::nullopt;
  }
  return chunk_iter->second;
}

void LookupTable::WritePageLengthTo(pb::Metadata* out) {
  assert(out != nullptr);
  auto columns = lengths_.rbegin()->first + 1;
  int chunks = 0;
  for (auto& [k, m] : lengths_) {
    chunks = std::max(chunks, m.rbegin()->first + 1);
  }

  for (int col = 0; col < columns; ++col) {
    for (int chk = 0; chk < chunks; ++chk) {
      auto result = GetPageLength(col, chk);
      if (result.status().IsIndexError()) {
        out->add_page_lengths(0);
      } else {
        out->add_page_lengths(*result);
      }
    }
  }
};

}  // namespace lance::format