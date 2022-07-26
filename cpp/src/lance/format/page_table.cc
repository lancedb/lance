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

#include "lance/format/page_table.h"

#include <arrow/builder.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <memory>
#include <vector>

#include "lance/format/format.pb.h"

namespace lance::format {

void PageTable::SetPageInfo(int32_t column_id,
                            int32_t batch_id,
                            int64_t position,
                            int64_t length) noexcept {
  page_info_map_[column_id][batch_id] = std::make_tuple(position, length);
}

/// Get PageInfo
std::optional<PageTable::PageInfo> PageTable::GetPageInfo(int32_t column_id,
                                                          int32_t batch_id) const noexcept {
  auto column_it = page_info_map_.find(column_id);
  if (column_it == page_info_map_.end()) {
    return std::nullopt;
  }
  auto page_it = column_it->second.find(batch_id);
  if (page_it == column_it->second.end()) {
    return std::nullopt;
  }
  return page_it->second;
}

::arrow::Result<int64_t> PageTable::Write(const std::shared_ptr<::arrow::io::OutputStream>& out) {
  ::arrow::Int64Builder builder;

  auto num_columns = page_info_map_.rbegin()->first + 1;
  int32_t num_batches = 0;
  for (auto& [k, m] : page_info_map_) {
    num_batches = std::max(num_batches, m.rbegin()->first + 1);
  }

  ARROW_RETURN_NOT_OK(builder.Reserve(num_columns * num_batches * 2));
  for (int32_t column_id = 0; column_id < num_columns; ++column_id) {
    for (int32_t batch_id = 0; batch_id < num_batches; ++batch_id) {
      auto page_info = GetPageInfo(column_id, batch_id);
      if (page_info.has_value()) {
        ARROW_RETURN_NOT_OK(builder.Append(std::get<0>(page_info.value())));
        ARROW_RETURN_NOT_OK(builder.Append(std::get<1>(page_info.value())));
      }
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto page_table, builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto pos, out->Tell());
  ARROW_RETURN_NOT_OK(
      out->Write(std::static_pointer_cast<::arrow::Int64Array>(page_table)->values()));
  return pos;
}

::arrow::Result<std::shared_ptr<PageTable>> PageTable::Make(
    const std::shared_ptr<::arrow::io::RandomAccessFile>& in,
    int64_t page_table_position,
    int32_t num_columns,
    int32_t num_batches) {
  int columns, chunks;
  ARROW_ASSIGN_OR_RAISE(auto buf,
                        in->ReadAt(page_table_position,
                                   (columns * chunks * sizeof(int64_t) * 2)));

  auto arr = ::arrow::Int64Array(columns * chunks, buf);

  auto lt = std::make_shared<PageTable>();
  for (int col = 0; col < num_columns; col++) {
    for (int ch = 0; ch < num_batches; ch++) {
      auto idx = col * num_batches + ch;
      auto position = arr.Value(idx * 2);
      auto length = arr.Value(idx * 2 + 1);
      lt->SetPageInfo(col, ch, position, length);
    }
  }
  return lt;
}


}  // namespace lance::format