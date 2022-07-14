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

#pragma once

#include <arrow/io/api.h>
#include <arrow/result.h>

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace lance::format {

namespace pb {
class Metadata;
}

class LookupTable {
 public:
  LookupTable() = default;

  void AddOffset(int32_t column, int32_t chunk, int64_t offset);

  void AddPageLength(int32_t column, int32_t chunk, int64_t length);

  /// Get the file offset of a chunk of a array.
  ///
  /// \param column_id the column / field ID.
  /// \param chunk_id the chunk id.
  /// \return file offset if available.
  std::optional<int64_t> GetOffset(int32_t column_id, int32_t chunk_id) const;

  ::arrow::Result<int64_t> GetPageLength(int32_t column_id, int32_t chunk_id) const;

  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::io::OutputStream> out);

  void WritePageLengthTo(pb::Metadata* out);

  /// Read lookup table from the opened file.
  ///
  static ::arrow::Result<std::shared_ptr<LookupTable>> Read(
      const std::shared_ptr<::arrow::io::RandomAccessFile>& in,
      int64_t offset,
      const pb::Metadata& pb);

 private:
  /// Map<column, Map<chunk, offset>>
  std::map<int32_t, std::map<int32_t, int64_t>> offsets_;

  /// Map<column, Map<chunk, length>>
  std::map<int32_t, std::map<int32_t, int64_t>> lengths_;

  std::vector<int64_t> page_lengths_;
};

}  // namespace lance::format
