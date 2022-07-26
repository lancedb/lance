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

#pragma once

#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/type.h>

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "lance/format/format.pb.h"

namespace lance::format {

class Manifest;

/// File Metadata.
class Metadata final {
 public:
  Metadata() = default;

  ~Metadata() = default;

  /// Parse a Metadata from an arrow buffer.
  static ::arrow::Result<std::shared_ptr<Metadata>> Make(
      const std::shared_ptr<::arrow::Buffer>& buffer);

  /// Get the number of batches in this file.
  int32_t num_batches() const;

  /// Add the length of the batch.
  void AddBatchLength(int32_t length);

  /// Get the logical length of a chunk.
  int32_t GetBatchLength(int32_t batch_id) const;

  /// Locate the chunk index where the idx belongs.
  ///
  /// \param idx the absolute index of a row in the file.
  /// \return a tuple of <chunk id, idx in the chunk>
  ::arrow::Result<std::tuple<int32_t, int32_t>> LocateChunk(int32_t idx) const;

  /// Get the number of records in this file.
  int64_t length() const;

  /// Get the file position to the page table.
  int64_t page_table_position() const;

  /// Set the position of the page table.
  void SetPageTablePosition(int64_t position);

  ::arrow::Result<std::shared_ptr<Manifest>> GetManifest(
      std::shared_ptr<::arrow::io::RandomAccessFile> in);

 private:
  pb::Metadata pb_;
};

}  // namespace lance::format
