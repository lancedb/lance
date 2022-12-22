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
#include <tuple>
#include <vector>

namespace lance::format {

/// PageTable lookup table for pages.
class PageTable {
 public:
  /// Page info: [position, length].
  using PageInfo = std::tuple<int64_t, int64_t>;

  PageTable() = default;

  /// Make the page table from an opened file.
  ///
  /// \param in The input file to read
  /// \param page_table_position The file position to the page table.
  /// \param num_columns the total number of columns, including the nested columns.
  /// \param num_batches the total number of batches in the file.
  ///
  /// \return a LookupTable if success.
  static ::arrow::Result<std::shared_ptr<PageTable>> Make(
      const std::shared_ptr<::arrow::io::RandomAccessFile>& in,
      int64_t page_table_position,
      int32_t num_columns,
      int32_t num_batches);

  /// Set PageInfo.
  void SetPageInfo(int32_t column_id, int32_t batch_id, int64_t position, int64_t length) noexcept;

  /// Get PageInfo (a tuple of `[position, length]`) of a page.
  ///
  /// \param column_id the column / field ID.
  /// \param batch_id the ID of the batch
  /// \return a tuple of `[position, length]` if available. Can return `std::nullopt` if
  //          the page is virtual (i.e., parent field)
  std::optional<PageInfo> GetPageInfo(int32_t column_id, int32_t batch_id) const noexcept;

  /// Write PageTable to a file.
  ///
  /// \param out the output stream to write page table to.
  /// \return file position if success.
  ::arrow::Result<int64_t> Write(const std::shared_ptr<::arrow::io::OutputStream>& out);

 private:
  /// Map<column, Map<page, {position, length}>>
  std::map<int32_t, std::map<int32_t, PageInfo>> page_info_map_;
};

}  // namespace lance::format
