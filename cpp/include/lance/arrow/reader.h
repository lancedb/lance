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
#include <arrow/status.h>
#include <arrow/table.h>

#include <memory>
#include <string>
#include <vector>

namespace lance::arrow {

/// Lance File format reader.
///
class FileReader final {
 public:
  FileReader() = delete;

  ~FileReader();

  /// Factory method.
  ///
  static ::arrow::Result<std::unique_ptr<FileReader>> Make(
      std::shared_ptr<::arrow::io::RandomAccessFile> in,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// Get the arrow schema of the dataset.
  ::arrow::Result<std::shared_ptr<::arrow::Schema>> GetSchema();

  /// Returns the primary key of the dataset.
  [[nodiscard]] std::string primary_key() const;

  /// Number of batches.
  [[nodiscard]] int64_t num_batches() const;

  /// Return the row count, logical number of rows in the file.
  [[nodiscard]] int64_t length() const;

  /// Get a single Row at index.
  ///
  /// \param idx Row index
  /// \return a vector of each column in the row.
  ::arrow::Result<std::vector<std::shared_ptr<::arrow::Scalar>>> Get(int32_t idx);

  /// Get a single ROW with selected columns.
  ///
  /// \param idx the index of the row in the file.
  /// \param columns selected columns.
  /// \return a single row.
  ::arrow::Result<std::vector<std::shared_ptr<::arrow::Scalar>>> Get(
      int32_t idx, const std::vector<std::string>& columns);

  /// Read the entire table from the file.
  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable();

  /// Read the selected columns from the file.
  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable(
      const std::vector<std::string>& columns);

 private:
  FileReader(std::shared_ptr<::arrow::io::RandomAccessFile> in, ::arrow::MemoryPool* pool) noexcept;

  // PIMPL: https://en.cppreference.com/w/cpp/language/pimpl
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lance::arrow
