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

#include <arrow/io/type_fwd.h>
#include <arrow/type_fwd.h>

#include <atomic>
#include <memory>
#include <optional>

namespace lance::format {
class Field;
class LookupTable;
class Manifest;
class Metadata;
class Schema;
}  // namespace lance::format

namespace lance::io {

/// FileReader implementation.
class FileReader {
 public:
  FileReader(std::shared_ptr<::arrow::io::RandomAccessFile> in,
             ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) noexcept;

  ::arrow::Status Open();

  /// Get the reference to the lance schema.
  const lance::format::Schema& schema() const;

  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable();

  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable(
      const std::vector<std::string>& columns);

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadBatch(
      int32_t offset, int32_t length, const std::vector<std::string>& columns) const;

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadBatch(
      int32_t offset, int32_t length, const lance::format::Schema& schema) const;

  /// Get file metadata.
  const lance::format::Metadata& metadata() const;

  /// Get file manifest.
  const lance::format::Manifest& manifest() const;

  /// Read one single row at the index.
  ::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> Get(int32_t idx);

  /// Read one single row at the index, only for specified columns.
  ::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> Get(
      int32_t idx, const std::vector<std::string>& columns);

 private:
  FileReader() = delete;

  /// Read the table with the given schema.
  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable(
      const lance::format::Schema& schema) const;

  /// Get an ARRAY from column / file at chunk.
  ///
  /// \param field the field (column) specification
  /// \param chunk_id the index of the chunk in the file.
  /// \param start start position
  /// \param length the length of the array to fetch.
  ///
  /// \return An array if success.
  /// TODO: use std::optional for length
  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetArray(
      const std::shared_ptr<lance::format::Field>& field,
      int chunk_id,
      int32_t start = 0,
      std::optional<int32_t> length = std::nullopt) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetPrimitiveArray(
      const std::shared_ptr<lance::format::Field>& field,
      int chunk_id,
      int32_t start = 0,
      std::optional<int32_t> length = std::nullopt) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetStructArray(
      const std::shared_ptr<lance::format::Field>& field,
      int chunk_id,
      int32_t start = 0,
      std::optional<int32_t> length = std::nullopt) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetListArray(
      const std::shared_ptr<lance::format::Field>& field,
      int chunk_id,
      int32_t start = 0,
      std::optional<int32_t> length = std::nullopt) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetDictionaryArray(
      const std::shared_ptr<lance::format::Field>& field,
      int chunk_id,
      int32_t start = 0,
      std::optional<int32_t> length = std::nullopt) const;

  ::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> Get(
      int32_t idx, const lance::format::Schema& schema);
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const;
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetPrimitiveScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const;
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetListScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const;
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetStructScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const;

  /// Get the file offset of a chunk for a column.
  ///
  /// \param field_id the field / column Id
  /// \param chunk_id the chunk index in the file
  /// \return the offset where the chunk starts. Returns Status::Invalid otherwise.
  ::arrow::Result<int64_t> GetChunkOffset(int64_t field_id, int64_t chunk_id) const;

 private:
  std::shared_ptr<::arrow::io::RandomAccessFile> file_;
  ::arrow::MemoryPool* pool_;
  std::shared_ptr<lance::format::Metadata> metadata_;
  std::shared_ptr<lance::format::Manifest> manifest_;
  std::shared_ptr<lance::format::LookupTable> lookup_table_;

  std::shared_ptr<::arrow::Buffer> cached_last_page_;
};

}  // namespace lance::io
