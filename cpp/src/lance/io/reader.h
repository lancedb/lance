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
#include <tuple>

namespace lance::format {
class Field;
class Manifest;
class Metadata;
class PageTable;
class Schema;
}  // namespace lance::format

namespace lance::io {

/// FileReader implementation.
class FileReader {
 public:
  explicit FileReader(std::shared_ptr<::arrow::io::RandomAccessFile> in,
                      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) noexcept;

  /// Opens the FileReader.
  ::arrow::Status Open();

  /// Get the reference to the lance schema.
  const lance::format::Schema& schema() const;

  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable();

  ::arrow::Result<std::shared_ptr<::arrow::Table>> ReadTable(
      const std::vector<std::string>& columns);

  /// Read a RecordBatch at the file offset.
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadAt(const lance::format::Schema& schema,
                                                                int32_t offset,
                                                                int32_t length) const;

  /// Read a Batch
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadBatch(
      const lance::format::Schema& schema,
      int32_t batch_id,
      int32_t offset = 0,
      std::optional<int32_t> length = std::nullopt) const;

  /// Read a Batch with indices.
  ///
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadBatch(
      const lance::format::Schema& schema,
      int32_t batch_id,
      std::shared_ptr<::arrow::Int32Array> indices) const;

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

  /// Array Read Parameters.
  ///  - ReadAt offset + length.
  ///  - Take elements by indices.
  struct ArrayReadParams {
    explicit ArrayReadParams(int32_t offset, std::optional<int32_t> length = std::nullopt);

    explicit ArrayReadParams(std::shared_ptr<::arrow::Int32Array> indices);

    std::optional<int32_t> offset = std::nullopt;
    std::optional<int32_t> length = std::nullopt;
    std::optional<std::shared_ptr<::arrow::Int32Array>> indices = std::nullopt;
  };

  /// Read a batch using ArrayReadParams.
  ///
  /// \param schema the schema to read.
  /// \param batch_id the id of the batch to read
  /// \param params read params.
  /// \return a RecordBatch if success.
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadBatch(
      const lance::format::Schema& schema, int32_t batch_id, const ArrayReadParams& params) const;

  /// Get an ARRAY from column / file from a given Batch.
  ///
  /// \param field the field (column) specification
  /// \param batch_id the index of the batch in the file.
  /// \param params Read parameters
  ///
  /// \return An array if success.
  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetArray(
      const std::shared_ptr<lance::format::Field>& field,
      int32_t batch_id,
      const ArrayReadParams& params) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetPrimitiveArray(
      const std::shared_ptr<lance::format::Field>& field,
      int32_t batch_id,
      const ArrayReadParams& params) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetStructArray(
      const std::shared_ptr<lance::format::Field>& field,
      int32_t batch_id,
      const ArrayReadParams& params) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetListArray(
      const std::shared_ptr<lance::format::Field>& field,
      int32_t batch_id,
      const ArrayReadParams& params) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> GetDictionaryArray(
      const std::shared_ptr<lance::format::Field>& field,
      int32_t batch_id,
      const ArrayReadParams& params) const;

  ::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> Get(
      int32_t idx, const lance::format::Schema& schema);
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const;
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetPrimitiveScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const;
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetListScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const;
  ::arrow::Result<::std::shared_ptr<::arrow::Scalar>> GetStructScalar(
      const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const;

  /// Get the file position and page length for a page.
  ///
  /// \param field_id the field / column Id
  /// \param batch_id the index of a batch.
  /// \return a tuple of `[position, length]` for a page. Returns `Status::Invalid` otherwise.
  ::arrow::Result<std::tuple<int64_t, int64_t>> GetPageInfo(int32_t field_id,
                                                            int32_t batch_id) const;

 private:
  std::shared_ptr<::arrow::io::RandomAccessFile> file_;
  ::arrow::MemoryPool* pool_;
  std::shared_ptr<lance::format::Metadata> metadata_;
  std::shared_ptr<lance::format::Manifest> manifest_;
  std::shared_ptr<lance::format::PageTable> page_table_;

  std::shared_ptr<::arrow::Buffer> cached_last_page_;
};

}  // namespace lance::io
