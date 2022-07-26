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

#include <memory>

namespace lance::format {

class Schema;

/// \brief Manifest.
///
/// Organize the less-frequently updated metadata about the full dataset:
///  * Primary key
///  * Schema
///
class Manifest final {
 public:
  Manifest() = default;

  /// Construct a Manifest with primary key and schema.
  ///
  /// \param primary_key the primary key of the dataset.
  /// \param schema the dataset schema.
  Manifest(const std::string& primary_key, std::shared_ptr<Schema> schema);

  /// Move constructor.
  Manifest(Manifest&& other) noexcept;

  ~Manifest() = default;

  /// Parse a Manifest from input file at the offset.
  static ::arrow::Result<std::shared_ptr<Manifest>> Parse(
      std::shared_ptr<::arrow::io::RandomAccessFile> in, int64_t offset);

  /// Write the Manifest to a file.
  ///
  /// \param out the output stream to write this Manifest to.
  /// \return The offset of the manifest.
  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::io::OutputStream> out) const;

  /// Get the primary key of this dataset.
  const std::string& primary_key() const;

  /// Get schema of the dataset.
  const Schema& schema() const;

 private:
  /// Primary key of the datasets.
  std::string primary_key_;

  /// Table schema.
  std::shared_ptr<Schema> schema_;
};

}  // namespace lance::format
