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

#include "lance/format/data_fragment.h"
#include "lance/format/format.pb.h"

namespace lance::format {

class Schema;

/// \brief Manifest.
///
///  * Schema
///  * Version
///  * Fragments.
///
class Manifest final {
 public:
  Manifest() = default;

  /// Construct a Manifest with the schema.
  ///
  /// \param schema the dataset schema.
  Manifest(std::shared_ptr<Schema> schema);

  /// Move constructor.
  Manifest(Manifest&& other) noexcept;

  /// Copy constructor.
  Manifest(const Manifest& other) noexcept;

  ~Manifest() = default;

  /// Parse a Manifest from input file at the offset.
  static ::arrow::Result<std::shared_ptr<Manifest>> Parse(
      std::shared_ptr<::arrow::io::RandomAccessFile> in, int64_t offset);

  /// Write the Manifest to a file.
  ///
  /// \param out the output stream to write this Manifest to.
  /// \return The offset of the manifest.
  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::io::OutputStream> out) const;

  /// Increase the version number and returns the new Manifest.
  ///
  std::shared_ptr<Manifest> BumpVersion(bool overwrite = false);

  /// Get schema of the dataset.
  const Schema& schema() const;

  /// Returns the version number.
  uint64_t version() const;

  /// Get the fragments existed in this version of dataset.
  const std::vector<std::shared_ptr<DataFragment>>& fragments() const;

  /// Append more fragments to the dataset.
  void AppendFragments(const std::vector<std::shared_ptr<DataFragment>>& fragments);

 private:
  /// Table schema.
  std::shared_ptr<Schema> schema_;

  std::uint64_t version_ = 1;

  std::vector<std::shared_ptr<DataFragment>> fragments_;

  Manifest(const lance::format::pb::Manifest& pb);
};

}  // namespace lance::format
