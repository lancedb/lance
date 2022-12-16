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

/// \brief extended internal interface for dataset.cc

#pragma once

#include <arrow/filesystem/api.h>

#include <memory>

#include "lance/arrow/dataset.h"
#include "lance/format/manifest.h"

namespace lance::arrow {

/// LanceDataset Implementation.
class LanceDataset::Impl {
 public:
  Impl() = delete;

  Impl(std::shared_ptr<::arrow::fs::FileSystem> filesystem,
       std::string uri,
       std::shared_ptr<lance::format::Manifest> m)
      : fs(std::move(filesystem)), base_uri(std::move(uri)), manifest(std::move(m)) {}

  /// Data directory.
  std::string data_dir() const;

  /// The URI to store versioned metadata.
  std::string versions_dir() const;

  /// Write the manifest version file.
  /// It only supports single writer at the moment.
  ///
  ///
  ::arrow::Result<std::unique_ptr<Impl>> WriteNewVersion(
      std::shared_ptr<lance::format::Manifest> new_manifest,
      const DatasetVersion& new_version) const;

  std::shared_ptr<::arrow::fs::FileSystem> fs;
  std::string base_uri;
  std::shared_ptr<lance::format::Manifest> manifest;
};

}  // namespace lance::arrow