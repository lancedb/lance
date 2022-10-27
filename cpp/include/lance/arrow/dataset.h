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

#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lance::arrow {


/// Lance Dataset, supports versioning and schema evolution.
///
class LanceDataset : public ::arrow::dataset::Dataset {
 public:
  /// Dataset Write Mode
  enum WriteMode {
    /// Create a new dataset. Expect the dataset does not exist.
    kCreate,
    /// Append to an existing dataset.
    kAppend,
    /// Overwrite a dataset as a new version, or create new dataset if not exist.
    kOverwrite,
  };

  /// Copy constructor.
  LanceDataset(const LanceDataset& other);

  ~LanceDataset() override;

  /// Write a in-memory Arrow dataset to disk.
  ///
  /// \param options Arrow Dataset Write Options.
  /// \param scanner the source dataset to be written.
  /// \param mode the mode to write the data. Default is `WriteMode::kCreate`.
  ///
  static ::arrow::Status Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                               std::shared_ptr<::arrow::dataset::Scanner> scanner,
                               WriteMode mode = kCreate);

  /// Load dataset, with a specific version.
  ///
  /// \param fs File system object
  /// \param base_uri base path to the dataset.
  /// \param version optional version to load. If not presented, load the latest version.
  /// \return A specific version of the dataset. Or return nullptr if the dataset does not exist.
  static ::arrow::Result<std::shared_ptr<LanceDataset>> Make(
      std::shared_ptr<::arrow::fs::FileSystem> fs,
      std::string base_uri,
      std::optional<uint64_t> version = std::nullopt);

  std::string type_name() const override { return "lance"; }

  ::arrow::Result<std::shared_ptr<Dataset>> ReplaceSchema(
      std::shared_ptr<::arrow::Schema> schema) const override;

 protected:
  ::arrow::Result<::arrow::dataset::FragmentIterator> GetFragmentsImpl(
      ::arrow::compute::Expression predicate) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  explicit LanceDataset(std::unique_ptr<Impl> impl);
};

}  // namespace lance::arrow
