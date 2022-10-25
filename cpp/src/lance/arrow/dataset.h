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
#include <string>
#include <vector>

namespace lance::arrow {

class DatasetVersion {
 public:
  /// Current version ID
  uint64_t version() const;

 private:
  uint64_t version_;
};

/// Lance Dataset, supports versioning and schema evolution.
///
class LanceDataset : public ::arrow::dataset::Dataset {
 public:
  /// Copy constructor.
  LanceDataset(const LanceDataset& other);

  ~LanceDataset();

  /// Write a in-memory Arrow dataset to disk.
  ///
  /// If the dataset already exist, write new version.
  static ::arrow::Status Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                               std::shared_ptr<::arrow::dataset::Scanner> scanner);

  /// Load dataset.
  ///
  /// \param fs File system object
  /// \param base_uri base path to the dataset.
  /// Returns nullptr if the dataset does not exist.
  static ::arrow::Result<std::shared_ptr<LanceDataset>> Make(
      std::shared_ptr<::arrow::fs::FileSystem> fs,
      std::string base_uri,
      std::optional<uint64_t> version = std::nullopt);

  std::string type_name() const override { return "lance"; }

  ::arrow::Result<std::shared_ptr<Dataset>> ReplaceSchema(
      std::shared_ptr<::arrow::Schema> schema) const override;

  /// Get all sorted versions.
  std::vector<DatasetVersion> versions() const;

  /// Access to the latest version of dataset
  DatasetVersion latest_version() const;

  ::arrow::Result<std::shared_ptr<LanceDataset>> version(uint64_t version) const;

 protected:
  ::arrow::Result<::arrow::dataset::FragmentIterator> GetFragmentsImpl(
      ::arrow::compute::Expression predicate) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  LanceDataset(std::unique_ptr<Impl> impl);
};

}  // namespace lance::arrow
