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
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lance::arrow {

// Forward Declaration
class UpdaterBuilder;

/// Dataset Version
///
class DatasetVersion {
 public:
  using VersionNumberType = uint64_t;

  DatasetVersion() = default;

  /// Construct a Dataset version from version number.
  explicit DatasetVersion(VersionNumberType version);

  /// Get version number.
  VersionNumberType version() const;

  /// Increase version number
  DatasetVersion& operator++();

  /// Increase version number
  DatasetVersion operator++(int);

 private:
  VersionNumberType version_ = 0;

  /// Dataset creation time, in UTC timezone.
  std::chrono::time_point<std::chrono::system_clock> created_time_;
};

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

  /// Write Arrow dataset to disk.
  ///
  /// \param options Arrow Dataset Write Options.
  /// \param scanner the source dataset to be written.
  /// \param mode the mode to write the data. Default is `WriteMode::kCreate`.
  ///
  static ::arrow::Status Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                               std::shared_ptr<::arrow::dataset::Scanner> scanner,
                               WriteMode mode = kCreate);

  /// Write Arrow dataset to disk.
  ///
  /// \param options Arrow Dataset Write Options.
  /// \param scanner the source dataset to be written.
  /// \param mode the mode to write the data. Default is `WriteMode::kCreate`.
  ///
  /// GH-62. To accommodate Cython lacking of arrow Scanner interface, we directly write
  /// `::arrow::dataset::Dataset`, which is public interface in PyArrow.
  static ::arrow::Status Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                               const std::shared_ptr<::arrow::dataset::Dataset>& dataset,
                               WriteMode mode = kCreate);

  /// Load dataset, with a specific version.
  ///
  /// \param fs File system object
  /// \param base_uri base path to the dataset.
  /// \param version optional version to load. If not presented, load the latest version.
  /// \return A specific version of the dataset. Or return nullptr if the dataset does not exist.
  static ::arrow::Result<std::shared_ptr<LanceDataset>> Make(
      const std::shared_ptr<::arrow::fs::FileSystem>& fs,
      const std::string& base_uri,
      std::optional<uint64_t> version = std::nullopt);

  /// Get all the dataset versions.
  ::arrow::Result<std::vector<DatasetVersion>> versions() const;

  /// Get the latest version of the dataset
  ::arrow::Result<DatasetVersion> latest_version() const;

  /// Returns the version of this dataset.
  DatasetVersion version() const;

  std::string type_name() const override { return "lance"; }

  /// Begin to build a column updater against to this dataset.
  ///
  /// \param new_field the new field / column to be updated.
  /// \return a builder for `Updater`.
  ::arrow::Result<std::shared_ptr<UpdaterBuilder>> NewUpdate(
      const std::shared_ptr<::arrow::Field>& new_field) const;

  ::arrow::Result<std::shared_ptr<::arrow::dataset::Dataset>> ReplaceSchema(
      std::shared_ptr<::arrow::Schema> schema) const override;

  /// Add column via a compute expression.
  ///
  /// \param field the new field.
  /// \param expression the expression to compute the column.
  /// \return a new version of the dataset.
  ///
  /// See `Updater` for details.
  ::arrow::Result<std::shared_ptr<LanceDataset>> AddColumn(
      const std::shared_ptr<::arrow::Field>& field, ::arrow::compute::Expression expression);

 protected:
  ::arrow::Result<::arrow::dataset::FragmentIterator> GetFragmentsImpl(
      ::arrow::compute::Expression predicate) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  explicit LanceDataset(std::unique_ptr<Impl> impl);

  friend class Updater;
};

}  // namespace lance::arrow
