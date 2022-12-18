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
#include <ctime>
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
  explicit DatasetVersion(VersionNumberType version,
                          std::chrono::time_point<std::chrono::system_clock> created);

  /// Get version number.
  VersionNumberType version() const;

  /// Timestamp of dataset creation, in UTC.
  const std::chrono::time_point<std::chrono::system_clock>& timestamp() const;

  /// time_t representation of timestamp. Used for cython
  std::time_t timet_timestamp() const;

  /// Increase version number
  DatasetVersion& operator++();

  /// Increase version number
  DatasetVersion operator++(int);

  /// Change timestamp to `Now()`.
  void Touch();

 private:
  VersionNumberType version_ = 0;

  /// Dataset creation time, in UTC timezone.
  std::chrono::time_point<std::chrono::system_clock> timestamp_;
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

  /// Load dataset from URI, with a optional version.
  ///
  /// \param uri a fully qualified dataset URI
  /// \param version optional version to load.
  /// \return See `Make(fs, base_uri, version)`.
  static ::arrow::Result<std::shared_ptr<LanceDataset>> Make(
      const std::string& uri, std::optional<uint64_t> version = std::nullopt);

  /// Create a new LanceDataset at a given version
  ::arrow::Result<std::shared_ptr<LanceDataset>> Checkout(
      std::optional<uint64_t> version = std::nullopt) const;

  /// Get all the dataset versions.
  ::arrow::Result<std::vector<DatasetVersion>> versions() const;

  /// Get the latest version of the dataset
  ::arrow::Result<DatasetVersion> latest_version() const;

  /// Returns the version of this dataset.
  DatasetVersion version() const;

  std::string type_name() const override { return "lance"; }

  /// Get dataset URI.
  const std::string& uri() const;

  /// Begin to build a column updater against to this dataset.
  ///
  /// \param new_field the new field / column to be updated.
  /// \return a builder for `Updater`.
  ::arrow::Result<std::shared_ptr<UpdaterBuilder>> NewUpdate(
      const std::shared_ptr<::arrow::Field>& new_field) const;

  ::arrow::Result<std::shared_ptr<UpdaterBuilder>> NewUpdate(
      const std::shared_ptr<::arrow::Schema>& new_columns) const;

  /// Merge an in-memory table, except the "right_on" column.
  ///
  /// The algorithm follows the semantic of `LEFT JOIN` in SQL.
  /// The difference to LEFT JOIN is that this function does not allow one row
  /// on the left ("this" dataset) maps to two distinct rows on the right ("other").
  /// However, if it can not find a matched row on the right side, a NULL value is provided.
  ///
  /// For example,
  ///
  /// \code
  /// dataset (left) = {
  ///   "id": [1, 2, 3, 4],
  ///   "vals": ["a", "b", "c", "d"],
  /// }
  /// table (right) = {
  ///   "id": [5, 1, 10, 3, 8],
  ///   "attrs": [5.0, 1.0, 10.0, 3.0, 8.0],
  /// }
  ///
  /// dataset.AddColumn(table, on="id") =>
  ///   {
  ///     "id": [1, 2, 3, 4],
  ///     "vals": ["a", "b", "c", "d"],
  ///     "attrs": [1.0, Null, 3.0, Null],
  ///   }
  /// \endcode
  ///
  /// \param right the table to merge with this dataset.
  /// \param left_on the column in this dataset be compared to.
  /// \param right_on the column in the table to be compared to.
  ///           This column must exist in both side and have the same data type.
  /// \param pool memory pool
  /// \return `::arrow::Status::OK` if success.
  ///
  ::arrow::Result<std::shared_ptr<LanceDataset>> Merge(
      const std::shared_ptr<::arrow::Table>& right,
      const std::string& left_on,
      const std::string& right_on,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// Merge an in-memory table, both sides must have the same column specified by the "on" name.
  ///
  /// See `Merge(right, left_on, right_on, pool)` for details.
  ::arrow::Result<std::shared_ptr<LanceDataset>> Merge(
      const std::shared_ptr<::arrow::Table>& right,
      const std::string& on,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

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
