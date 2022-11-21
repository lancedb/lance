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

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>
#include <string>
#include <vector>

#include "lance/arrow/dataset.h"

namespace lance::arrow {

/// \brief Streaming Record Updater.
///
/// \warning This API is experimental.
///
/// It scans over the dataset, and updates a (new) column.
///
/// This class is not thread-safe.
///
///  ```cpp
///  // Example usage
///  auto builder = dataset->NewUpdate(arrow::Field("new_col", data_type)).ValueOrDie();
///  auto updater = builder->Finish().ValueOrDie();
///  while (true) {
///    auto batch = updater->Next().ValueOrDie();
///    if (!batch) {
///       // EOF break
///    }
///    auto arr = MyUpdateFunction(batch);
///    assert(updater->Update(arr).ok());
///  }
///  ```
class Updater {
 public:
  /// Destructor.
  ///
  /// Explicitly defined to make PIMPL work.
  ~Updater();

  /// Return the next batch as inputs. Or return `nullptr` for the end of dataset.
  ///
  /// The user must consume the returned results, by calling `Update`, before calling `Next()`
  /// again.
  ///
  /// \return `::arrow::RecordBatch` if success.
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Next();

  /// Update the values to new values, presented in the array.
  /// The array must has the same length as the batch returned previously via `Next()`.
  ::arrow::Status UpdateBatch(const std::shared_ptr<::arrow::Array>& arr);

  /// Finish the update and returns a new version of dataset.
  ::arrow::Result<std::shared_ptr<LanceDataset>> Finish();

 private:
  /// Make a new Updater
  ///
  /// \param dataset The dataset to be updated.
  /// \param field the (new) column to update.
  /// \param projection_columns the columns to read from source dataset.
  ///
  /// \return an Updater if success.
  static ::arrow::Result<std::shared_ptr<Updater>> Make(
      std::shared_ptr<LanceDataset> dataset,
      const std::shared_ptr<::arrow::Field>& field,
      const std::vector<std::string>& projection_columns);

  /// PIMPL
  class Impl;
  std::unique_ptr<Impl> impl_;

  /// Constructor
  explicit Updater(std::unique_ptr<Impl> impl);

  friend class UpdaterBuilder;
};

/// Updater Builder
///
/// This is similar to `::arrow::dataset::ScanBuilder`, use builder to pass
/// parameters to build a Updater.
class UpdaterBuilder {
 public:
  UpdaterBuilder(std::shared_ptr<LanceDataset> dataset, std::shared_ptr<::arrow::Field> field);

  /// Set the projection columns from the source dataset.
  void Project(std::vector<std::string> columns);

  ::arrow::Result<std::shared_ptr<Updater>> Finish();

 private:
  std::shared_ptr<LanceDataset> dataset_;

  std::shared_ptr<::arrow::Field> field_;

  std::vector<std::string> projection_columns_;
};

}  // namespace lance::arrow