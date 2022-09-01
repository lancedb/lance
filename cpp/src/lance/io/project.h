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

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>
#include <optional>

namespace lance::format {
class Schema;
}

namespace lance::io {

class FileReader;
class Filter;
class Limit;

/// \brief Projection over dataset.
///
class Project {
 public:
  Project() = delete;

  /// Make a Project from the full dataset schema and scan options.
  ///
  /// \param reader lane=ce file reader.
  /// \param scan_options Arrow scan options.
  /// \return Project if success. Returns the error status otherwise.
  ///
  static ::arrow::Result<std::unique_ptr<Project>> Make(
      const std::shared_ptr<FileReader>& reader,
      std::shared_ptr<::arrow::dataset::ScanOptions> scan_options);

  /// \brief Apply Projection over a batch.
  ///
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Execute(
      const std::shared_ptr<FileReader>& reader,
      int32_t batch_id,
      int32_t offset = 0,
      std::optional<int32_t> length = std::nullopt);

  /// Project schema
  const std::shared_ptr<format::Schema>& schema() const;

 private:
  Project(const std::shared_ptr<FileReader>& reader,
          const std::shared_ptr<::arrow::dataset::ScanOptions>& scan_options,
          std::shared_ptr<format::Schema> projected_schema,
          std::shared_ptr<format::Schema> scan_schema,
          std::unique_ptr<Filter> filter);

  /// Scan options batch size.
  int64_t batch_size() const;

  /// File reader
  std::shared_ptr<FileReader> reader_;
  std::shared_ptr<::arrow::dataset::ScanOptions> scan_options_;

  std::shared_ptr<format::Schema> projected_schema_;
  /// scan_schema_ equals to projected_schema_ - filters_.schema()
  /// It includes the columns that are not read from the filters yet.
  std::shared_ptr<format::Schema> scan_schema_;
  std::unique_ptr<Filter> filter_;

  std::unique_ptr<Limit> limit_;

  std::shared_ptr<::arrow::dataset::ScanOptions> options_;
};

}  // namespace lance::io