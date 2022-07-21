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

namespace lance::format {
class Schema;
}

namespace lance::io {

class FileReader;
class Filter;

/// Projection over dataset.
///
class Project {
 public:
  Project() = delete;

  /// Make a Project from the full dataset schema and scan options.
  static ::arrow::Result<std::unique_ptr<Project>> Make(
      std::shared_ptr<format::Schema> schema,
      std::shared_ptr<::arrow::dataset::ScanOptions> scan_options);

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Execute(std::shared_ptr<FileReader> reader,
                                                                 int32_t chunk_idx);

 private:
  Project(std::shared_ptr<format::Schema> dataset_schema,
          std::shared_ptr<format::Schema> projected_schema,
          std::shared_ptr<format::Schema> scan_schema,
          std::unique_ptr<Filter> filter);

  std::shared_ptr<format::Schema> dataset_schema_;
  std::shared_ptr<format::Schema> projected_schema_;
  /// scan_schema_ equals to projected_schema_ - filters_.schema()
  /// It includes the columns that are not read from the filters yet.
  std::shared_ptr<format::Schema> scan_schema_;
  std::unique_ptr<Filter> filter_;
};

}  // namespace lance::io