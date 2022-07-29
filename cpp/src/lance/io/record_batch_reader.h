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

#include <arrow/record_batch.h>
#include <arrow/type_fwd.h>

#include <atomic>
#include <future>
#include <memory>
#include <optional>
#include <queue>

// Forward declarations.
namespace arrow::dataset {
class ScanOptions;
}
namespace lance::format {
class Schema;
}

namespace lance::io {

class FileReader;
class Project;

/// Lance RecordBatchReader
class RecordBatchReader : ::arrow::RecordBatchReader {
 public:
  /// Constructor.
  RecordBatchReader(std::shared_ptr<FileReader> reader,
                    std::shared_ptr<::arrow::dataset::ScanOptions> options,
                    std::optional<int64_t> limit = std::nullopt,
                    int64_t offset = 0) noexcept;

  /// Copy constructor.
  RecordBatchReader(const RecordBatchReader& other) noexcept;

  /// Move constructor.
  RecordBatchReader(RecordBatchReader&& other) noexcept;

  ~RecordBatchReader() = default;

  /// Open the RecordBatchReader. Must call it before start iterating.
  ::arrow::Status Open();

  std::shared_ptr<::arrow::Schema> schema() const override;

  ::arrow::Status ReadNext(std::shared_ptr<::arrow::RecordBatch>* batch) override;

  /// Async read, to match LanceFileFormat::ScanBatchesAsync()
  ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> operator()();

 private:
  RecordBatchReader() = delete;

  std::shared_ptr<FileReader> reader_;
  std::shared_ptr<::arrow::dataset::ScanOptions> options_;
  std::optional<int64_t> limit_ = std::nullopt;
  int64_t offset_ = 0;
  std::shared_ptr<lance::format::Schema> schema_;
  /// Projection over the dataset.
  std::shared_ptr<Project> project_;

  std::atomic_int32_t current_batch_ = 0;
};

}  // namespace lance::io
