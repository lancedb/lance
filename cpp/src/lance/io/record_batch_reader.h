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
#include <arrow/util/thread_pool.h>

#include <atomic>
#include <future>
#include <memory>
#include <optional>
#include <queue>

// Forward declarations.
namespace arrow::dataset {
class ScanOptions;
}

namespace lance::arrow {
class LanceFragment;
}

namespace lance::format {
class Schema;
}

namespace lance::io {

class FileReader;

namespace exec {
class Project;
}

/// Lance RecordBatchReader
class RecordBatchReader : ::arrow::RecordBatchReader {
 public:
  /// Factory method.
  static ::arrow::Result<RecordBatchReader> Make(
      const lance::arrow::LanceFragment& fragment,
      std::shared_ptr<::arrow::dataset::ScanOptions> options,
      ::arrow::internal::Executor* executor = ::arrow::internal::GetCpuThreadPool()) noexcept;

  RecordBatchReader() = delete;

  /// Copy constructor.
  RecordBatchReader(const RecordBatchReader& other) noexcept;

  /// Move constructor.
  RecordBatchReader(RecordBatchReader&& other) noexcept;

  ~RecordBatchReader() override = default;

  std::shared_ptr<::arrow::Schema> schema() const override;

  ::arrow::Status ReadNext(std::shared_ptr<::arrow::RecordBatch>* batch) override;

  /// Async read, to match LanceFileFormat::ScanBatchesAsync()
  ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> operator()();

 private:
  RecordBatchReader(std::shared_ptr<exec::Project> project, ::arrow::internal::Executor* executor);

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> ReadBatch() const;

  /// Projection over the dataset.
  std::shared_ptr<exec::Project> project_;
  ::arrow::internal::Executor* executor_;
};

}  // namespace lance::io
