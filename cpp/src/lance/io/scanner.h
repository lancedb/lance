// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <arrow/type_fwd.h>

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

/// Lance Scanner
class Scanner {
 public:
  /// Constructor.
  Scanner(std::shared_ptr<FileReader> reader,
          std::shared_ptr<::arrow::dataset::ScanOptions> options);

  /// Move constructor.
  Scanner(Scanner&& other) noexcept;

  /// Copy constructor.
  Scanner(const Scanner&);

  ~Scanner() = default;

  /// Open the Scanner. Must call it before start iterating.
  ::arrow::Status Open();

  /// Returns the next record batch if any.
  ///
  /// \return A record batch. Returns `nullptr` if reaches the end.
  ::arrow::Result<::std::shared_ptr<::arrow::RecordBatch>> Next();

  /// Async read, to match LanceFileFormat::ScanBatchesAsync()
  ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> operator()();

 private:
  Scanner() = delete;

  std::shared_ptr<FileReader> reader_;
  std::shared_ptr<::arrow::dataset::ScanOptions> options_;
  std::shared_ptr<lance::format::Schema> schema_;
  /// Projection over the dataset.
  std::unique_ptr<Project> project_;

  int current_offset_ = 0;
  int prefetch_offset_ = 0;
  std::queue<
      std::future<std::tuple<::arrow::Result<std::shared_ptr<::arrow::RecordBatch>>, int32_t>>>
      q_;

  void AddPrefetchTask();
};

}  // namespace lance::io
