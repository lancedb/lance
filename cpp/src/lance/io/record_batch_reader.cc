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

#include "lance/io/record_batch_reader.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/thread_pool.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <tuple>

#include "lance/format/metadata.h"
#include "lance/io/exec/limit.h"
#include "lance/io/exec/project.h"
#include "lance/io/reader.h"

namespace lance::io {

::arrow::Result<RecordBatchReader> RecordBatchReader::Make(
    const lance::arrow::LanceFragment& fragment,
    std::shared_ptr<::arrow::dataset::ScanOptions> options,
    ::arrow::internal::Executor* executor) noexcept {
  ARROW_ASSIGN_OR_RAISE(auto project, exec::Project::Make(fragment, std::move(options)));
  return RecordBatchReader(std::move(project), executor);
};

RecordBatchReader::RecordBatchReader(std::shared_ptr<exec::Project> project,
                                     ::arrow::internal::Executor* executor)
    : project_(std::move(project)), executor_(executor) {}

RecordBatchReader::RecordBatchReader(const RecordBatchReader& other) noexcept
    : project_(other.project_), executor_(other.executor_) {}

RecordBatchReader::RecordBatchReader(RecordBatchReader&& other) noexcept
    : project_(std::move(other.project_)), executor_(std::move(other.executor_)) {}

std::shared_ptr<::arrow::Schema> RecordBatchReader::schema() const {
  return project_->schema()->ToArrow();
}

::arrow::Status RecordBatchReader::ReadNext(std::shared_ptr<::arrow::RecordBatch>* batch) {
  ARROW_ASSIGN_OR_RAISE(auto scan_batch, project_->Next());
  *batch = std::move(scan_batch.batch);
  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::ReadBatch() const {
  ARROW_ASSIGN_OR_RAISE(auto batch, project_->Next());
  return batch.batch;
}

::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::operator()() {
  ARROW_ASSIGN_OR_RAISE(auto fut, executor_->Submit([&]() {
    return ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(this->ReadBatch());
  }));
  return fut;
}

}  // namespace lance::io