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
#include "lance/format/schema.h"
#include "lance/io/filter.h"
#include "lance/io/limit.h"
#include "lance/io/project.h"
#include "lance/io/reader.h"

namespace lance::io {

RecordBatchReader::RecordBatchReader(std::shared_ptr<FileReader> reader,
                                     std::shared_ptr<arrow::dataset::ScanOptions> options,
                                     ::arrow::internal::ThreadPool* thread_pool,
                                     std::optional<int64_t> limit,
                                     int64_t offset) noexcept
    : reader_(reader),
      options_(options),
      limit_(limit),
      offset_(offset),
      thread_pool_(thread_pool) {
  assert(thread_pool_);
}

RecordBatchReader::RecordBatchReader(const RecordBatchReader& other) noexcept
    : reader_(other.reader_),
      options_(other.options_),
      limit_(other.limit_),
      offset_(other.offset_),
      schema_(other.schema_),
      project_(other.project_),
      thread_pool_(other.thread_pool_),
      current_batch_(int(other.current_batch_)) {}

RecordBatchReader::RecordBatchReader(RecordBatchReader&& other) noexcept
    : reader_(std::move(other.reader_)),
      options_(std::move(other.options_)),
      limit_(other.limit_),
      offset_(other.offset_),
      schema_(std::move(other.schema_)),
      project_(std::move(other.project_)),
      thread_pool_(std::move(other.thread_pool_)),
      current_batch_(int(other.current_batch_)) {}

::arrow::Status RecordBatchReader::Open() {
  schema_ = std::make_shared<lance::format::Schema>(reader_->schema());
  ARROW_ASSIGN_OR_RAISE(project_, Project::Make(schema_, options_, limit_, offset_));
  return ::arrow::Status::OK();
}

std::shared_ptr<::arrow::Schema> RecordBatchReader::schema() const {
  return project_->schema()->ToArrow();
}

::arrow::Status RecordBatchReader::ReadNext(std::shared_ptr<::arrow::RecordBatch>* batch) {
  int32_t batch_id = current_batch_++;
  ARROW_ASSIGN_OR_RAISE(auto batch_read, ReadBatch(batch_id));
  if (batch_read) {
    *batch = std::move(batch_read);
  }
  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::ReadBatch(
    int32_t batch_id) const {
  if (batch_id < reader_->metadata().num_batches()) {
    return project_->Execute(reader_, batch_id);
  }
  return nullptr;
}

::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::operator()() {
  int32_t batch_id = current_batch_++;
  auto result = thread_pool_->Submit(
      [&](int32_t batch_id) {
        /// TODO: how to handle error in thread pool?
        auto result = this->ReadBatch(batch_id);
        return result.ValueOrDie();
      },
      batch_id);
  if (result.ok()) {
    return result.ValueOrDie();
  }
  return result.status();
}

}  // namespace lance::io