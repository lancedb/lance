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
#include "lance/io/exec/filter.h"
#include "lance/io/exec/limit.h"
#include "lance/io/exec/project.h"
#include "lance/io/reader.h"

namespace lance::io {

RecordBatchReader::RecordBatchReader(std::shared_ptr<FileReader> reader,
                                     std::shared_ptr<::arrow::dataset::ScanOptions> options,
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
      project_(other.project_),
      thread_pool_(other.thread_pool_),
      current_batch_(int(other.current_batch_)) {}

RecordBatchReader::RecordBatchReader(RecordBatchReader&& other) noexcept
    : reader_(std::move(other.reader_)),
      options_(std::move(other.options_)),
      limit_(other.limit_),
      offset_(other.offset_),
      project_(std::move(other.project_)),
      thread_pool_(std::move(other.thread_pool_)),
      current_batch_(int(other.current_batch_)),
      readahead_queue_(std::move(other.readahead_queue_)) {}

::arrow::Status RecordBatchReader::Open() {
  ARROW_ASSIGN_OR_RAISE(project_, exec::Project::Make(reader_, options_, limit_, offset_));
  return ::arrow::Status::OK();
}

std::shared_ptr<::arrow::Schema> RecordBatchReader::schema() const {
  return options_->projected_schema;
}

::arrow::Status RecordBatchReader::ReadNext(std::shared_ptr<::arrow::RecordBatch>* batch) {
  ARROW_ASSIGN_OR_RAISE(auto scan_batch, project_->Next());
  *batch = std::move(scan_batch.batch);
  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::ReadBatch(
    [[maybe_unused]] int32_t batch_id) const {
  ARROW_ASSIGN_OR_RAISE(auto batch, project_->Next());
  return batch.batch;
}

::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::operator()() {
  int total_batches = reader_->metadata().num_batches();
  while (static_cast<int32_t>(readahead_queue_.size()) < options_->batch_readahead &&
         current_batch_ < total_batches) {
    int32_t batch_id = current_batch_++;
    auto result = thread_pool_->Submit(
        [&](int32_t batch_id) {
          return ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(
              this->ReadBatch(batch_id));
        },
        batch_id);
    if (!result.ok()) {
      return result.status();
    }
    readahead_queue_.emplace(std::move(result.ValueOrDie()));
  }
  if (readahead_queue_.empty()) {
    return std::shared_ptr<::arrow::RecordBatch>(nullptr);
  }
  auto first = readahead_queue_.front();
  readahead_queue_.pop();
  return first;
}

}  // namespace lance::io