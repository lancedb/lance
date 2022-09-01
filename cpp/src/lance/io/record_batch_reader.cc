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
                                     std::shared_ptr<::arrow::dataset::ScanOptions> options,
                                     std::optional<int64_t> limit,
                                     int64_t offset,
                                     ::arrow::internal::ThreadPool* thread_pool) noexcept
    : reader_(reader),
      options_(options),
      num_batches_(reader->metadata().num_batches()),
      limit_(limit),
      offset_(offset),
      thread_pool_(thread_pool) {
  assert(thread_pool_);
  current_batch_length_ = reader->metadata().GetBatchLength(0);
}

RecordBatchReader::RecordBatchReader(const RecordBatchReader& other) noexcept
    : reader_(other.reader_),
      options_(other.options_),
      num_batches_(other.num_batches_),
      limit_(other.limit_),
      offset_(other.offset_),
      project_(other.project_),
      thread_pool_(other.thread_pool_),
      current_batch_(int(other.current_batch_)),
      current_batch_length_(other.current_batch_length_),
      current_offset_(other.current_offset_) {}

RecordBatchReader::RecordBatchReader(RecordBatchReader&& other) noexcept
    : reader_(std::move(other.reader_)),
      options_(std::move(other.options_)),
      num_batches_(other.num_batches_),
      limit_(other.limit_),
      offset_(other.offset_),
      project_(std::move(other.project_)),
      thread_pool_(std::move(other.thread_pool_)),
      current_batch_(int(other.current_batch_)),
      current_batch_length_(other.current_batch_length_),
      current_offset_(other.current_offset_) {}

::arrow::Status RecordBatchReader::Open() {
  ARROW_ASSIGN_OR_RAISE(project_, Project::Make(reader_, options_, limit_, offset_));
  return ::arrow::Status::OK();
}

std::shared_ptr<::arrow::Schema> RecordBatchReader::schema() const {
  return project_->schema()->ToArrow();
}

std::optional<RecordBatchReader::Task> RecordBatchReader::NextTask() {
  std::lock_guard guard(lock_);
  if (current_batch_ >= num_batches_) {
    return std::nullopt;
  }

  auto task = Task();
  task.batch_id = current_batch_;
  task.length = options_->batch_size;
  task.offset = current_offset_;

  current_offset_ += task.length;
  if (current_offset_ >= current_batch_length_) {
    current_batch_++;
    current_offset_ = 0;
    if (current_batch_ < num_batches_) {
      current_batch_length_ = reader_->metadata().GetBatchLength(current_batch_);
    }
  }
  return task;
}

::arrow::Status RecordBatchReader::ReadNext(std::shared_ptr<::arrow::RecordBatch>* batch) {
  auto task = NextTask();
  if (task) {
    ARROW_ASSIGN_OR_RAISE(auto batch_read, ReadBatch(task->batch_id, task->offset, task->length));
    if (batch_read) {
      *batch = std::move(batch_read);
    }
  }
  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::ReadBatch(
    int32_t batch_id, int32_t offset, std::optional<int32_t> length) const {
  if (batch_id < reader_->metadata().num_batches()) {
    return project_->Execute(reader_, batch_id, offset, length);
  }
  return nullptr;
}

::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> RecordBatchReader::operator()() {
  int total_batches = reader_->metadata().num_batches();
  fmt::print("Operator(): batch_size={} total_batches={}\n", options_->batch_size, total_batches);
  ::arrow::Result<::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>> submit_result;

  auto result = thread_pool_->Submit([&]() {
    auto task = NextTask();
    if (task.has_value()) {
      return ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(
          this->ReadBatch(task->batch_id, task->offset, task->length));
    } else {
      return ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(nullptr);
    }
  });
  if (result.ok()) {
    return result.ValueOrDie();
  }
  return ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(result.status());
}

}  // namespace lance::io