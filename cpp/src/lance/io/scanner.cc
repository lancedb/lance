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

#include "lance/io/scanner.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/util/future.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <future>
#include <tuple>

#include "lance/format/metadata.h"
#include "lance/format/schema.h"
#include "lance/io/filter.h"
#include "lance/io/limit.h"
#include "lance/io/project.h"
#include "lance/io/reader.h"

namespace lance::io {

Scanner::Scanner(std::shared_ptr<FileReader> reader,
                 std::shared_ptr<arrow::dataset::ScanOptions> options,
                 std::optional<int64_t> limit,
                 int64_t offset) noexcept
    : reader_(reader),
      options_(options),
      limit_(limit),
      offset_(offset),
      max_queue_size_(static_cast<std::size_t>(options->batch_readahead)) {}

Scanner::Scanner(const Scanner& other) noexcept
    : reader_(other.reader_),
      options_(other.options_),
      limit_(other.limit_),
      offset_(other.offset_),
      schema_(other.schema_),
      project_(other.project_),
      current_chunk_(other.current_chunk_),
      max_queue_size_(other.max_queue_size_) {}

Scanner::Scanner(Scanner&& other) noexcept
    : reader_(std::move(other.reader_)),
      options_(std::move(other.options_)),
      limit_(other.limit_),
      offset_(other.offset_),
      schema_(std::move(other.schema_)),
      project_(std::move(other.project_)),
      current_chunk_(other.current_chunk_),
      max_queue_size_(other.max_queue_size_),
      q_(std::move(other.q_)) {}

::arrow::Status Scanner::Open() {
  schema_ = std::make_shared<lance::format::Schema>(reader_->schema());
  ARROW_ASSIGN_OR_RAISE(project_, Project::Make(schema_, options_, limit_, offset_));
  if (!project_->CanParallelScan()) {
    max_queue_size_ = 1;
  }
  return ::arrow::Status::OK();
}

void Scanner::AddPrefetchTask() {
  while (q_.size() < max_queue_size_ && current_chunk_ < reader_->metadata().num_chunks()) {
    auto chunk_id = current_chunk_++;
    auto f = std::async(
        [&](int32_t chunk_id) {
          auto result = project_->Execute(reader_, chunk_id);
          if (!result.ok()) {
            fmt::print(
                stderr, "Read bad chunk: chunk_id={}: {}\n", chunk_id, result.status().message());
          }
          return result;
        },
        chunk_id);
    q_.push(std::move(f));
  }
}

::arrow::Result<::std::shared_ptr<::arrow::RecordBatch>> Scanner::Next() {
  // Let's do something simple.
  // Each time just read batch_size * prefecth_size, to amortize I/O.
  AddPrefetchTask();
  if (q_.empty()) {
    return nullptr;
  }
  auto future = std::move(q_.front());
  q_.pop();
  return future.get();
}

::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> Scanner::operator()() {
  /// TODO: Make it truly async someday.
  auto f = ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(this->Next());
  return f;
}

}  // namespace lance::io