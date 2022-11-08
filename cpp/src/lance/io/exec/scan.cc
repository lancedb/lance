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

#include "lance/io/exec/scan.h"

#include <memory>

#include "lance/arrow/utils.h"
#include "lance/format/metadata.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

constexpr int kMinimalIOThreads = 4;

::arrow::Result<std::unique_ptr<Scan>> Scan::Make(const std::vector<FileReaderWithSchema>& readers,
                                                  int64_t batch_size) {
  if (readers.empty()) {
    return ::arrow::Status::Invalid("Scan::Make: can not accept zero readers");
  }
  if (std::get<0>(readers[0])->metadata().num_batches() == 0) {
    return ::arrow::Status::IOError("Can not open Scan on empty file");
  }
  return std::unique_ptr<Scan>(new Scan(readers, batch_size));
}

Scan::Scan(const std::vector<FileReaderWithSchema>& readers, int64_t batch_size)
    : readers_(std::begin(readers), std::end(readers)),
      batch_size_(batch_size),
      current_batch_page_length_(std::get<0>(readers_[0])->metadata().GetBatchLength(0)) {}

::arrow::Result<ScanBatch> Scan::Next() {
  assert(!readers_.empty());
  int32_t offset;
  int32_t batch_id;

  auto& first_reader = std::get<0>(readers_[0]);
  {
    // Make the plan to how much data to read next.
    std::lock_guard guard(lock_);
    batch_id = current_batch_id_;
    offset = current_offset_;
    current_offset_ += batch_size_;
    if (current_offset_ >= current_batch_page_length_) {
      current_batch_id_++;
      current_offset_ = 0;
      if (current_batch_id_ < first_reader->metadata().num_batches()) {
        current_batch_page_length_ = first_reader->metadata().GetBatchLength(current_batch_id_);
      }
    }
    // Lock released after scope.
  }

  if (batch_id >= first_reader->metadata().num_batches()) {
    // Reach EOF
    return ScanBatch::Null();
  }

  if (::arrow::GetCpuThreadPoolCapacity() < kMinimalIOThreads) {
    // Keep a minimal number of threads, preventing live lock on low CPU count (<=2) machines,
    // i.e., Github Action runners.
    ARROW_RETURN_NOT_OK(::arrow::SetCpuThreadPoolCapacity(kMinimalIOThreads));
  }
  auto executor = ::arrow::internal::GetCpuThreadPool();
  std::vector<::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>> futs;
  for (auto [reader, schema] : readers_) {
    ARROW_ASSIGN_OR_RAISE(
        auto fut,
        executor->Submit(
            [batch_id, offset](
                std::shared_ptr<lance::io::FileReader> r,
                std::shared_ptr<lance::format::Schema> s,
                auto batch_size) -> ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> {
              ARROW_ASSIGN_OR_RAISE(auto batch, r->ReadBatch(*s, batch_id, offset, batch_size));
              return batch;
            },
            reader,
            schema,
            batch_size_));
    futs.emplace_back(std::move(fut));
  }

  std::vector<std::shared_ptr<::arrow::RecordBatch>> batches;
  for (auto& fut : futs) {
    ARROW_ASSIGN_OR_RAISE(auto& b, fut.result());
    batches.emplace_back(b);
  }

  ARROW_ASSIGN_OR_RAISE(auto batch, lance::arrow::MergeRecordBatches(batches));
  return ScanBatch{
      batch,
      batch_id,
      offset,
  };
}

::arrow::Status Scan::Seek(int32_t offset) {
  assert(!readers_.empty());
  auto& reader = std::get<0>(readers_[0]);
  ARROW_ASSIGN_OR_RAISE(auto batch_and_offset, reader->metadata().LocateBatch(offset));
  current_batch_id_ = std::get<0>(batch_and_offset);
  current_offset_ = std::get<1>(batch_and_offset);
  return ::arrow::Status::OK();
}

std::string Scan::ToString() const { return "Scan"; }

}  // namespace lance::io::exec