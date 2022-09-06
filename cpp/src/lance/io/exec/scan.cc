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

#include <fmt/format.h>

#include <memory>

#include "lance/format/metadata.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

::arrow::Result<std::unique_ptr<Scan>> Scan::Make(std::shared_ptr<FileReader> reader,
                                                  std::shared_ptr<lance::format::Schema> schema,
                                                  int64_t batch_size) {
  auto scan = std::unique_ptr<Scan>(new Scan(reader, schema, batch_size));
  if (reader->metadata().num_batches() == 0) {
    return ::arrow::Status::IOError("Can not open Scan on empty file");
  }
  scan->current_batch_page_length_ = reader->metadata().GetBatchLength(0);
  return scan;
}

Scan::Scan(std::shared_ptr<FileReader> reader,
           std::shared_ptr<lance::format::Schema> schema,
           int64_t batch_size)
    : reader_(std::move(reader)), schema_(std::move(schema)), batch_size_(batch_size) {}

::arrow::Result<ScanBatch> Scan::Next() {
  int32_t offset;
  int32_t batch_id;

  {
    std::lock_guard guard(lock_);
    batch_id = current_batch_id_;
    offset = current_offset_;
    current_offset_ += batch_size_;
    if (current_offset_ >= current_batch_page_length_) {
      current_batch_id_++;
      current_offset_ = 0;
      if (current_batch_id_ < reader_->metadata().num_batches()) {
        current_batch_page_length_ = reader_->metadata().GetBatchLength(current_batch_id_);
      }
    }
  }
  if (batch_id >= reader_->metadata().num_batches()) {
    // Reach EOF
    return ScanBatch::Null();
  }

  ARROW_ASSIGN_OR_RAISE(auto batch, reader_->ReadBatch(*schema_, batch_id, offset, batch_size_));
  return ScanBatch{
      batch,
      batch_id,
  };
}

::arrow::Status Scan::Seek(int32_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto batch_and_offset, reader_->metadata().LocateBatch(offset));
  current_batch_id_ = std::get<0>(batch_and_offset);
  current_offset_ = std::get<1>(batch_and_offset);
  return ::arrow::Status::OK();
}

std::string Scan::ToString() const { return "Scan"; }

}  // namespace lance::io::exec