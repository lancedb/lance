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

#include "lance/io/reader.h"

namespace lance::io::exec {

Scan::Scan(std::shared_ptr<FileReader> reader,
           std::shared_ptr<lance::format::Schema> schema,
           int64_t batch_size)
    : reader_(std::move(reader)), schema_(std::move(schema)), batch_size_(batch_size) {}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Scan::Next() {
  return reader_->ReadBatch(*schema_, current_batch_id_, current_offset_, batch_size_);
}

}  // namespace lance::io::exec