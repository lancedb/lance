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

#include <cstdint>
#include <memory>
#include <mutex>

#include "lance/io/exec/base.h"

namespace lance::io::exec {

/// Data structure to count the number of records.
class Counter {
 public:
  Counter() = delete;

  /// Construct
  Counter(int64_t limit, int64_t offset) noexcept;

  /// Slice a RecordBatch, and update the number of seen records.
  ScanBatch Slice(const ScanBatch& batch);

  /// Returns true if there are more records.
  bool HasMore() const;

  int64_t limit() const;

  int64_t offset() const;

 private:
  int64_t limit_;
  int64_t offset_;
  int64_t seen_ = 0;

  std::mutex lock_;
};

}  // namespace lance::io::exec