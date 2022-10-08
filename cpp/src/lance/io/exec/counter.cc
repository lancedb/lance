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

#include "lance/io/exec/counter.h"

namespace lance::io::exec {

Counter::Counter(int64_t limit, int64_t offset) noexcept : limit_(limit), offset_(offset) {}

ScanBatch Counter::Slice(const ScanBatch& batch) {
  int64_t start = 0;
  int64_t length = 0;
  auto batch_size = batch.length();

  {
    std::lock_guard guard(lock_);

    auto left = std::max(offset_, seen_);
    auto right = std::min(seen_ + batch_size, offset_ + limit_);
    if (left < right) {
      start = left - seen_;
      length = right - left;
    }
    seen_ += batch_size;
  }

  return batch.Slice(start, length);
}

bool Counter::HasMore() const {
  std::lock_guard guard(const_cast<std::mutex&>(lock_));
  return seen_ < offset_ + limit_;
}

int64_t Counter::limit() const { return limit_; }

int64_t Counter::offset() const { return offset_; }

}  // namespace lance::io::exec