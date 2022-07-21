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

#include "lance/io/limit_offset.h"

#include <fmt/format.h>

#include <algorithm>

namespace lance::io {

Limit::Limit(int32_t limit) noexcept : limit_(limit) {}

std::shared_ptr<::arrow::Array> Limit::Execute(const std::shared_ptr<::arrow::Array>& array) {
  if (!array) {
    return nullptr;
  }
  auto array_length = array->length();
  auto desired_length = Execute(array_length);
  return array->Slice(0, desired_length);
}

int32_t Limit::Execute(int32_t length) {
  auto len = std::min(limit_ - seen_, length);
  seen_ += len;
  return len;
}

std::string Limit::ToString() const { return fmt::format("Limit(n={})", limit_); }

Offset::Offset(int32_t offset) noexcept : offset_(offset) {}

/// Apply offset to the input array.
std::shared_ptr<::arrow::Array> Offset::Execute(const std::shared_ptr<::arrow::Array>& array) {
  if (!array) {
    // nullptr
    return array;
  }
  auto offset = Execute(array->length());
  if (!offset.has_value()) {
    return nullptr;
  }
  return array->Slice(offset.value());
}

std::optional<int32_t> Offset::Execute(int32_t length) {
  if (seen_ >= offset_) {
    return 0;
  }
  seen_ += length;
  if (seen_ < offset_) {
    return std::nullopt;
  }
  return offset_ - (seen_ - length);
}

std::string Offset::ToString() const { return fmt::format("Offset(n={})", offset_); }

}  // namespace lance::io