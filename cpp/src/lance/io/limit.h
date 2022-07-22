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

#include <arrow/array.h>
#include <arrow/result.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace lance::io {

/// Plan for Limit clause:
///
///    LIMIT value:int64 [OFFSET value:int64]
///
class Limit {
 public:
  Limit() = delete;

  explicit Limit(int64_t limit, int64_t offset = 0) noexcept;

  /// Apply limit to the size of the length.
  ///
  /// \return a tuple of [offset, length]. Return [0, 0] to skip a chunk.
  /// Returns std::nullopt to indicate the end of the iteration.
  std::optional<std::tuple<int64_t, int64_t>> Apply(int64_t length);

  /// Debug String
  std::string ToString() const;

 private:
  int64_t limit_;
  int64_t offset_ = 0;
  int64_t seen_ = 0;
};

}  // namespace lance::io
