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
///   LIMIT value:int64 [OFFSET value:int64]
///
class Limit {
 public:
  Limit() = delete;

  /// Construct a Limit Clause with limit, and optionally, offset.
  explicit Limit(int64_t limit, int64_t offset = 0) noexcept;

  /// Apply limit when reading a chunk.
  ///
  /// \param chunk_length the length of a chunk to be read.
  ///
  /// \return a tuple of `[offset, length]` that should be physically loaded
  /// into memory. Return `[0, 0]` to skip this chunk.
  /// Return `std::nullopt` to indicate the end of the iteration.
  ///
  /// \code{.cpp}
  /// auto chunk_length = GetChunkLength(chunk_id);
  /// auto limit = Limit(20, 30);
  /// auto offset_and_length = limit.Apply(chunk_length);
  /// if (!offset_and_length) {
  ///    // stop
  ///    return;
  /// }
  /// auto [offset, length] = offset_and_length;
  /// return infile->ReadAt(offset, length);
  /// \endcode
  std::optional<std::tuple<int64_t, int64_t>> Apply(int64_t chunk_length);

  /// Debug String
  std::string ToString() const;

 private:
  int64_t limit_ = 0;
  int64_t offset_ = 0;
  int64_t seen_ = 0;
};

}  // namespace lance::io
