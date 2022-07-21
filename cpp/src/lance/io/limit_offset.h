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

namespace lance::io {

/// Plan for Limit clause.
class Limit {
 public:
  Limit() = delete;

  explicit Limit(int32_t limit) noexcept;

  /// Apply limit to the input array.
  std::shared_ptr<::arrow::Array> Execute(const std::shared_ptr<::arrow::Array>& array);

  /// Apply limit to the size of the length.
  ///
  /// \return a positive value. Returns 0 if the limit of records is reached.
  int32_t Execute(int32_t length);

  /// Debug String
  std::string ToString() const;

 private:
  int32_t limit_;
  int32_t seen_ = 0;
};

/// Execution Plan for Offset clause
class Offset {
 public:
  Offset() = delete;

  explicit Offset(int32_t offset) noexcept;

  /// Apply offset to the input array.
  std::shared_ptr<::arrow::Array> Execute(const std::shared_ptr<::arrow::Array>& array);

  /// Apply offset to the size of the length.
  ///
  /// \return a positive value.
  std::optional<int32_t> Execute(int32_t length);

  /// Debug String
  std::string ToString() const;

 private:
  int32_t offset_ = 0;
  int32_t seen_ = 0;
};

}  // namespace lance::io
