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
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#include "lance/io/exec/base.h"
#include "lance/io/exec/counter.h"

namespace lance::io::exec {

/// Node for the Limit clause:
///
///   LIMIT value:int64 [OFFSET value:int64]
///
class Limit : public ExecNode {
 public:
  Limit() = delete;

  /// Factory method.
  static ::arrow::Result<std::unique_ptr<ExecNode>> Make(std::shared_ptr<Counter> counter,
                                                         std::unique_ptr<ExecNode> child) noexcept;

  /// Construct a Limit Clause with limit, and optionally, with offset.
  explicit Limit(std::shared_ptr<Counter> counter, std::unique_ptr<ExecNode> child) noexcept;

  /// Apply the limits and returns the next batch.
  ::arrow::Result<ScanBatch> Next() override;

  constexpr Type type() const override { return kLimit; }

  /// Debug String
  std::string ToString() const override;

 private:
  std::shared_ptr<Counter> counter_;
  std::unique_ptr<ExecNode> child_;
};

}  // namespace lance::io::exec
