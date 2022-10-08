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

#include "lance/io/exec/limit.h"

#include <arrow/array/util.h>
#include <arrow/record_batch.h>
#include <fmt/format.h>

#include "lance/io/exec/scan.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

::arrow::Result<std::unique_ptr<ExecNode>> Limit::Make(std::shared_ptr<Counter> counter,
                                                       std::unique_ptr<ExecNode> child) noexcept {
  auto limit_node = std::make_unique<Limit>(std::move(counter), std::move(child));
  return limit_node;
}

Limit::Limit(std::shared_ptr<Counter> counter, std::unique_ptr<ExecNode> child) noexcept
    : counter_(std::move(counter)), child_(std::move(child)) {}

::arrow::Result<ScanBatch> Limit::Next() {
  if (!counter_->HasMore()) {
    return ScanBatch{};
  }
  ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
  if (batch.eof()) {
    return batch;
  }
  return counter_->Slice(batch);
}

std::string Limit::ToString() const {
  return fmt::format("Limit(n={}, offset={})", counter_->limit(), counter_->offset());
}

}  // namespace lance::io::exec