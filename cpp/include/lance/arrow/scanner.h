//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <arrow/compute/exec/expression.h>
#include <arrow/dataset/scanner.h>
#include <arrow/dataset/type_fwd.h>
#include <arrow/result.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lance::arrow {

/// \brief Lance Scanner Builder
///
/// The main difference between ScannerBuilder and `::arrow::ScannerBuilder` is that
/// the lance one allows fine-grained project/predicate/limit/offset push-downs.
///
/// We can deprecate this class once we've pushed these features to upstream project.
class ScannerBuilder final {
 public:
  /// Construct ScannerBuilder with an arrow Dataset
  ///
  /// \param dataset An Arrow Dataset.
  explicit ScannerBuilder(std::shared_ptr<::arrow::dataset::Dataset> dataset);

  ~ScannerBuilder() = default;

  /// Project over selected columns.
  ///
  /// \param columns Selected column names.
  ::arrow::Status Project(const std::vector<std::string>& columns);

  /// Set batch size to scan.
  ::arrow::Status BatchSize(int64_t batch_size);

  /// Apply Filter
  ::arrow::Status Filter(const ::arrow::compute::Expression& filter);

  /// Set limit and offset to scan.
  ::arrow::Status Limit(int64_t limit, int64_t offset = 0);

  ::arrow::Result<std::shared_ptr<::arrow::dataset::Scanner>> Finish();

 private:
  ::arrow::dataset::ScannerBuilder builder_;
  std::optional<std::vector<std::string>> columns_ = std::nullopt;
};

}  // namespace lance::arrow