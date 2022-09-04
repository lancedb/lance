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

#include <arrow/compute/api.h>
#include <arrow/result.h>

#include <memory>
#include <tuple>

#include "lance/format/schema.h"
#include "lance/io/exec/base.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

/// Filter.
class Filter : public ExecNode {
 public:
  Filter() = delete;

  /// Build a filter from arrow's filter expression and dataset schema.
  static ::arrow::Result<std::unique_ptr<Filter>> Make(const ::arrow::compute::Expression& filter,
                                                       std::unique_ptr<ExecNode> child);

  /// Returns true if the filter expression has filter over actual columns.
  static bool HasFilter(const ::arrow::compute::Expression& filter);

  ::arrow::Result<ScanBatch> Next() override;

  constexpr Type type() const override { return Type::kFilter; }

  std::string ToString() const override;

 private:
  Filter(const ::arrow::compute::Expression& filter, std::unique_ptr<ExecNode> child);

  /// Execute the filter on an arrow RecordBatch.
  ///
  /// \return a tuple of [indices, filtered_array].
  ///
  /// For example, with a record batch of {"bar": [0, 2, 32, 5, 32]}, and filter "bar = 32",
  /// this function returns:
  ///
  /// { Int32Array({2, 4}), RecordBatch({"bar": [32, 32]}) }
  ::arrow::Result<
      std::tuple<std::shared_ptr<::arrow::Int32Array>, std::shared_ptr<::arrow::RecordBatch>>>
  Apply(const ::arrow::RecordBatch& batch) const;

  ::arrow::compute::Expression filter_;
  std::unique_ptr<ExecNode> child_;
};

}  // namespace lance::io::exec
