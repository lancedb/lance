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

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "lance/arrow/type.h"

namespace lance::arrow {

class HashMerger {
 public:
  HashMerger() = default;

  /// Build a hash map on column specified by "col_name".
  ::arrow::Status Build(const ::arrow::Table& table, const std::string& col_name);

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Collect(
      const std::shared_ptr<::arrow::Array>& on_col);

 private:
  std::unordered_map<std::size_t, std::tuple<int64_t, int64_t>> index_map_;
  std::shared_ptr<::arrow::DataType> index_column_type_;
};

}  // namespace lance::arrow
