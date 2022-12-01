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
#include <unordered_map>

#include "lance/arrow/type.h"

namespace lance::arrow {

/// A basic implementation of in-memory hash (join) merge.
///
class HashMerger {
 public:
  HashMerger() = delete;

  /// HashMerger constructor.
  explicit HashMerger(std::shared_ptr<::arrow::Table> table,
                      std::string index_column,
                      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  ~HashMerger();

  /// Build a hash map on column specified by "column".
  ::arrow::Status Init();

  /// Collect the batch records with the same keys in the column.
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Collect(
      const std::shared_ptr<::arrow::Array>& index_arr);

 private:
  std::shared_ptr<::arrow::Table> table_;
  std::string column_name_;

  class Impl;
  std::unique_ptr<Impl> impl_;
  /// A map from `std::hash(key)` to the index (`int64_t`) in the table.
  std::unordered_map<std::size_t, int64_t> index_map_;
  std::shared_ptr<::arrow::DataType> index_column_type_;
  ::arrow::MemoryPool* pool_;

  template <ArrowType T, typename CType>
  friend class TypedHashMerger;
};

}  // namespace lance::arrow
