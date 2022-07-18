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

#include <arrow/compute/exec/expression.h>
#include <arrow/result.h>

#include <memory>
#include <tuple>

#include "lance/format/schema.h"

namespace lance::io {

class FileReader;

/// Filter.
class Filter {
 public:
  Filter() = delete;

  /// Build a filter from arrow's filter expression.
  static ::arrow::Result<std::unique_ptr<Filter>> Make(const lance::format::Schema& schema,
                                                       const ::arrow::compute::Expression& filter);

  /// Execute the filter on a batch.
  ::arrow::Result<
      std::tuple<std::shared_ptr<::arrow::BooleanArray>, std::shared_ptr<::arrow::Array>>>
      Exec(std::shared_ptr<::arrow::RecordBatch>) const;

  /// Execute the filter on a batch.
  ::arrow::Result<
      std::tuple<std::shared_ptr<::arrow::BooleanArray>, std::shared_ptr<::arrow::Array>>>
  Exec(std::shared_ptr<FileReader> reader, int32_t chunk_id) const;

  std::string ToString() const;

 private:
  Filter(std::shared_ptr<lance::format::Schema> schema, const ::arrow::compute::Expression& filter);

  std::shared_ptr<lance::format::Schema> schema_;
  ::arrow::compute::Expression filter_;
};

}  // namespace lance::io
