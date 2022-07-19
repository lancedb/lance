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

#include "lance/arrow/utils.h"

#include <arrow/result.h>
#include <fmt/format.h>

#include <string>
#include <vector>

#include "lance/arrow/type.h"

namespace lance::arrow {

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Merge(
    const std::shared_ptr<::arrow::RecordBatch>& lhs,
    const std::shared_ptr<::arrow::RecordBatch>& rhs,
    ::arrow::MemoryPool* pool) {
  ARROW_ASSIGN_OR_RAISE(auto left_struct, lhs->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto right_struct, rhs->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto struct_arr, Merge(left_struct, right_struct, pool));
  return ::arrow::RecordBatch::FromStructArray(struct_arr);
}

::arrow::Result<std::shared_ptr<::arrow::StructArray>> Merge(
    const std::shared_ptr<::arrow::StructArray>& lhs,
    const std::shared_ptr<::arrow::StructArray>& rhs,
    ::arrow::MemoryPool* pool) {
  if (lhs->length() != rhs->length()) {
    return ::arrow::Status::Invalid("Two StructArrays have different length");
  }
  std::vector<std::string> names;
  ::arrow::ArrayVector arrays;
  auto left_type = lhs->struct_type();
  for (auto& field : left_type->fields()) {
    auto& name = field->name();
    names.emplace_back(name);
    auto left_arr = lhs->GetFieldByName(name);

    auto right_arr = rhs->GetFieldByName(name);
    if (right_arr) {
      if (!is_struct(left_arr->type()) || !is_struct(right_arr->type())) {
        return ::arrow::Status::Invalid(
            fmt::format("Can only merge two struct types: left={} right={}",
                        left_arr->type(),
                        right_arr->type()));
      }
      ARROW_ASSIGN_OR_RAISE(left_arr,
                            Merge(std::static_pointer_cast<::arrow::StructArray>(left_arr),
                                  std::static_pointer_cast<::arrow::StructArray>(right_arr),
                                  pool));
    }
    arrays.emplace_back(left_arr);
  }

  for (auto& field : rhs->struct_type()->fields()) {
    auto& name = field->name();
    if (lhs->GetFieldByName(name)) {
      // We've seen each other before
      continue;
    }
    names.emplace_back(name);
    arrays.emplace_back(rhs->GetFieldByName(name));
  }
  return ::arrow::StructArray::Make(arrays, names);
}

}  // namespace lance::arrow