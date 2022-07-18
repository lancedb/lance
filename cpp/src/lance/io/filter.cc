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

#include "filter.h"

#include <arrow/array.h>
#include <arrow/result.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "lance/arrow/type.h"

namespace lance::io {

Filter::Filter(std::shared_ptr<lance::format::Schema> schema,
               const ::arrow::compute::Expression& filter)
    : schema_(schema), filter_(filter) {}

::arrow::Result<std::unique_ptr<Filter>> Filter::Make(const lance::format::Schema& schema,
                                                      const ::arrow::compute::Expression& filter) {
  if (!::arrow::compute::ExpressionHasFieldRefs(filter)) {
    /// All scalar?
    return nullptr;
  }
  fmt::print("Making filter: {}, type={}\n",
             filter.ToString(),
             ::arrow::compute::ExpressionHasFieldRefs(filter));
  auto field_refs = ::arrow::compute::FieldsInExpression(filter);
  fmt::print("Field_refs: {}\n", field_refs);
  auto call = filter.call();
  if (call != nullptr) {
    if (call->function_name == "equal") {
      fmt::print("This is an equal\n");
    }
  }
  auto field_ref = filter.field_ref();
  if (field_ref == nullptr) {
    fmt::print("This one does not have field_ref, {}\n", filter.ToString());
  }
  return nullptr;
}

::arrow::Result<std::tuple<::arrow::BooleanArray, ::arrow::Array>> Filter::Exec(
    std::shared_ptr<::arrow::RecordBatch>) const {}

::arrow::Result<std::tuple<::arrow::BooleanArray, ::arrow::Array>> Filter::Exec(
    std::shared_ptr<FileReader> reader, int32_t chunk_id) const {
  return ::arrow::Status::NotImplemented("not implemented");
}

std::string Filter::ToString() const { return ""; }

}  // namespace lance::io