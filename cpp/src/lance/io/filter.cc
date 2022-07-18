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
#include <arrow/record_batch.h>
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
             filter,
             ::arrow::compute::ExpressionHasFieldRefs(filter));
  auto field_refs = ::arrow::compute::FieldsInExpression(filter);
  fmt::print("Filters= {}\n", field_refs);
  std::vector<std::string> columns;
  for (auto& ref : field_refs) {
    columns.emplace_back(std::string(*ref.name()));
  }
  fmt::print("All columns: {}\n", columns);
  ARROW_ASSIGN_OR_RAISE(auto filter_schema, schema.Project(columns));
  return std::unique_ptr<Filter>(new Filter(filter_schema, filter));
}

::arrow::Result<std::tuple<std::shared_ptr<::arrow::BooleanArray>, std::shared_ptr<::arrow::Array>>>
Filter::Exec(std::shared_ptr<::arrow::RecordBatch> batch) const {
  ARROW_ASSIGN_OR_RAISE(auto filter_expr, filter_.Bind(*(batch->schema())));
  ARROW_ASSIGN_OR_RAISE(auto mask,
                        ::arrow::compute::ExecuteScalarExpression(
                            filter_expr, *(batch->schema()), ::arrow::Datum(batch)));
  ARROW_ASSIGN_OR_RAISE(auto values, batch->ToStructArray());
  return std::make_tuple(std::static_pointer_cast<::arrow::BooleanArray>(mask.make_array()),
                         values);
}

::arrow::Result<std::tuple<std::shared_ptr<::arrow::BooleanArray>, std::shared_ptr<::arrow::Array>>>
Filter::Exec(std::shared_ptr<FileReader> reader, int32_t chunk_id) const {
  return ::arrow::Status::NotImplemented("not implemented");
}

std::string Filter::ToString() const { return ""; }

}  // namespace lance::io