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

#include "lance/io/filter.h"

#include <arrow/array.h>
#include <arrow/compute/api.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

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
  auto field_refs = ::arrow::compute::FieldsInExpression(filter);
  std::vector<std::string> columns;
  for (auto& ref : field_refs) {
    columns.emplace_back(std::string(*ref.name()));
  }
  ARROW_ASSIGN_OR_RAISE(auto filter_schema, schema.Project(columns));
  return std::unique_ptr<Filter>(new Filter(filter_schema, filter));
}

::arrow::Result<
    std::tuple<std::shared_ptr<::arrow::UInt64Array>, std::shared_ptr<::arrow::RecordBatch>>>
Filter::Exec(std::shared_ptr<::arrow::RecordBatch> batch) const {
  ARROW_ASSIGN_OR_RAISE(auto filter_expr, filter_.Bind(*(batch->schema())));
  ARROW_ASSIGN_OR_RAISE(auto mask,
                        ::arrow::compute::ExecuteScalarExpression(
                            filter_expr, *(batch->schema()), ::arrow::Datum(batch)));
  ARROW_ASSIGN_OR_RAISE(auto data, batch->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto indices_datum,
                        ::arrow::compute::CallFunction("indices_nonzero", {mask}));
  ARROW_ASSIGN_OR_RAISE(auto values, ::arrow::compute::CallFunction("filter", {data, mask}));

  auto indices = std::static_pointer_cast<::arrow::UInt64Array>(indices_datum.make_array());
  auto values_arr = values.make_array();
  ARROW_ASSIGN_OR_RAISE(auto result_batch, ::arrow::RecordBatch::FromStructArray(values_arr));
  return std::make_tuple(indices, result_batch);
}

std::string Filter::ToString() const { return filter_.ToString(); }

}  // namespace lance::io