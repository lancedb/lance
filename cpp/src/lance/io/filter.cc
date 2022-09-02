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

#include "lance/arrow/type.h"
#include "lance/io/reader.h"

namespace lance::io {

Filter::Filter(std::shared_ptr<lance::format::Schema> schema,
               const ::arrow::compute::Expression& filter)
    : schema_(schema), filter_(filter) {}

::arrow::Result<std::unique_ptr<Filter>> Filter::Make(const lance::format::Schema& schema,
                                                      const ::arrow::compute::Expression& filter) {
  ARROW_ASSIGN_OR_RAISE(auto filter_schema, schema.Project(filter));
  if (!filter_schema) {
    return nullptr;
  }
  return std::unique_ptr<Filter>(new Filter(filter_schema, filter));
}

::arrow::Result<
    std::tuple<std::shared_ptr<::arrow::Int32Array>, std::shared_ptr<::arrow::RecordBatch>>>
Filter::Execute(std::shared_ptr<::arrow::RecordBatch> batch) const {
  ARROW_ASSIGN_OR_RAISE(auto filter_expr, filter_.Bind(*(batch->schema())));
  ARROW_ASSIGN_OR_RAISE(auto mask,
                        ::arrow::compute::ExecuteScalarExpression(
                            filter_expr, *(batch->schema()), ::arrow::Datum(batch)));
  ARROW_ASSIGN_OR_RAISE(auto data, batch->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto indices_datum,
                        ::arrow::compute::CallFunction("indices_nonzero", {mask}));
  ARROW_ASSIGN_OR_RAISE(indices_datum, ::arrow::compute::Cast(indices_datum, ::arrow::int32()));
  ARROW_ASSIGN_OR_RAISE(auto values, ::arrow::compute::CallFunction("filter", {data, mask}));

  auto indices = std::static_pointer_cast<::arrow::Int32Array>(indices_datum.make_array());
  auto values_arr = values.make_array();
  ARROW_ASSIGN_OR_RAISE(auto result_batch, ::arrow::RecordBatch::FromStructArray(values_arr));
  return std::make_tuple(indices, result_batch);
}

::arrow::Result<
    std::tuple<std::shared_ptr<::arrow::Int32Array>, std::shared_ptr<::arrow::RecordBatch>>>
Filter::Execute(std::shared_ptr<FileReader> reader, int32_t batch_id) const {
  ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadBatch(*schema_, batch_id));
  return Execute(batch);
}

const std::shared_ptr<lance::format::Schema>& Filter::schema() const { return schema_; }

std::string Filter::ToString() const { return filter_.ToString(); }

}  // namespace lance::io