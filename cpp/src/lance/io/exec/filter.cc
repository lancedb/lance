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

#include "lance/io/exec/filter.h"

#include <arrow/array.h>
#include <arrow/compute/api.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include "lance/io/reader.h"

namespace lance::io::exec {

Filter::Filter(const ::arrow::compute::Expression& filter, std::unique_ptr<ExecNode> child)
    : filter_(filter), child_(std::move(child)) {}

::arrow::Result<std::unique_ptr<Filter>> Filter::Make(const ::arrow::compute::Expression& filter,
                                                      std::unique_ptr<ExecNode> scan) {
  return std::unique_ptr<Filter>(new Filter(filter, std::move(scan)));
}

bool Filter::HasFilter(const ::arrow::compute::Expression& filter) {
  return ::arrow::compute::ExpressionHasFieldRefs(filter);
}

::arrow::Result<ScanBatch> Filter::Next() {
  ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
  if (batch.eof()) {
    return ScanBatch::Null();
  }
  if (batch.length() == 0) {
    return batch;
  }
  ARROW_ASSIGN_OR_RAISE(auto indices_and_values, Apply(*batch.batch));
  auto [indices, values] = indices_and_values;
  ARROW_ASSIGN_OR_RAISE(auto values_arr, values->ToStructArray());
  return ScanBatch(values, batch.batch_id, batch.offset, indices);
}

::arrow::Result<
    std::tuple<std::shared_ptr<::arrow::Int32Array>, std::shared_ptr<::arrow::RecordBatch>>>
Filter::Apply(const ::arrow::RecordBatch& batch) const {
  ARROW_ASSIGN_OR_RAISE(auto filter_expr, filter_.Bind(*(batch.schema())));
  ARROW_ASSIGN_OR_RAISE(auto mask,
                        ::arrow::compute::ExecuteScalarExpression(
                            filter_expr, *(batch.schema()), ::arrow::Datum(batch)));
  ARROW_ASSIGN_OR_RAISE(auto data, batch.ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto indices_datum,
                        ::arrow::compute::CallFunction("indices_nonzero", {mask}));
  ARROW_ASSIGN_OR_RAISE(indices_datum, ::arrow::compute::Cast(indices_datum, ::arrow::int32()));
  ARROW_ASSIGN_OR_RAISE(auto values, ::arrow::compute::CallFunction("filter", {data, mask}));

  auto indices = std::static_pointer_cast<::arrow::Int32Array>(indices_datum.make_array());
  auto values_arr = values.make_array();
  ARROW_ASSIGN_OR_RAISE(auto result_batch, ::arrow::RecordBatch::FromStructArray(values_arr));
  assert(indices->length() == result_batch->num_rows());
  return std::make_tuple(indices, result_batch);
}

std::string Filter::ToString() const { return filter_.ToString(); }

}  // namespace lance::io::exec