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

#include "lance/io/project.h"

#include <arrow/api.h>
#include <arrow/result.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "lance/arrow/utils.h"
#include "lance/io/filter.h"
#include "lance/io/limit.h"
#include "lance/io/reader.h"

namespace lance::io {

Project::Project(std::shared_ptr<format::Schema> dataset_schema,
                 std::shared_ptr<format::Schema> projected_schema,
                 std::shared_ptr<format::Schema> scan_schema,
                 std::unique_ptr<Filter> filter,
                 std::optional<int32_t> limit,
                 int32_t offset)
    : dataset_schema_(dataset_schema),
      projected_schema_(projected_schema),
      scan_schema_(scan_schema),
      filter_(std::move(filter)),
      limit_(limit.has_value() ? new Limit(limit.value(), offset) : nullptr) {}

::arrow::Result<std::unique_ptr<Project>> Project::Make(
    std::shared_ptr<format::Schema> schema,
    std::shared_ptr<::arrow::dataset::ScanOptions> scan_options,
    std::optional<int32_t> limit,
    int32_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto filter, Filter::Make(*schema, scan_options->filter));
  auto projected_arrow_schema = scan_options->projected_schema;
  if (projected_arrow_schema->num_fields() == 0) {
    projected_arrow_schema = scan_options->dataset_schema;
  }
  ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema->Project(*projected_arrow_schema));
  auto scan_schema = projected_schema;
  if (filter) {
    // Remove the columns in filter from the project schema, to avoid duplicated scan
    ARROW_ASSIGN_OR_RAISE(scan_schema, projected_schema->Exclude(filter->schema()));
  }
  return std::unique_ptr<Project>(
      new Project(schema, projected_schema, scan_schema, std::move(filter), limit, offset));
}

const std::shared_ptr<format::Schema>& Project::schema() const { return projected_schema_; }

bool Project::CanParallelScan() const { return limit_.operator bool(); }

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Project::Execute(
    std::shared_ptr<FileReader> reader, int32_t batch_id) {
  if (filter_) {
    auto result = filter_->Execute(reader, batch_id);
    if (!result.ok()) {
      return result.status();
    }
    auto [indices, values] = result.ValueUnsafe();
    assert(indices->length() == values->num_rows());
    if (limit_) {
      auto offset_and_len = limit_->Apply(indices->length());
      if (!offset_and_len.has_value()) {
        /// Indicate the end of iteration.
        return nullptr;
      }
      auto [offset, len] = offset_and_len.value();
      indices =
          std::static_pointer_cast<decltype(indices)::element_type>(indices->Slice(offset, len));
      values = values->Slice(offset, len);
    }
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadBatch(*scan_schema_, batch_id, indices));
    assert(values->num_rows() == batch->num_rows());
    ARROW_ASSIGN_OR_RAISE(auto merged, lance::arrow::MergeRecordBatches(values, batch));
    return merged;
  } else {
    // Read without filter.
    if (limit_) {
      return limit_->ReadBatch(reader, *scan_schema_);
    } else {
      return reader->ReadBatch(*scan_schema_, batch_id);
    }
  }
}

}  // namespace lance::io