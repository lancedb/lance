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

#include "lance/io/exec/project.h"

#include <arrow/api.h>
#include <arrow/result.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "lance/arrow/utils.h"
#include "lance/io/exec/filter.h"
#include "lance/io/exec/limit.h"
#include "lance/io/exec/scan.h"
#include "lance/io/exec/take.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

Project::Project(std::unique_ptr<ExecNode> child) : child_(std::move(child)) {}

::arrow::Result<std::unique_ptr<Project>> Project::Make(
    std::shared_ptr<FileReader> reader,
    std::shared_ptr<::arrow::dataset::ScanOptions> scan_options,
    std::optional<int32_t> limit,
    int32_t offset) {
  auto& schema = reader->schema();
  auto projected_arrow_schema = scan_options->projected_schema;
  if (projected_arrow_schema->num_fields() == 0) {
    projected_arrow_schema = scan_options->dataset_schema;
  }
  ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema.Project(*projected_arrow_schema));

  std::unique_ptr<ExecNode> child;
  if (Filter::HasFilter(scan_options->filter)) {
    /// Build a chain of:
    ///
    /// Take -> (optionally) Limit -> Filter -> Scan
    ARROW_ASSIGN_OR_RAISE(auto scan_schema, schema.Project(scan_options->filter));
    ARROW_ASSIGN_OR_RAISE(auto scan_node,
                          Scan::Make(reader, scan_schema, scan_options->batch_size));
    ARROW_ASSIGN_OR_RAISE(child, Filter::Make(scan_options->filter, std::move(scan_node)));
    if (limit.has_value()) {
      ARROW_ASSIGN_OR_RAISE(child, Limit::Make(limit.value(), offset, std::move(child)));
    }
    ARROW_ASSIGN_OR_RAISE(auto take_schema, projected_schema->Exclude(*scan_schema));
    ARROW_ASSIGN_OR_RAISE(child, Take::Make(reader, take_schema, std::move(child)));
  } else {
    /// (*optionally) Limit -> Scan
    ARROW_ASSIGN_OR_RAISE(child, Scan::Make(reader, projected_schema, scan_options->batch_size));
    if (limit.has_value()) {
      ARROW_ASSIGN_OR_RAISE(child, Limit::Make(limit.value(), offset, std::move(child)));
    }
  }
  return std::unique_ptr<Project>(new Project(std::move(child)));
}

const std::shared_ptr<format::Schema>& Project::schema() const { return projected_schema_; }

std::string Project::ToString() const { return "Project"; }

::arrow::Result<ScanBatch> Project::Next() {
  assert(child_);
  return child_->Next();
}

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
    if (scan_schema_->fields().empty()) {
      // No extra columns other than the filtered columns need to be read, for example,
      // SELECT id FROM t WHERE id > 10
      return values;
    } else {
      ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadBatch(*scan_schema_, batch_id, indices));
      assert(values->num_rows() == batch->num_rows());
      ARROW_ASSIGN_OR_RAISE(auto merged, lance::arrow::MergeRecordBatches(values, batch));
      return merged;
    }
  } else {
    // Read without filter.
    if (limit_) {
      return limit_->ReadBatch(reader, *scan_schema_);
    } else {
      return reader->ReadBatch(*scan_schema_, batch_id);
    }
  }
}

}  // namespace lance::io::exec