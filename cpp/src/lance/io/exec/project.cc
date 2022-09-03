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

}  // namespace lance::io::exec