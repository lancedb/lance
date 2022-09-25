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

#include "lance/arrow/file_lance_ext.h"
#include "lance/arrow/utils.h"
#include "lance/io/exec/filter.h"
#include "lance/io/exec/limit.h"
#include "lance/io/exec/scan.h"
#include "lance/io/exec/take.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

Project::Project(std::unique_ptr<ExecNode> child,
                 std::shared_ptr<lance::format::Schema> projected_schema)
    : child_(std::move(child)), projected_schema_(std::move(projected_schema)) {}

::arrow::Result<std::unique_ptr<Project>> Project::Make(
    std::shared_ptr<FileReader> reader,
    std::shared_ptr<::arrow::dataset::ScanOptions> scan_options) {
  auto& schema = reader->schema();
  auto projected_arrow_schema = scan_options->projected_schema;
  if (projected_arrow_schema->num_fields() == 0) {
    projected_arrow_schema = scan_options->dataset_schema;
  }
  fmt::print("Scan Options:\nprojected={}\n-----\ndataset={}\n-----\n",
             scan_options->projected_schema->ToString(),
             scan_options->dataset_schema->ToString());
  ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema.Project(*projected_arrow_schema));
  fmt::print("After project schema: schema={} projected={}\n", schema, projected_schema);

  std::optional<int64_t> limit;
  int64_t offset;
  if (scan_options->fragment_scan_options &&
      lance::arrow::IsLanceFragmentScanOptions(*scan_options->fragment_scan_options)) {
    auto fso = std::dynamic_pointer_cast<lance::arrow::LanceFragmentScanOptions>(
        scan_options->fragment_scan_options);
    limit = fso->limit;
    offset = fso->offset;
  }
  std::unique_ptr<ExecNode> child;
  if (Filter::HasFilter(scan_options->filter)) {
    /// Build a chain of:
    ///
    /// Take -> (optionally) Limit -> Filter -> Scan
    ARROW_ASSIGN_OR_RAISE(auto filter_scan_schema, schema.Project(scan_options->filter));
    fmt::print("Has filter: filter scan schema: {}\n", filter_scan_schema->ToString());
    ARROW_ASSIGN_OR_RAISE(auto filter_scan_node,
                          Scan::Make(reader, filter_scan_schema, scan_options->batch_size));
    ARROW_ASSIGN_OR_RAISE(child, Filter::Make(scan_options->filter, std::move(filter_scan_node)));
    if (limit.has_value()) {
      ARROW_ASSIGN_OR_RAISE(child, Limit::Make(limit.value(), offset, std::move(child)));
    }
    ARROW_ASSIGN_OR_RAISE(auto take_schema, projected_schema->Exclude(*filter_scan_schema));
    fmt::print("Take schema: {}\n", take_schema);
    ARROW_ASSIGN_OR_RAISE(child, Take::Make(reader, take_schema, std::move(child)));
  } else {
    /// (*optionally) Limit -> Scan
    fmt::print("Build scan with schema: {}\n", projected_schema->ToString());
    ARROW_ASSIGN_OR_RAISE(child, Scan::Make(reader, projected_schema, scan_options->batch_size));
    if (limit.has_value()) {
      ARROW_ASSIGN_OR_RAISE(child, Limit::Make(limit.value(), offset, std::move(child)));
    }
  }
  fmt::print("Create Project: projected schema={}\n", projected_schema->ToString());
  return std::unique_ptr<Project>(new Project(std::move(child), std::move(projected_schema)));
}

std::string Project::ToString() const { return "Project"; }

::arrow::Result<ScanBatch> Project::Next() {

  assert(child_);
  ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
  if (batch.eof()) {
    return batch;
  }
  fmt::print("After read, Projected schema: {}\n, actual dataset schema: {}\n",
             projected_schema_->ToString(),
             !batch.eof() ? batch.batch->schema()->ToString() : "{}");
  ARROW_ASSIGN_OR_RAISE(auto projected_batch,
                        lance::arrow::ApplyProjection(batch.batch, *projected_schema_));
  fmt::print("Projected batch: {}\n", projected_batch->ToString());
  return ScanBatch(projected_batch, batch.batch_id, batch.indices);
}

}  // namespace lance::io::exec