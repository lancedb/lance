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
  ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema.Project(*projected_arrow_schema));

  std::shared_ptr<Counter> counter;
  if (scan_options->fragment_scan_options &&
      lance::arrow::IsLanceFragmentScanOptions(*scan_options->fragment_scan_options)) {
    auto fso = std::dynamic_pointer_cast<lance::arrow::LanceFragmentScanOptions>(
        scan_options->fragment_scan_options);
    counter = fso->counter;
  }
  std::unique_ptr<ExecNode> child;
  if (Filter::HasFilter(scan_options->filter)) {
    /// Build a chain of:
    ///
    /// Take -> (optionally) Limit -> Filter -> Scan
    ARROW_ASSIGN_OR_RAISE(auto filter_scan_schema, schema.Project(scan_options->filter));
    ARROW_ASSIGN_OR_RAISE(auto filter_scan_node,
                          Scan::Make({{reader, filter_scan_schema}}, scan_options->batch_size));
    ARROW_ASSIGN_OR_RAISE(child, Filter::Make(scan_options->filter, std::move(filter_scan_node)));
    if (counter) {
      ARROW_ASSIGN_OR_RAISE(child, Limit::Make(counter, std::move(child)));
    }
    ARROW_ASSIGN_OR_RAISE(auto take_schema, projected_schema->Exclude(*filter_scan_schema));
    ARROW_ASSIGN_OR_RAISE(child, Take::Make(reader, take_schema, std::move(child)));
  } else {
    /// (*optionally) Limit -> Scan
    ARROW_ASSIGN_OR_RAISE(
        child, Scan::Make({{std::move(reader), projected_schema}}, scan_options->batch_size));
    if (counter) {
      ARROW_ASSIGN_OR_RAISE(child, Limit::Make(counter, std::move(child)));
    }
  }
  return std::unique_ptr<Project>(new Project(std::move(child), std::move(projected_schema)));
}

std::string Project::ToString() const { return "Project"; }

::arrow::Result<ScanBatch> Project::Next() {
  assert(child_);
  ARROW_ASSIGN_OR_RAISE(auto batch, child_->Next());
  return batch;
}

}  // namespace lance::io::exec