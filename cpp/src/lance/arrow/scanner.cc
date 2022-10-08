//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/arrow/scanner.h"

#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include "lance/arrow/file_lance_ext.h"
#include "lance/arrow/utils.h"
#include "lance/format/schema.h"

namespace lance::arrow {

ScannerBuilder::ScannerBuilder(std::shared_ptr<::arrow::dataset::Dataset> dataset)
    : builder_(dataset) {}

::arrow::Status ScannerBuilder::Project(const std::vector<std::string>& columns) {
  columns_ = columns;
  return ::arrow::Status::OK();
}

::arrow::Status ScannerBuilder::Filter(const ::arrow::compute::Expression& filter) {
  return builder_.Filter(filter);
}

::arrow::Status ScannerBuilder::BatchSize(int64_t batch_size) {
  return builder_.BatchSize(batch_size);
}

::arrow::Status ScannerBuilder::Limit(int64_t limit, int64_t offset) {
  if (limit <= 0 || offset < 0) {
    return ::arrow::Status::Invalid("Limit / offset is invalid: limit=", limit, " offset=", offset);
  }
  auto fragment_opts = std::make_shared<LanceFragmentScanOptions>();
  fragment_opts->counter = std::make_shared<lance::io::exec::Counter>(limit, offset);

  return builder_.FragmentScanOptions(fragment_opts);
}

::arrow::Result<std::shared_ptr<::arrow::dataset::Scanner>> ScannerBuilder::Finish() {
  ARROW_ASSIGN_OR_RAISE(auto scanner, builder_.Finish());

  /// We do the schema projection manually here to support nested structs.
  /// Esp. for `list<struct>`, supports Spark-like access, for example,
  ///
  /// for schema: `objects: list<struct<id:int, value:float>>`,
  /// We can access subfields via column name `objects.value`.
  ///
  /// TODO: contribute this change to Apache Arrow.
  if (columns_.has_value()) {
    auto schema = lance::format::Schema(scanner->options()->dataset_schema);
    auto columns = columns_.value();

    ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema.Project(columns));

    scanner->options()->projected_schema = projected_schema->ToArrow();
    auto dataset_schema = scanner->options()->projected_schema;
    if (::arrow::compute::ExpressionHasFieldRefs(scanner->options()->filter)) {
      ARROW_ASSIGN_OR_RAISE(auto filter_schema, schema.Project(scanner->options()->filter));
      // This is a hack for GH204. To make filter column and projection columns all available
      //
      ARROW_ASSIGN_OR_RAISE(dataset_schema,
                            lance::arrow::MergeSchema(*dataset_schema, *filter_schema->ToArrow()));
    }

    scanner->options()->dataset_schema = dataset_schema;
    ARROW_ASSIGN_OR_RAISE(scanner->options()->filter,
                          scanner->options()->filter.Bind(*scanner->options()->dataset_schema));

    std::vector<std::string> top_names;
    for (const auto& field : projected_schema->fields()) {
      top_names.emplace_back(field->name());
    }
    ARROW_ASSIGN_OR_RAISE(auto project_desc,
                          ::arrow::dataset::ProjectionDescr::FromNames(
                              top_names, *scanner->options()->dataset_schema));
    scanner->options()->projection = project_desc.expression;
  }

  if (scanner->options()->fragment_scan_options) {
    auto fso = std::dynamic_pointer_cast<LanceFragmentScanOptions>(
        scanner->options()->fragment_scan_options);
    if (fso->counter) {
      scanner->options()->batch_size = fso->counter->limit();
      /// We need to limit the parallelism for Project to calculate LIMIT / Offset
      scanner->options()->batch_readahead = 1;
    }
  }
  return scanner;
}

}  // namespace lance::arrow
