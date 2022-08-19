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
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/file_lance_ext.h"
#include "lance/arrow/utils.h"
#include "lance/format/schema.h"

namespace lance::arrow {

ScannerBuilder::ScannerBuilder(std::shared_ptr<::arrow::dataset::Dataset> dataset)
    : dataset_(dataset) {}

void ScannerBuilder::Project(const std::vector<std::string>& columns) { columns_ = columns; }

void ScannerBuilder::Filter(const ::arrow::compute::Expression& filter) { filter_ = filter; }

void ScannerBuilder::Limit(int64_t limit, int64_t offset) {
  limit_ = limit;
  offset_ = offset;
}

::arrow::Result<std::shared_ptr<::arrow::dataset::Scanner>> ScannerBuilder::Finish() const {
  if (offset_ < 0) {
    return ::arrow::Status::Invalid("Offset is negative");
  }
  auto builder = ::arrow::dataset::ScannerBuilder(dataset_);
  ARROW_RETURN_NOT_OK(builder.Filter(filter_));

  auto fragment_opts = std::make_shared<LanceFragmentScanOptions>();
  fragment_opts->limit = limit_;
  fragment_opts->offset = offset_;
  ARROW_RETURN_NOT_OK(builder.FragmentScanOptions(fragment_opts));

  ARROW_ASSIGN_OR_RAISE(auto scanner, builder.Finish());

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

    if (::arrow::compute::ExpressionHasFieldRefs(scanner->options()->filter)) {
      for (auto& filtered_column :
           ::arrow::compute::FieldsInExpression(scanner->options()->filter)) {
        auto filtered_col_name = ColumnNameFromFieldRef(filtered_column);
        if (std::find(std::begin(columns), std::end(columns), filtered_col_name) == columns.end()) {
          columns.emplace_back(filtered_col_name);
        }
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema.Project(columns));
    scanner->options()->dataset_schema = projected_schema->ToArrow();
    ARROW_ASSIGN_OR_RAISE(scanner->options()->filter,
                          scanner->options()->filter.Bind(*scanner->options()->dataset_schema));

    /// Only keep the top level column names to build the output schema.
    std::vector<std::string> top_names;
    for (const auto& field : projected_schema->fields()) {
      auto name = field->name();
      if (auto it = std::find(top_names.begin(), top_names.end(), name); it == top_names.end()) {
        auto dot_pos = name.find_first_of(".");
        if (dot_pos == std::string::npos) {
          top_names.emplace_back(name);
        } else {
          top_names.emplace_back(name.substr(0, dot_pos));
        }
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto project_desc,
                          ::arrow::dataset::ProjectionDescr::FromNames(
                              top_names, *scanner->options()->dataset_schema));
    scanner->options()->projected_schema = project_desc.schema;
    scanner->options()->projection = project_desc.expression;
  }

  if (limit_.has_value()) {
    scanner->options()->batch_size = offset_ + limit_.value();
    /// We need to limit the parallelism for Project to calculate LIMIT / Offset
    scanner->options()->batch_readahead = 1;
  }

  return scanner;
}

}  // namespace lance::arrow
