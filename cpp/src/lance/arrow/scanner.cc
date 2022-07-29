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

#include "lance/arrow/file_lance.h"
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

  auto options = std::make_shared<::arrow::dataset::ScanOptions>();
  options->dataset_schema = dataset_->schema();
  options->filter = filter_;

  if (columns_.has_value()) {
    auto schema = lance::format::Schema(dataset_->schema());
    ARROW_ASSIGN_OR_RAISE(auto projected_schema, schema.Project(columns_.value()));
    options->projected_schema = projected_schema->ToArrow();
  }

  // Limit / Offset pushdown
  if (limit_.has_value()) {
    options->batch_size = limit_.value() + offset_;
    options->use_threads = false;
    options->batch_readahead = 1;
  }

  auto builder = ::arrow::dataset::ScannerBuilder(dataset_, options);
  return builder.Finish();
}

}  // namespace lance::arrow
