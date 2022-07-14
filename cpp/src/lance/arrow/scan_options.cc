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

#include "lance/arrow/scan_options.h"

#include <arrow/dataset/scanner.h>
#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "lance/format/schema.h"

namespace lance::arrow {

ScanOptions::ScanOptions(std::shared_ptr<lance::format::Schema> schema,
                         std::shared_ptr<::arrow::dataset::ScanOptions> arrow_opts,
                         std::optional<int64_t> limit,
                         std::optional<int64_t> offset)
    : schema_(schema), arrow_options_(arrow_opts), limit_(limit), offset_(offset) {}

const std::shared_ptr<lance::format::Schema>& ScanOptions::schema() const { return schema_; }

const std::shared_ptr<::arrow::dataset::ScanOptions>& ScanOptions::arrow_options() const {
  return arrow_options_;
}

std::string ScanOptions::ToString() const {
  return fmt::format("ScanOptions(dataset={},\n, project={}\n, filter={}\n, limit={}\noffset={})",
                     schema_->ToString(),
                     arrow_options_->projected_schema->ToString(),
                     arrow_options_->filter.ToString(),
                     limit_.value_or(-1),
                     offset_.value_or(-1));
}

}  // namespace lance::arrow