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

#pragma once

#include <arrow/dataset/scanner.h>

#include <memory>
#include <optional>

namespace lance::arrow {

/// Extends Arrow ScanOptions with Lance additionally options.
///
/// Add limit and offset pushdown supports.
class ScanOptions {
 public:
  ScanOptions(std::shared_ptr<::arrow::dataset::ScanOptions> arrow_opts,
              std::optional<int64_t> limit = std::nullopt,
              std::optional<int64_t> offset = std::nullopt);

 private:
  std::shared_ptr<::arrow::dataset::ScanOptions> arrow_options_;

  std::optional<int64_t> limit_;
  std::optional<int64_t> offset_;
};

}  // namespace lance::arrow
