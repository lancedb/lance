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

#pragma once

#include <arrow/dataset/file_base.h>

#include <cstdint>
#include <optional>

namespace lance::arrow {

class LanceFragmentScanOptions : public ::arrow::dataset::FragmentScanOptions {
 public:
  LanceFragmentScanOptions() = default;

  std::string type_name() const override;

  /// Support limit / offset pushdown
  std::optional<int64_t> limit;
  int64_t offset = 0;
};

/// Check if the fragment scan option is LanceFragmentScanOptions.
bool IsLanceFragmentScanOptions(const ::arrow::dataset::FragmentScanOptions& fso);

}  // namespace lance::arrow