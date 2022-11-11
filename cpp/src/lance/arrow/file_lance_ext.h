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

#include <vector>

#include "lance/format/data_fragment.h"
#include "lance/io/exec/counter.h"

namespace lance::arrow {

/// Lance FragmentScanOptions.
///
/// Extra lance scan options.
class LanceFragmentScanOptions : public ::arrow::dataset::FragmentScanOptions {
 public:
  LanceFragmentScanOptions() = default;

  [[nodiscard]] std::string type_name() const override;

  /// Singleton of the Limit object shared between one Scan run.
  std::shared_ptr<lance::io::exec::Counter> counter;
};

/// Check if the fragment scan option is LanceFragmentScanOptions.
bool IsLanceFragmentScanOptions(const ::arrow::dataset::FragmentScanOptions& fso);

}  // namespace lance::arrow