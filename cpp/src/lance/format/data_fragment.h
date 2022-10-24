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

#include <arrow/dataset/api.h>
#include <arrow/status.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "lance/format/data_file.h"
#include "lance/format/format.pb.h"

namespace lance::format {

/// Data Fragment contains a group of files that represent the different columns
/// of the same rows.
class DataFragment {
 public:
  /// Construct from protobuf.
  DataFragment(const pb::DataFragment& pb);

  /// Fragment ID.
  uint64_t id() const;

  std::shared_ptr<::arrow::dataset::FileFragment> ToArrow() const;

  // Run validation.
  ::arrow::Status Validate() const;

 private:
  /// Fragment ID
  uint64_t id_;

  std::vector<DataFile> files_;
};

}  // namespace lance::format
