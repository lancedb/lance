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

#include <arrow/result.h>

#include <memory>
#include <string>
#include <vector>

#include "lance/io/reader.h"

namespace lance::format {

/// DataFile is a lance file contains the data.
class DataFile final {
 public:
  DataFile(const std::string& path, const std::vector<int32_t>& fields);

  ::arrow::Result<std::shared_ptr<lance::io::FileReader>> Open(const std::string& base_uri);

  const std::string& path() const;

 private:
  /// Relative path of a lance file.
  std::string path_;

  /// The ids of the field in this file.
  std::vector<int32_t> fields_;
};

}  // namespace lance::format
