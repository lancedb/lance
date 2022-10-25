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

#include <cstdint>
#include <string>
#include <vector>

#include "lance/format/format.pb.h"
#include "lance/format/format.h"

namespace lance::format {

class DataFile : public ConvertToProto<pb::DataFile> {
 public:
  explicit DataFile(const format::pb::DataFile& pb);

  DataFile(std::string path, const std::vector<int32_t>& fields);

  /// Get the relative path of the data
  const std::string& path() const;

  const std::vector<int32_t>& fields() const;

  pb::DataFile ToProto() const override;

 private:
  std::string path_;
  std::vector<int32_t> fields_;
};

class DataFragment : public ConvertToProto<pb::DataFragment> {
 public:
  explicit DataFragment(const format::pb::DataFragment& pb);

  explicit DataFragment(const DataFile& data_file);

  DataFragment(std::vector<DataFile> data_files);

  const std::vector<DataFile>& data_files() const;

  pb::DataFragment ToProto() const override;

 private:
  std::vector<DataFile> files_;
};

}  // namespace lance::format
