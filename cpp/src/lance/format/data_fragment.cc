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

#include "lance/format/data_fragment.h"

#include <range/v3/view.hpp>

using namespace ranges::views;

namespace lance::format {

DataFile::DataFile(const pb::DataFile& pb)
    : path_(pb.path()), fields_(pb.fields().begin(), pb.fields().end()) {}

DataFile::DataFile(std::string path, const std::vector<int32_t>& fields)
    : path_(path), fields_(std::begin(fields), std::end(fields)) {}

const std::string& DataFile::path() const { return path_; }

const std::vector<int32_t>& DataFile::fields() const { return fields_; }

pb::DataFile DataFile::ToProto() const {
  auto proto = lance::format::pb::DataFile();
  proto.set_path(path_);
  for (auto field : fields_) {
    proto.add_fields(field);
  }
  return proto;
}

DataFragment::DataFragment(const pb::DataFragment& pb) {
  files_ = pb.files() | transform([](auto& pb_file) { return DataFile(pb_file); }) |
           ranges::to<decltype(files_)>();
}

DataFragment::DataFragment(const DataFile& data_file)
    : files_(std::vector<DataFile>({data_file})) {}

DataFragment::DataFragment(std::vector<DataFile> data_files) : files_(std::move(data_files)) {}

DataFragment::DataFragment(const DataFragment& other)
    : files_(other.files_.begin(), other.files_.end()) {}

const std::vector<DataFile>& DataFragment::data_files() const { return files_; }

std::vector<DataFile>& DataFragment::data_files() { return files_; }

pb::DataFragment DataFragment::ToProto() const {
  auto proto = pb::DataFragment();
  for (auto& file : files_) {
    auto pb_file = proto.add_files();
    *pb_file = file.ToProto();
  }
  return proto;
}

}  // namespace lance::format