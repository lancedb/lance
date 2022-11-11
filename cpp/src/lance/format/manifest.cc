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

#include "lance/format/manifest.h"

#include <arrow/result.h>

#include <memory>

#include "lance/format/schema.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;

namespace lance::format {

Manifest::Manifest(std::shared_ptr<Schema> schema) : schema_(std::move(schema)), version_(1) {}

Manifest::Manifest(Manifest&& other) noexcept
    : schema_(std::move(other.schema_)),
      version_(other.version_),
      fragments_(std::move(other.fragments_)) {}

Manifest::Manifest(const Manifest& other) noexcept
    : schema_(other.schema_), version_(other.version_), fragments_(other.fragments()) {}

Manifest::Manifest(const lance::format::pb::Manifest& pb)
    : schema_(std::make_unique<Schema>(pb.fields(), pb.metadata())), version_(pb.version()) {
  for (auto& pb_fragment : pb.fragments()) {
    fragments_.emplace_back(std::make_shared<DataFragment>(pb_fragment));
  }
}

::arrow::Result<std::shared_ptr<Manifest>> Manifest::Parse(
    std::shared_ptr<::arrow::io::RandomAccessFile> in, int64_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto pb, io::ParseProto<pb::Manifest>(in, offset));
  return std::shared_ptr<Manifest>(new Manifest(pb));
}

::arrow::Result<int64_t> Manifest::Write(std::shared_ptr<::arrow::io::OutputStream> out) const {
  lance::format::pb::Manifest pb;
  for (auto field : schema_->ToProto()) {
    auto pb_field = pb.add_fields();
    *pb_field = field;
  }
  for (const auto& [key, value] : schema_->metadata()) {
    (*pb.mutable_metadata())[key] = value;
  }
  pb.set_version(version_);
  for (const auto& fragment : fragments_) {
    auto pb_fragment = pb.add_fragments();
    *pb_fragment = fragment->ToProto();
  }
  return io::WriteProto(out, pb);
}

std::shared_ptr<Manifest> Manifest::BumpVersion(bool overwrite) {
  auto new_manifest = std::make_shared<Manifest>(*this);
  new_manifest->version_++;
  if (overwrite) {
    new_manifest->fragments_.clear();
  }
  return new_manifest;
}

const std::shared_ptr<Schema>& Manifest::schema() const { return schema_; }

uint64_t Manifest::version() const { return version_; }

const std::vector<std::shared_ptr<DataFragment>>& Manifest::fragments() const { return fragments_; }

void Manifest::AppendFragments(const std::vector<std::shared_ptr<DataFragment>>& fragments) {
  fragments_.insert(std::end(fragments_), std::begin(fragments), std::end(fragments));
}

arrow::DatasetVersion Manifest::GetDatasetVersion() const {
  return arrow::DatasetVersion{version_};
}

}  // namespace lance::format
