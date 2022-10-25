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

Manifest::Manifest(Manifest&& other) noexcept : schema_(std::move(other.schema_)) {}

Manifest::Manifest(const lance::format::pb::Manifest& pb)
    : schema_(std::make_unique<Schema>(pb.fields(), pb.metadata())), version_(pb.version()) {
  for (auto& pb_fragment : pb.fragments()) {
    fragments_.emplace_back(std::make_shared<lance::arrow::LanceFragment>(pb_fragment));
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
    pb.mutable_fragments()->Add(fragment->ToProto());
  }
  fmt::print("Write pb fragments: {}\n", pb.fragments_size());
  return io::WriteProto(out, pb);
}

std::shared_ptr<Manifest> Manifest::BumpVersion() const {
  auto new_version = std::shared_ptr<Manifest>(new Manifest(schema_));
  new_version->version_ = version_ + 1;
  return new_version;
}

const Schema& Manifest::schema() const { return *schema_; }

uint64_t Manifest::version() const { return version_; }

const std::vector<std::shared_ptr<lance::arrow::LanceFragment>>& Manifest::fragments() const {
  return fragments_;
}

void Manifest::AppendFragments(
    const std::vector<std::shared_ptr<arrow::LanceFragment>>& fragments) {
  fragments_.insert(fragments_.end(), std::begin(fragments), std::end(fragments));
}

}  // namespace lance::format
