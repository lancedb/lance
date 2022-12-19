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
#include <google/protobuf/util/time_util.h>

#include <memory>

#include "lance/format/schema.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;

namespace lance::format {

Manifest::Manifest(std::shared_ptr<Schema> schema) : schema_(std::move(schema)), version_(1) {}

Manifest::Manifest(std::shared_ptr<Schema> schema,
                   std::vector<std::shared_ptr<DataFragment>> fragments,
                   uint64_t version)
    : schema_(std::move(schema)), version_(version), fragments_(std::move(fragments)) {}

Manifest::Manifest(Manifest&& other) noexcept
    : schema_(std::move(other.schema_)),
      version_(other.version_),
      fragments_(std::move(other.fragments_)) {}

Manifest::Manifest(const Manifest& other) noexcept
    : schema_(other.schema_), version_(other.version_), fragments_(other.fragments()) {}

Manifest::Manifest(const lance::format::pb::Manifest& pb)
    : schema_(std::make_unique<Schema>(pb.fields(), pb.metadata())),
      version_(pb.version()),
      version_aux_data_position_(pb.version_aux_data()) {
  for (auto& pb_fragment : pb.fragments()) {
    fragments_.emplace_back(std::make_shared<DataFragment>(pb_fragment));
  }
}

::arrow::Result<std::shared_ptr<Manifest>> Manifest::Parse(
    const std::shared_ptr<::arrow::io::RandomAccessFile>& in, int64_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto pb, io::ParseProto<pb::Manifest>(in, offset));
  return std::shared_ptr<Manifest>(new Manifest(pb));
}

::arrow::Result<std::shared_ptr<Manifest>> Manifest::Parse(
    const std::shared_ptr<::arrow::Buffer>& buffer) {
  ARROW_ASSIGN_OR_RAISE(auto pb, io::ParseProto<pb::Manifest>(buffer));
  return std::shared_ptr<Manifest>(new Manifest(pb));
}

pb::Manifest Manifest::ToProto() const {
  lance::format::pb::Manifest pb;
  for (auto field : schema_->ToProto()) {
    auto pb_field = pb.add_fields();
    *pb_field = field;
  }
  pb.mutable_metadata()->insert(schema_->metadata().begin(), schema_->metadata().end());
  pb.set_version(version_);
  pb.set_version_aux_data(version_aux_data_position());

  for (const auto& fragment : fragments_) {
    auto pb_fragment = pb.add_fragments();
    *pb_fragment = fragment->ToProto();
  }
  return pb;
}

const std::shared_ptr<Schema>& Manifest::schema() const { return schema_; }

uint64_t Manifest::version() const { return version_; }

uint64_t Manifest::version_aux_data_position() const {
  return version_aux_data_position_.value_or(0);
}

void Manifest::SetVersionAuxDataPosition(uint64_t pos) { version_aux_data_position_ = pos; }

const std::vector<std::shared_ptr<DataFragment>>& Manifest::fragments() const { return fragments_; }

void Manifest::AppendFragments(const std::vector<std::shared_ptr<DataFragment>>& fragments) {
  fragments_.insert(std::end(fragments_), std::begin(fragments), std::end(fragments));
}

}  // namespace lance::format
