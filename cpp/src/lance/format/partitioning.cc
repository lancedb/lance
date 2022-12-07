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

#include "lance/format/partitioning.h"

#include <cstdint>
#include <vector>

#include "lance/format/format.pb.h"
#include "lance/format/schema.h"

namespace lance::format {

Partitioning::Partitioning(std::shared_ptr<Schema> schema) : schema_(std::move(schema)) {}

::arrow::Result<Partitioning> Partitioning::Make(const Schema& dataset_schema,
                                                 const pb::Partitioning& proto) {
  std::vector<Schema::FieldIdType> field_ids;
  for (auto& fid : proto.field_ids()) {
    field_ids.emplace_back(fid);
  }
  ARROW_ASSIGN_OR_RAISE(auto schema, dataset_schema.Project(field_ids));
  return Partitioning(schema);
}

const std::shared_ptr<Schema>& Partitioning::schema() const { return schema_; }

}  // namespace lance::format