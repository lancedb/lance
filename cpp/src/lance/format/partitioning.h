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

#include <arrow/dataset/partition.h>
#include <arrow/result.h>

#include <memory>

namespace lance::format {

namespace pb {
class Partitioning;
}

class Schema;

class Partitioning {
 public:
  Partitioning() = delete;

  static ::arrow::Result<Partitioning> Make(const Schema& dataset_schema,
                                            const pb::Partitioning& proto);

  /// Convert to an Arrow Partitioning.
  std::shared_ptr<::arrow::dataset::Partitioning> ToArrow();

  pb::Partitioning ToProto() const;

  /// Partition schema.
  const std::shared_ptr<Schema>& schema() const;

 private:
  Partitioning(std::shared_ptr<Schema> schema);

  /// Partitioning schema
  std::shared_ptr<Schema> schema_;
};

}  // namespace lance::format
