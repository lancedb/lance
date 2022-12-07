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

/// Partitioning.
///
/// Stored in each manifest, to describe the partitioning scheme of the dataset.
class Partitioning {
 public:
  Partitioning() = delete;

  /// Make a partitioning from partitioning schema.
  ///
  /// \param schema Partitioning schema.
  /// \return Partitioning object on success. `::arrow::Status::Invalid` if schema is empty or
  ///         any of the field is not primitive / string field.
  static ::arrow::Result<Partitioning> Make(std::shared_ptr<Schema> schema);

  /// Make partitioning from protobuf and full dataset schema.
  ///
  /// \param dataset_schema the full schema of the dataset.
  /// \param proto persisted the partitioning.
  /// \return Partitioning object on success. `::arrow::Status::Invalid` if it can not construct
  ///         partitioning object, i.e., the protobuf refers to a field that does not exist in the
  ///         provided `dataset_schema`.
  static ::arrow::Result<Partitioning> Make(const Schema& dataset_schema,
                                            const pb::Partitioning& proto);

  /// Convert to an Arrow Partitioning.
  std::shared_ptr<::arrow::dataset::Partitioning> ToArrow() const;

  /// Convert to protobuf.
  pb::Partitioning ToProto() const;

  /// Partition schema.
  const std::shared_ptr<Schema>& schema() const;

 private:
  /// Construct with partitioning schema.
  explicit Partitioning(std::shared_ptr<Schema> schema);

  /// Partitioning schema
  std::shared_ptr<Schema> schema_;
};

}  // namespace lance::format
