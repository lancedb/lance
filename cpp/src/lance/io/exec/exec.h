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

#pragma once

#include <arrow/array.h>
#include <arrow/dataset/scanner.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>
#include <string>
#include <vector>

#include "lance/arrow/scan_options.h"
#include "lance/format/format.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

/// I/O execution plan.
///
///
class PlanNode {
 public:
  virtual ~PlanNode() = default;

  /// Execute the plan over a chunk.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Array>> Execute(
      std::shared_ptr<FileReader> reader, int32_t chunk_idx) = 0;

  /// Short node name.
  virtual std::string type_name() const = 0;

  /// Debug String.
  virtual std::string ToString() const = 0;

  /// Validate the plan.
  virtual ::arrow::Status Validate() const = 0;

  /// Returns True if two plans are equal.
  virtual bool Equals(const PlanNode& other) const = 0;

  /// Returns True if two plans are equal.
  virtual bool Equals(const std::shared_ptr<PlanNode>& other) const;

  bool operator==(const PlanNode& other) const;
};

/// Scan. The leaf node to actual read a page from storage.
class Scan : public PlanNode {
 public:
  Scan(std::shared_ptr<format::Field> field);

  /// Scan the full chunk.
  ::arrow::Result<std::shared_ptr<::arrow::Array>> Execute(std::shared_ptr<FileReader> reader,
                                                           int32_t chunk_idx) override;

  /// Executing Scan on a page with specified indices.
  ///
  /// \param reader a Lance FileReader
  /// \param chunk_id
  /// \param indices the indices array.
  /// \return The result array.
  ::arrow::Result<std::shared_ptr<::arrow::Array>> Execute(std::shared_ptr<FileReader> reader,
                                                           int32_t chunk_id,
                                                           std::shared_ptr<::arrow::Array> indices);

  std::string type_name() const override;

  std::string ToString() const override;

  bool Equals(const PlanNode& other) const override;

  ::arrow::Status Validate() const override;

 private:
  std::shared_ptr<format::Field> field_;
};

///
class Filter : public PlanNode {
 public:
  std::string type_name() const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Execute(std::shared_ptr<FileReader> reader,
                                                           int32_t chunk_idx) override;

 private:
  std::unique_ptr<Scan> child_;
};

/// Projection.
class Project : public PlanNode {
 public:
  /// Create Project node from schema.
  Project(const std::shared_ptr<lance::format::Schema>& schema) noexcept;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Execute(
      std::shared_ptr<FileReader> reader, int32_t chunk_idx) override;

  std::string type_name() const override;

  std::string ToString() const override;

  bool Equals(const PlanNode& other) const override;

  ::arrow::Status Validate() const override;

 private:
  std::vector<Filter> filters_;
  std::shared_ptr<lance::format::Schema> projected_schema_;
};

/// Make (and optimize) a plan tree.
::arrow::Result<std::shared_ptr<PlanNode>> Make(
    std::shared_ptr<lance::arrow::ScanOptions> scan_options);

}  // namespace lance::io::exec
