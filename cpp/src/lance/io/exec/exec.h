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

namespace lance::io::exec {

/// I/O execution plan.
///
///
class PlanNode {
 public:
  virtual ~PlanNode() = default;

  virtual ::arrow::Result<::arrow::Array> Execute() = 0;

  /// Short node name.
  virtual std::string type_name() const = 0;

  /// Debug String.
  virtual std::string ToString() const = 0;

  /// Validate the plan.
  virtual ::arrow::Status Validate() const {
    return ::arrow::Status::OK();
  }

  /// Returns True if two plans are the same.
  virtual bool Equals(const PlanNode& other) const = 0;

  /// Returns True if two plans are the same.
  virtual bool Equals(const std::shared_ptr<PlanNode>& other) const;
};

class Filter : public PlanNode {
 public:
  std::string type_name() const override;

  ::arrow::Result<::arrow::Array> Execute() override;
};

/// Scan node.
class Scan : public PlanNode {
 public:
  ::arrow::Result<::arrow::Array> Execute() override;

  std::string type_name() const override;

  std::string ToString() const override;

  bool Equals(const PlanNode& other) const override;

 private:
};

/// Make (and optimize) a plan tree.
::arrow::Result<std::shared_ptr<PlanNode>> Make(
    std::shared_ptr<::arrow::dataset::ScanOptions> scan_options);

}  // namespace lance::io::exec
