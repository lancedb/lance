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

#include "lance/io/exec/exec.h"

#include <arrow/dataset/scanner.h>
#include <fmt/format.h>

#include <string>

#include "lance/format/schema.h"

namespace lance::io::exec {

bool PlanNode::Equals(const std::shared_ptr<PlanNode>& other) const {
  return other && Equals(*other);
}

bool PlanNode::operator==(const PlanNode& other) const { return Equals(other); }

// -------- Scan ----------

Scan::Scan(std::shared_ptr<format::Field> field) : field_(field) {}

std::string Scan::type_name() const { return "Scan"; }

::arrow::Result<std::shared_ptr<::arrow::Array>> Scan::Execute(std::shared_ptr<FileReader> reader,
                                                               int32_t chunk_idx) {
  ARROW_RETURN_NOT_OK(Validate());
  return reader->GetArray(field_, chunk_idx);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> Scan::Execute(
    std::shared_ptr<FileReader> reader, int32_t chunk_id, std::shared_ptr<::arrow::Array> indices) {
  ARROW_RETURN_NOT_OK(Validate());
  return reader->GetArray(field_, chunk_id, indices);
}

std::string Scan::ToString() const {
  return fmt::format("Scan({}, id={})", field_->name(), field_->id());
}

bool Scan::Equals(const PlanNode& other) const {
  if (type_name() != other.type_name()) {
    return false;
  }
  auto other_scan = dynamic_cast<const Scan&>(other);
  if (other_scan.field_ && field_->Equals(other_scan.field_)) {
    return true;
  }
  return false;
}

::arrow::Status Scan::Validate() const {
  if (!field_) {
    return ::arrow::Status::Invalid("Scan() does not bind to a field");
  }
  return ::arrow::Status::OK();
}

//----- Project

std::string Project::type_name() const { return "Project"; }

std::string Project::ToString() const { return fmt::format("Project(children=)"); }

::arrow::Status Project::Validate() const {
  if (!children_.empty()) {
    return ::arrow::Status::Invalid("Invalid Project: children is empty");
  }
  return ::arrow::Status::OK();
}

bool Project::Equals(const PlanNode& other) const {
  if (type_name() != other.type_name()) {
    return false;
  }
  const auto& other_proj = dynamic_cast<const Project&>(other);
  return children_ == other_proj.children_ && filters_ == other_proj.filters_;
}

::arrow::Result<std::shared_ptr<PlanNode>> Make(
    std::shared_ptr<lance::arrow::ScanOptions> scan_options) {
  fmt::print("Scan Options: {}\n", scan_options->ToString());
  return ::arrow::Status::NotImplemented("");
}

}  // namespace lance::io::exec
