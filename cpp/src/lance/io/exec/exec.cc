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

#include "exec.h"

#include <arrow/dataset/scanner.h>

#include <string>

#include "fmt/format.h"

namespace lance::io::exec {

bool PlanNode::Equals(const std::shared_ptr<PlanNode>& other) const {
  return other && Equals(*other);
}

class LimitOffset : PlanNode {};

class Project : PlanNode {};

std::string Scan::type_name() const {
  return "Scan";
}

::arrow::Result<::arrow::Array> Scan::Execute() {
  return ::arrow::Status::NotImplemented("Scan::Execute not implemented yet");
}

std::string Scan::ToString() const {
  return "Scan()";
}

bool Scan::Equals(const PlanNode& other) const {
  return type_name() == other.type_name();
}

::arrow::Result<std::shared_ptr<PlanNode>> Make(
    std::shared_ptr<::arrow::dataset::ScanOptions> scan_options) {
  fmt::print("Scan Options: dataset={}\nproject={}\nfilter={}\n",
             scan_options->dataset_schema->ToString(),
             scan_options->projected_schema ? scan_options->projected_schema->ToString() : "{}",
             scan_options->filter.ToString());
  return std::make_shared<Scan>();
}

}  // namespace lance::io::exec
