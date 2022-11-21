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

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>
#include <optional>

#include "lance/arrow/fragment.h"
#include "lance/format/schema.h"
#include "lance/io/exec/base.h"

namespace lance::io {
class FileReader;
}

namespace lance::arrow {
class LanceFragment;
}

namespace lance::io::exec {

/// \brief Projection over dataset.
///
class Project : ExecNode {
 public:
  Project() = delete;

  /// Make a Project from the fragment and scan options.
  ///
  /// \param fragment Lance fragment
  /// \param scan_options Arrow scan options.
  /// \return Project if success. Returns the error status otherwise.
  static ::arrow::Result<std::unique_ptr<Project>> Make(
      const lance::arrow::LanceFragment& fragment,
      std::shared_ptr<::arrow::dataset::ScanOptions> scan_options);

  /// Read the next batch. Returns nullptr if EOF.
  ::arrow::Result<ScanBatch> Next() override;

  /// ExecNode type.
  constexpr Type type() const override { return kProject; }

  /// Debug String.
  std::string ToString() const override;

  /// Returns the projected schema.
  const std::shared_ptr<lance::format::Schema>& schema() const { return projected_schema_; }

 private:
  Project(std::unique_ptr<ExecNode> child, std::shared_ptr<lance::format::Schema> projected_schema);

  std::unique_ptr<ExecNode> child_;
  std::shared_ptr<lance::format::Schema> projected_schema_;
};

}  // namespace lance::io::exec