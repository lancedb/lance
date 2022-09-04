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

#include "lance/io/exec/base.h"


namespace lance::io {
class FileReader;
}

namespace lance::io::exec {

/// \brief Projection over dataset.
///
class Project : ExecNode {
 public:
  Project() = delete;

  /// Make a Project from the full dataset schema and scan options.
  ///
  /// \param reader file reader.
  /// \param schema dataset schema.
  /// \param scan_options Arrow scan options.
  /// \return Project if success. Returns the error status otherwise.
  ///
  static ::arrow::Result<std::unique_ptr<Project>> Make(
      std::shared_ptr<FileReader> reader,
      std::shared_ptr<::arrow::dataset::ScanOptions> scan_options);

  ::arrow::Result<ScanBatch> Next() override;

  constexpr Type type() const override { return kProject; }

  std::string ToString() const override;

 private:
  Project(std::unique_ptr<ExecNode> child);

  std::unique_ptr<ExecNode> child_;
};

}  // namespace lance::io::exec