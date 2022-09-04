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

#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>

#include "lance/format/schema.h"
#include "lance/io/exec/base.h"
#include "lance/io/reader.h"

namespace lance::io::exec {

/// \brief Take Node
///
/// Take node takes the filtered results from child node, and uses the indices to
/// conduct random access to fetch the result of columns.
///
/// For example, "SELECT col1, col2 FROM dataet WHERE col3 = 123",
///   Take.child -> Filter.child -> Scan
///
/// If limit / offsets are presented in the statement. The Limit node will be placed
/// in front of the filter node.
///
///  Take.child -> Limit.child -> Filter.child -> Scan
class Take : public ExecNode {
 public:
  /// Factory method.
  ///
  /// \param reader An opened lance file reader.
  /// \param schema Schema for the takers.
  /// \param child the child filter node.
  /// \return a take node if success.
  static ::arrow::Result<std::unique_ptr<Take>> Make(std::shared_ptr<FileReader> reader,
                                                     std::shared_ptr<lance::format::Schema> schema,
                                                     std::unique_ptr<ExecNode> child);

  /// Returns the next batch.
  ///
  /// It reads the filtered data from child_, run FileReader::take(indices) to fetch
  /// the remaining columns, and rebuild ScanBatch, i.e., remove the indices column from the
  /// batch.
  ///
  /// \return ScanBatch if succeed.
  ::arrow::Result<ScanBatch> Next() override;

  constexpr Type type() const override { return kTake; }

  std::string ToString() const override;

 private:
  /// Constructor.
  ///
  /// \param reader An opened lance file reader.
  /// \param schema Schema for the takers.
  /// \param child the child filter node.
  Take(std::shared_ptr<FileReader> reader,
       std::shared_ptr<lance::format::Schema> schema,
       std::unique_ptr<ExecNode> child);

  std::shared_ptr<FileReader> reader_;
  std::shared_ptr<lance::format::Schema> schema_;
  std::unique_ptr<ExecNode> child_;
};

}  // namespace lance::io::exec
