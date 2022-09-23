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

#include <arrow/dataset/api.h>
#include <arrow/result.h>
#include <arrow/table.h>

#include <memory>
#include <string>

#include "lance/io/exec/base.h"
#include "lance/io/reader.h"

namespace lance::testing {

/// Make temporary directory and returns the directory path.
::arrow::Result<std::string> MakeTemporaryDir();

/// Make lance::io::FileReader from an Arrow Table.
::arrow::Result<std::shared_ptr<lance::io::FileReader>> MakeReader(
    const std::shared_ptr<::arrow::Table>& table);

/// Make a FileSystem Dataset from the table.
///
/// \param table The table to write
/// \param partitions the column names of partitioning.
/// \return a FileSystem Dataset with lance format.
::arrow::Result<std::shared_ptr<::arrow::dataset::Dataset>> MakeDataset(
    const std::shared_ptr<::arrow::Table>& table,
    const std::vector<std::string>& partitions = {});

/// A ExecNode that scans a Table in memory.
///
/// This node can be used in test without creating files.
class TableScan : lance::io::exec::ExecNode {
 public:
  TableScan() = default;

  TableScan(const ::arrow::Table& table, int64_t batch_size);

  TableScan(TableScan&&) = default;

  static std::unique_ptr<io::exec::ExecNode> Make(const ::arrow::Table& table,
                                                  int64_t batch_size = 1024) {
    return std::unique_ptr<io::exec::ExecNode>(new TableScan(table, batch_size));
  }

  /// Make an empty table scanner.
  static std::unique_ptr<io::exec::ExecNode> MakeEmpty();

  ::arrow::Result<io::exec::ScanBatch> Next() override;

  constexpr Type type() const override { return kTableScan; }

  std::string ToString() const override { return "Dummy"; }

 private:
  std::unique_ptr<::arrow::TableBatchReader> reader_;
};

}  // namespace lance::testing
