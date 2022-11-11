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

#include <arrow/dataset/api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "lance/format/data_fragment.h"
#include "lance/format/schema.h"

namespace lance::io {
class FileReader;
}

namespace lance::arrow {

/// \brief LanceFragment. a fragment of data.
///
/// Physically, it contains a set of lance files, each of which represents subset of
/// columns of the same rows in dataset.
///
class LanceFragment : public ::arrow::dataset::Fragment {
 public:
  using FileReaderWithSchema =
      std::tuple<std::shared_ptr<lance::io::FileReader>, std::shared_ptr<format::Schema>>;

  /// Constructor
  ///
  /// \param fs a file system instance to conduct IOs.
  /// \param data_dir the base directory to store data.
  /// \param fragment data fragment, the metadata of the fragment.
  /// \param schema the schema of the Fragment.
  LanceFragment(std::shared_ptr<::arrow::fs::FileSystem> fs,
                std::string data_dir,
                std::shared_ptr<lance::format::DataFragment> fragment,
                std::shared_ptr<format::Schema> schema);

  /// Destructor.
  ~LanceFragment() override = default;

  /// Scan Batches.
  ///
  /// \param options ScanOptions.
  /// \return `RecordBatchGenerator` if succeed. Otherwise, the error message.
  ::arrow::Result<::arrow::RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<::arrow::dataset::ScanOptions>& options) override;

  std::string type_name() const override { return "lance"; }

  /// Access data fragment.
  const std::shared_ptr<lance::format::DataFragment>& data_fragment() const { return fragment_; }

  ::arrow::Result<std::vector<FileReaderWithSchema>> Open(
      const format::Schema& schema,
      ::arrow::internal::Executor* executor = ::arrow::internal::GetCpuThreadPool()) const;

  /// Data files.
  const std::vector<lance::format::DataFile>& data_files() const { return fragment_->data_files(); }

  /// Dataset schema.
  const std::shared_ptr<format::Schema>& schema() const { return schema_; }

 protected:
  ::arrow::Result<std::shared_ptr<::arrow::Schema>> ReadPhysicalSchemaImpl() override;

 private:
  /// Fast path `CountRow()`, it only reads the metadata of one data file.
  ::arrow::Result<int64_t> FastCountRow() const;

  /// Open file reader on a data file.
  ///
  /// \param data_file_idx the index of data file in `fragment_->data_files()`.
  /// \return an Opened lance FileReader if success.
  ::arrow::Result<std::unique_ptr<lance::io::FileReader>> OpenReader(
      std::size_t data_file_idx) const;

  std::shared_ptr<::arrow::fs::FileSystem> fs_;
  std::string data_uri_;
  std::shared_ptr<lance::format::DataFragment> fragment_;
  std::shared_ptr<format::Schema> schema_;
};

}  // namespace lance::arrow
