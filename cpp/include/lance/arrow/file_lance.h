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

#include <arrow/dataset/file_base.h>

#include <mutex>
#include <optional>
#include <string>

namespace lance::arrow {

/// Lance File Format
class LanceFileFormat : public ::arrow::dataset::FileFormat {
 public:
  LanceFileFormat();

  ~LanceFileFormat() override;

  static std::shared_ptr<LanceFileFormat> Make();

  std::string type_name() const override;

  bool Equals(const FileFormat &other) const override;

  ::arrow::Result<bool> IsSupported(const ::arrow::dataset::FileSource &source) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Schema>> Inspect(
      const ::arrow::dataset::FileSource &source) const override;

  ::arrow::Future<std::optional<int64_t>> CountRows(
      const std::shared_ptr<::arrow::dataset::FileFragment> &file,
      ::arrow::compute::Expression predicate,
      const std::shared_ptr<::arrow::dataset::ScanOptions> &options) override;

  ::arrow::Result<::arrow::RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<::arrow::dataset::ScanOptions> &options,
      const std::shared_ptr<::arrow::dataset::FileFragment> &file) const override;

  ::arrow::Result<std::shared_ptr<::arrow::dataset::FileWriter>> MakeWriter(
      std::shared_ptr<::arrow::io::OutputStream> destination,
      std::shared_ptr<::arrow::Schema> schema,
      std::shared_ptr<::arrow::dataset::FileWriteOptions> options,
      ::arrow::fs::FileLocator destination_locator) const override;

  std::shared_ptr<::arrow::dataset::FileWriteOptions> DefaultWriteOptions() override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  std::mutex lock_;
};

class FileWriteOptions : public ::arrow::dataset::FileWriteOptions {
 public:
  FileWriteOptions();

  ~FileWriteOptions() override = default;

  ::arrow::Status Validate() const;

  int32_t batch_size = 1024;
};

}  // namespace lance::arrow
