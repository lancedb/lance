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
#include <arrow/filesystem/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <arrow/util/future.h>

#include <memory>

#include "lance/arrow/dataset.h"
#include "lance/format/metadata.h"
#include "lance/format/page_table.h"

namespace lance::format {
class Field;
class Schema;
class Metadata;
}  // namespace lance::format

namespace lance::io {

/// Write Manifest and Dataset version to the destination.
::arrow::Status WriteManifestWithVersion(
    const std::shared_ptr<::arrow::io::OutputStream>& destination,
    const lance::format::Manifest& manifest,
    const lance::arrow::DatasetVersion& version);

/// Lance FileWriter
class FileWriter final : public ::arrow::dataset::FileWriter {
 public:
  FileWriter(std::shared_ptr<lance::format::Schema> schema,
             std::shared_ptr<::arrow::dataset::FileWriteOptions> options,
             std::shared_ptr<::arrow::io::OutputStream> destination,
             ::arrow::fs::FileLocator destination_locator = {});

  ~FileWriter() override;

  /// Write an arrow RecordBatch to the file.
  ::arrow::Status Write(const std::shared_ptr<::arrow::RecordBatch>& batch) override;

 private:
  ::arrow::Future<> FinishInternal() override;

  ::arrow::Status WriteFooter();

  ::arrow::Status WriteArray(const std::shared_ptr<format::Field>& field,
                             const std::shared_ptr<::arrow::Array>& arr);

  /// Write Arrow Arrows with fixed length values, include:
  ///  - primitive arrays
  ///  - fixed sized binary array
  ///  - fixed sized list array
  ::arrow::Status WriteFixedLengthArray(const std::shared_ptr<format::Field>& field,
                                        const std::shared_ptr<::arrow::Array>& arr);

  ::arrow::Status WriteStructArray(const std::shared_ptr<format::Field>& field,
                                   const std::shared_ptr<::arrow::Array>& arr);
  ::arrow::Status WriteListArray(const std::shared_ptr<format::Field>& field,
                                 const std::shared_ptr<::arrow::Array>& arr);
  /// Write Arrow DictionaryArray.
  ::arrow::Status WriteDictionaryArray(const std::shared_ptr<format::Field>& field,
                                       const std::shared_ptr<::arrow::Array>& arr);

  std::shared_ptr<lance::format::Schema> lance_schema_;
  std::unique_ptr<lance::format::Metadata> metadata_;
  format::PageTable lookup_table_;
  int32_t batch_id_ = 0;
};

}  // namespace lance::io