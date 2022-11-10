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

#include "lance/arrow/fragment.h"

#include <arrow/array/concatenate.h>
#include <arrow/chunked_array.h>
#include <arrow/table.h>

#include <filesystem>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/utils.h"
#include "lance/format/data_fragment.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"
#include "lance/io/record_batch_reader.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;

namespace lance::arrow {

LanceFragment::LanceFragment(std::shared_ptr<::arrow::fs::FileSystem> fs,
                             std::string data_dir,
                             std::shared_ptr<lance::format::DataFragment> fragment,
                             std::shared_ptr<format::Schema> schema)
    : fs_(std::move(fs)),
      data_uri_(std::move(data_dir)),
      fragment_(std::move(fragment)),
      schema_(std::move(schema)) {}

::arrow::Result<::arrow::RecordBatchGenerator> LanceFragment::ScanBatchesAsync(
    const std::shared_ptr<::arrow::dataset::ScanOptions>& options) {
  for (std::size_t i = 0; i < fragment_->data_files().size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto reader, OpenReader(i));
    auto batch_reader = lance::io::RecordBatchReader(
        std::move(reader), options, ::arrow::internal::GetCpuThreadPool());
    ARROW_RETURN_NOT_OK(batch_reader.Open());
    return ::arrow::RecordBatchGenerator(std::move(batch_reader));
  }
  return ::arrow::Status::IOError("Lance Fragment has zero file");
}

::arrow::Result<std::shared_ptr<::arrow::Schema>> LanceFragment::ReadPhysicalSchemaImpl() {
  return schema_->ToArrow();
}

::arrow::Result<int64_t> LanceFragment::FastCountRow() const {
  assert(!fragment_->data_files().empty());
  ARROW_ASSIGN_OR_RAISE(auto reader, OpenReader(0));
  return reader->length();
}

::arrow::Result<std::unique_ptr<lance::io::FileReader>> LanceFragment::OpenReader(
    std::size_t data_file_idx) const {
  assert(data_file_idx < fragment_->data_files().size());
  auto data_file = fragment_->data_files()[data_file_idx];
  auto full_path = (fs::path(data_uri_) / data_file.path()).string();
  ARROW_ASSIGN_OR_RAISE(auto infile, fs_->OpenInputFile(full_path));
  return lance::io::FileReader::Make(infile);
}

}  // namespace lance::arrow