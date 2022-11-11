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
  ARROW_ASSIGN_OR_RAISE(auto batch_reader, lance::io::RecordBatchReader::Make(*this, options));
  return ::arrow::RecordBatchGenerator(std::move(batch_reader));
}

::arrow::Result<std::shared_ptr<::arrow::Schema>> LanceFragment::ReadPhysicalSchemaImpl() {
  return schema_->ToArrow();
}

::arrow::Result<std::vector<LanceFragment::FileReaderWithSchema>> LanceFragment::Open(
    const format::Schema& schema, ::arrow::internal::Executor* executor) const {
  assert(executor);
  fmt::print("Filter by schema; {}\n", schema);
  std::vector<LanceFragment::FileReaderWithSchema> readers;
  // TODO: it is very slow to open s3 files, open files in parallell.

  std::vector<::arrow::Future<FileReaderWithSchema>> futs;
  for (std::size_t i = 0; i < fragment_->data_files().size(); i++) {
    ARROW_ASSIGN_OR_RAISE(auto future,
                          executor->Submit(
                              [this](auto idx) -> ::arrow::Result<FileReaderWithSchema> {
                                auto& data_file = this->fragment_->data_files()[idx];
                                auto full_path = (fs::path(data_uri_) / data_file.path()).string();
                                ARROW_ASSIGN_OR_RAISE(auto infile, fs_->OpenInputFile(full_path))
                                ARROW_ASSIGN_OR_RAISE(auto reader,
                                                      lance::io::FileReader::Make(infile));
                                return std::make_tuple(std::move(reader), schema_);
                              },
                              i));
    futs.emplace_back(std::move(future));
  }
//  for (auto& data_file : fragment_->data_files()) {
//    //    auto data_file_schema = schema_->Project(data_file.fields());
//    auto full_path = (fs::path(data_uri_) / data_file.path()).string();
//    ARROW_ASSIGN_OR_RAISE(auto infile, fs_->OpenInputFile(full_path))
//    ARROW_ASSIGN_OR_RAISE(auto reader, lance::io::FileReader::Make(infile));
//    readers.emplace_back(std::make_tuple(std::move(reader), schema_));
//  }
  return std::move(readers);
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