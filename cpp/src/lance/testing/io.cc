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

#include "lance/testing/io.h"

#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/writer.h"
#include "lance/io/reader.h"

namespace lance::testing {

::arrow::Result<std::string> MakeTemporaryDir() {
  std::string temp = "lance-test-XXXXXX";
  auto temp_dir = mkdtemp(temp.data());
  if (temp_dir == nullptr) {
    return ::arrow::Status::IOError(strerror(errno));
  }
  return std::string(temp_dir);
}

::arrow::Result<std::shared_ptr<io::FileReader>> MakeReader(
    const std::shared_ptr<::arrow::Table>& table) {
  auto sink = ::arrow::io::BufferOutputStream::Create().ValueOrDie();
  ARROW_RETURN_NOT_OK(lance::arrow::WriteTable(*table, sink));
  auto infile = make_shared<::arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  auto reader = std::make_shared<io::FileReader>(infile);
  ARROW_RETURN_NOT_OK(reader->Open());
  return reader;
}

::arrow::Result<std::shared_ptr<::arrow::dataset::Dataset>> MakeDataset(
    const std::shared_ptr<::arrow::Table>& table) {
  auto sink = ::arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto dataset = ::arrow::dataset::InMemoryDataset(table);
  ARROW_ASSIGN_OR_RAISE(auto scanner_builder, dataset.NewScan());
  ARROW_ASSIGN_OR_RAISE(auto scanner, scanner_builder->Finish());

  auto format = lance::arrow::LanceFileFormat::Make();
  ::arrow::dataset::FileSystemDatasetWriteOptions write_options;
  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  write_options.file_write_options = format->DefaultWriteOptions();
  write_options.filesystem = fs;
  write_options.base_dir = MakeTemporaryDir().ValueOrDie();
  write_options.basename_template = "part{i}.lance";

  ARROW_RETURN_NOT_OK(::arrow::dataset::FileSystemDataset::Write(write_options, scanner));

  ::arrow::fs::FileSelector selector;
  selector.base_dir = write_options.base_dir;
  ARROW_ASSIGN_OR_RAISE(auto factory,
                        ::arrow::dataset::FileSystemDatasetFactory::Make(
                            fs, selector, format, ::arrow::dataset::FileSystemFactoryOptions()));
  return factory->Finish();
}

TableScan::TableScan(const ::arrow::Table& table, int64_t batch_size)
    : reader_(new ::arrow::TableBatchReader(table)) {
  reader_->set_chunksize(batch_size);
};

std::unique_ptr<io::exec::ExecNode> TableScan::MakeEmpty() {
  return std::unique_ptr<io::exec::ExecNode>(new TableScan());
}

::arrow::Result<io::exec::ScanBatch> TableScan::Next() {
  if (!reader_) {
    return io::exec::ScanBatch{};
  }

  std::shared_ptr<::arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(reader_->ReadNext(&batch));
  return io::exec::ScanBatch{
      batch,
      0,
  };
}

}  // namespace lance::testing
