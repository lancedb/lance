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
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/utils.h"
#include "lance/format/data_fragment.h"
#include "lance/format/manifest.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"
#include "lance/io/record_batch_reader.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;

namespace lance::arrow {

::arrow::Result<std::shared_ptr<LanceFragment>> LanceFragment::Make(
    const ::arrow::dataset::FileFragment& file_fragment,
    std::shared_ptr<format::Manifest> manifest) {
  auto field_ids = manifest->schema()->GetFieldIds();
  auto data_fragment = std::make_shared<format::DataFragment>(
      format::DataFile(file_fragment.source().path(), field_ids));
  return std::make_shared<LanceFragment>(
      file_fragment.source().filesystem(), "", std::move(data_fragment), std::move(manifest));
}

LanceFragment::LanceFragment(std::shared_ptr<::arrow::fs::FileSystem> fs,
                             std::string data_dir,
                             std::shared_ptr<lance::format::DataFragment> fragment,
                             std::shared_ptr<lance::format::Manifest> manifest)
    : fs_(std::move(fs)),
      data_uri_(std::move(data_dir)),
      fragment_(std::move(fragment)),
      manifest_(std::move(manifest)) {}

::arrow::Result<::arrow::RecordBatchGenerator> LanceFragment::ScanBatchesAsync(
    const std::shared_ptr<::arrow::dataset::ScanOptions>& options) {
  ARROW_ASSIGN_OR_RAISE(auto batch_reader, lance::io::RecordBatchReader::Make(*this, options));
  return ::arrow::RecordBatchGenerator(std::move(batch_reader));
}

::arrow::Result<std::shared_ptr<::arrow::Schema>> LanceFragment::ReadPhysicalSchemaImpl() {
  return schema()->ToArrow();
}

::arrow::Result<std::vector<LanceFragment::FileReaderWithSchema>> LanceFragment::Open(
    const format::Schema& schema, ::arrow::internal::Executor* executor) const {
  assert(executor);

  std::vector<::arrow::Future<FileReaderWithSchema>> futs;
  for (std::size_t i = 0; i < fragment_->data_files().size(); i++) {
    ARROW_ASSIGN_OR_RAISE(
        auto future,
        executor->Submit(
            [this, &schema](auto idx) -> ::arrow::Result<FileReaderWithSchema> {
              auto& data_file = this->fragment_->data_files()[idx];
              ARROW_ASSIGN_OR_RAISE(auto data_file_schema,
                                    this->schema()->Project(data_file.fields()));
              ARROW_ASSIGN_OR_RAISE(auto intersection, schema.Intersection(*data_file_schema));
              if (intersection->fields().empty()) {
                return std::make_tuple(nullptr, nullptr);
              }
              auto full_path = (fs::path(data_uri_) / data_file.path()).string();
              ARROW_ASSIGN_OR_RAISE(auto infile, fs_->OpenInputFile(full_path))
              ARROW_ASSIGN_OR_RAISE(auto reader,
                                    lance::io::FileReader::Make(infile, this->manifest_));
              return std::make_tuple(std::move(reader), intersection);
            },
            i));
    futs.emplace_back(std::move(future));
  }

  std::vector<LanceFragment::FileReaderWithSchema> readers;
  for (auto& future : futs) {
    ARROW_ASSIGN_OR_RAISE(auto& reader, future.result());
    if (std::get<0>(reader) != nullptr) {
      readers.emplace_back(std::move(reader));
    }
  }

  return readers;
}

const std::shared_ptr<format::Schema>& LanceFragment::schema() const { return manifest_->schema(); }

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

::arrow::Result<std::shared_ptr<LanceFragment>> LanceFragment::AddColumn(
    const std::shared_ptr<::arrow::dataset::ScanOptions>& options,
    const std::shared_ptr<format::Schema>& column_schema,
    Updater updater) {
  std::string file_path = fs::path(data_uri_) / fmt::format("{}.lance", GetUUIDString());
  ARROW_ASSIGN_OR_RAISE(auto output, fs_->OpenOutputStream(file_path));
  auto write_options = LanceFileFormat().DefaultWriteOptions();
  io::FileWriter writer(column_schema, std::move(write_options), std::move(output));

  ARROW_ASSIGN_OR_RAISE(auto generator, ScanBatchesAsync(options));
  auto arrow_schema = column_schema->ToArrow();
  while (true) {
    auto fut = generator();
    ARROW_ASSIGN_OR_RAISE(auto record_batch, fut.result());
    ARROW_ASSIGN_OR_RAISE(auto arr, updater(record_batch));
    if (arr->length() != record_batch->num_rows()) {
      return ::arrow::Status::IOError(
          "LanceFragment::AddColumn: updater must return the same size of output (",
          arr->length(),
          ") as input (",
          record_batch->num_rows(),
          ")");
    }
    auto result = ::arrow::RecordBatch::Make(arrow_schema, arr->length(), {arr});
    ARROW_RETURN_NOT_OK(writer.Write(result));
  }

  auto field_ids = column_schema->GetFieldIds();
  auto data_file = ::lance::format::DataFile({file_path, field_ids});
}

::arrow::Result<std::shared_ptr<LanceFragment>> LanceFragment::AddColumn(
    const std::shared_ptr<format::Schema>& dataset_schema,
    const std::shared_ptr<format::Schema>& column_schema,
    const std::shared_ptr<::arrow::ChunkedArray>& column,
    ::arrow::MemoryPool* pool) {
  ARROW_ASSIGN_OR_RAISE(auto reader, OpenReader(0));
  if (reader->length() != column->length()) {
    return ::arrow::Status::Invalid(
        fmt::format("LanceFragment::AddColumn: array length ({}) != existing data length ({}).",
                    reader->length(),
                    column->length()));
  }

  // Step 2. loop over chunk size, and slice the column
  auto arrow_schema = column_schema->ToArrow();
  auto new_table = ::arrow::Table::Make(arrow_schema, {column});

  auto field_ids = column_schema->GetFieldIds();
  std::string file_path = fs::path(data_uri_) / fmt::format("{}.lance", GetUUIDString());

  {
    ARROW_ASSIGN_OR_RAISE(auto output, fs_->OpenOutputStream(file_path));
    auto write_options = LanceFileFormat().DefaultWriteOptions();
    io::FileWriter writer(column_schema, std::move(write_options), std::move(output));

    for (int64_t batch = 0, start = 0; batch < reader->num_batches(); batch++) {
      auto length = reader->metadata().GetBatchLength(batch);
      auto slice = new_table->Slice(start, length);
      ARROW_ASSIGN_OR_RAISE(auto record_batch, slice->CombineChunksToBatch(pool));
      ARROW_RETURN_NOT_OK(writer.Write(record_batch));
      start += length;
    }
    writer.Finish().Wait();
    // Writer is closed because out of scope.
  }

  /// Append the file to the data fragment and create new LangeFragment.
  auto new_data_fragment = std::make_shared<format::DataFragment>(*fragment_);
  new_data_fragment->data_files().emplace_back(::lance::format::DataFile({file_path, field_ids}));

  //  return std::make_shared<LanceFragment>(fs_, data_uri_, new_data_fragment, dataset_schema);
}

}  // namespace lance::arrow