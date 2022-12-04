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

#include "lance/io/reader.h"

#include <arrow/array/concatenate.h>
#include <arrow/array/util.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <range/v3/algorithm/max.hpp>

#include "lance/arrow/type.h"
#include "lance/arrow/utils.h"
#include "lance/encodings/binary.h"
#include "lance/encodings/plain.h"
#include "lance/format/format.h"
#include "lance/format/manifest.h"
#include "lance/format/metadata.h"
#include "lance/format/page_table.h"
#include "lance/format/schema.h"
#include "lance/io/endian.h"

using arrow::Result;
using arrow::Status;
using std::unique_ptr;

using lance::arrow::is_list;
using lance::arrow::is_struct;

typedef ::arrow::Result<std::shared_ptr<::arrow::Scalar>> ScalarResult;

namespace lance::io {

namespace {

::arrow::Result<int64_t> ReadFooter(const std::shared_ptr<::arrow::Buffer>& buf) {
  assert(buf->size() >= 16);
  if (auto magic_buf = ::arrow::SliceBuffer(buf, buf->size() - 4);
      !magic_buf->Equals(::arrow::Buffer(lance::format::kMagic))) {
    return Status::IOError(
        fmt::format("Invalidate file format: MAGIC NUM is not {}", lance::format::kMagic));
  }
  return ReadInt<int64_t>(buf->data() + buf->size() - 16);
}

}  // namespace

::arrow::Result<std::unique_ptr<FileReader>> FileReader::Make(
    std::shared_ptr<::arrow::io::RandomAccessFile> in,
    std::shared_ptr<::lance::format::Manifest> manifest,
    ::arrow::MemoryPool* pool) {
  auto reader = std::make_unique<FileReader>(std::move(in), std::move(manifest), pool);
  ARROW_RETURN_NOT_OK(reader->Open());
  return reader;
}

::arrow::Result<std::shared_ptr<::lance::format::Manifest>> FileReader::OpenManifest(
    const std::shared_ptr<::arrow::io::RandomAccessFile>& in) {
  constexpr auto kBufReadBytes = 8 * 1024 * 1024;  // Read 8 MB;
  ::arrow::BufferVector buffers;
  int64_t pos = 0;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto buf, in->ReadAt(pos, kBufReadBytes));
    auto read_nbytes = buf->size();
    if (read_nbytes > 0) {
      buffers.emplace_back(std::move(buf));
    }
    if (read_nbytes < kBufReadBytes) {
      break;
    }
    pos += read_nbytes;
  }
  assert(!buffers.empty());
  auto buf = buffers[0];
  if (buffers.size() > 1) {
    // Unlikely to get here
    ARROW_ASSIGN_OR_RAISE(buf, ::arrow::ConcatenateBuffers(buffers));
  }
  ARROW_ASSIGN_OR_RAISE(auto manifest_pos, ReadFooter(buf));
  ARROW_ASSIGN_OR_RAISE(auto manifest,
                        lance::format::Manifest::Parse(SliceBuffer(buf, manifest_pos)));

  /// TODO: optimize ReadDictionaryVisitor to read from buffer.
  auto visitor = format::ReadDictionaryVisitor(in);
  ARROW_RETURN_NOT_OK(visitor.VisitSchema(*manifest->schema()));
  return manifest;
}

FileReader::FileReader(std::shared_ptr<::arrow::io::RandomAccessFile> in,
                       std::shared_ptr<::lance::format::Manifest> manifest,
                       ::arrow::MemoryPool* pool) noexcept
    : file_(std::move(in)), pool_(pool), manifest_(std::move(manifest)) {}

Status FileReader::Open() {
  ARROW_ASSIGN_OR_RAISE(auto size, file_->GetSize());

  const int64_t kPrefetchSize = 1024 * 64;  // Read 64K
  int64_t footer_read_len = std::min(size, kPrefetchSize);
  if (footer_read_len < 16) {
    return Status::IOError(fmt::format("Invalidate file format: file size ({}) < 16", size));
  }
  // TODO: use memory pool for buffer?
  ARROW_ASSIGN_OR_RAISE(cached_last_page_, file_->ReadAt(size - footer_read_len, footer_read_len));

  ARROW_ASSIGN_OR_RAISE(auto metadata_offset, ReadFooter(cached_last_page_));

  // Lets assume the footer is not bigger than 1KB, so we've already read it.
  // TODO: we should prob adjust buffer again in production.
  auto inbuf_offset = footer_read_len - (size - metadata_offset);
  assert(inbuf_offset >= 0);
  ARROW_ASSIGN_OR_RAISE(
      metadata_, format::Metadata::Make(::arrow::SliceBuffer(cached_last_page_, inbuf_offset)));

  if (!manifest_) {
    /// Multiple files can share the manifest.
    ARROW_ASSIGN_OR_RAISE(manifest_,
                          format::Manifest::Parse(file_, metadata_->manifest_position()));
    // We need read the dictionary from the same file.
    auto visitor = format::ReadDictionaryVisitor(file_);
    ARROW_RETURN_NOT_OK(visitor.VisitSchema(*manifest_->schema()));
  }

  // TODO: Let's assume that page position is prefetched in memory already.
  assert(metadata_->page_table_position() >= size - kPrefetchSize);

  auto num_batches = metadata_->num_batches();
  auto num_columns = ranges::max(manifest_->schema()->GetFieldIds()) + 1;
  ARROW_ASSIGN_OR_RAISE(
      page_table_,
      format::PageTable::Make(file_, metadata_->page_table_position(), num_columns, num_batches));
  return Status::OK();
}

const lance::format::Schema& FileReader::schema() const { return *manifest_->schema(); }

const std::shared_ptr<lance::format::Manifest>& FileReader::manifest() const { return manifest_; }

const lance::format::Metadata& FileReader::metadata() const { return *metadata_; }

int64_t FileReader::length() const { return metadata_->length(); }

int32_t FileReader::num_batches() const { return metadata_->num_batches(); }

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const {
  if (field->logical_type() == "struct") {
    return GetStructScalar(field, batch_id, idx);
  } else if (field->logical_type() == "list" || field->logical_type() == "list.struct") {
    return GetListScalar(field, batch_id, idx);
  } else {
    return GetPrimitiveScalar(field, batch_id, idx);
  }
}

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetPrimitiveScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const {
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
  ARROW_ASSIGN_OR_RAISE(auto page, GetPageInfo(field_id, batch_id));
  auto [pos, length] = page;
  decoder->Reset(pos, length);
  return decoder->GetScalar(idx);
}

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetStructScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const {
  ::arrow::StructScalar::ValueType values;
  std::vector<std::future<ScalarResult>> futures;
  for (auto& child : field->fields()) {
    futures.emplace_back(std::async(&FileReader::GetScalar, this, child, batch_id, idx));
  }
  for (auto& f : futures) {
    ARROW_ASSIGN_OR_RAISE(auto v, f.get());
    values.emplace_back(v);
  }
  return std::make_shared<::arrow::StructScalar>(values, field->type());
}

::arrow::Result<std::shared_ptr<::arrow::Int32Array>> ResetOffsets(
    const std::shared_ptr<::arrow::Int32Array>& offsets) {
  int32_t start_pos = offsets->Value(0);
  ARROW_ASSIGN_OR_RAISE(auto datum, ::arrow::compute::Subtract(offsets, ::arrow::Datum(start_pos)));
  return std::static_pointer_cast<::arrow::Int32Array>(datum.make_array());
}

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetListScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t batch_id, int32_t idx) const {
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
  ARROW_ASSIGN_OR_RAISE(auto page, GetPageInfo(field_id, batch_id));
  auto [pos, length] = page;
  decoder->Reset(pos, length);
  ARROW_ASSIGN_OR_RAISE(auto offsets_arr, decoder->ToArray(idx, 2));
  auto offsets = std::static_pointer_cast<::arrow::Int32Array>(offsets_arr);
  if (offsets->Value(0) == offsets->Value(1)) {
    return std::make_shared<::arrow::NullScalar>();
  }
  ARROW_ASSIGN_OR_RAISE(
      auto values,
      GetArray(field->fields()[0],
               batch_id,
               ArrayReadParams(offsets->Value(0), offsets->Value(1) - offsets->Value(0))));
  return std::make_shared<::arrow::ListScalar>(values);
}

::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> FileReader::Get(
    int32_t idx, const format::Schema& schema) {
  ARROW_ASSIGN_OR_RAISE(auto batch, metadata_->LocateBatch(idx));
  auto [batch_id, idx_in_batch] = batch;
  auto row = std::vector<::std::shared_ptr<::arrow::Scalar>>();
  std::vector<std::future<::arrow::Result<::std::shared_ptr<::arrow::Scalar>>>> futures;
  for (auto& field : schema.fields()) {
    auto f = std::async(&FileReader::GetScalar, this, field, batch_id, idx_in_batch);
    futures.emplace_back(std::move(f));
  }
  for (auto& f : futures) {
    ARROW_ASSIGN_OR_RAISE(auto val, f.get());
    row.emplace_back(val);
  }

  return row;
}

::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> FileReader::Get(
    int32_t idx, const std::vector<std::string>& columns) {
  ARROW_ASSIGN_OR_RAISE(auto projection, manifest_->schema()->Project(columns));
  return Get(idx, *projection);
}

::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> FileReader::Get(int32_t idx) {
  return Get(idx, *manifest_->schema());
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable() {
  std::vector<std::shared_ptr<::arrow::ChunkedArray>> columns;
  return ReadTable(*manifest_->schema());
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable(
    const lance::format::Schema& schema) const {
  std::vector<std::shared_ptr<::arrow::ChunkedArray>> columns;
  for (auto& field : schema.fields()) {
    ::arrow::ArrayVector chunks;
    for (int i = 0; i < metadata_->num_batches(); i++) {
      ARROW_ASSIGN_OR_RAISE(auto arr, GetArray(field, i, ArrayReadParams(0)));
      chunks.emplace_back(arr);
    }
    columns.emplace_back(std::make_shared<::arrow::ChunkedArray>(chunks));
  }
  return ::arrow::Table::Make(schema.ToArrow(), columns);
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> FileReader::ReadBatch(
    const lance::format::Schema& schema,
    int32_t batch_id,
    int32_t offset,
    std::optional<int32_t> length) const {
  return ReadBatch(schema, batch_id, ArrayReadParams(offset, length));
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> FileReader::ReadBatch(
    const lance::format::Schema& schema,
    int32_t batch_id,
    std::shared_ptr<::arrow::Int32Array> indices) const {
  return ReadBatch(schema, batch_id, ArrayReadParams(indices));
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> FileReader::ReadBatch(
    const lance::format::Schema& schema, int32_t batch_id, const ArrayReadParams& params) const {
  if (schema.fields().empty()) {
    return ::arrow::Status::Invalid("FileReader::ReadBatch: invalid schema: empty schema");
  }
  std::vector<std::shared_ptr<::arrow::Array>> arrs;
  /// TODO: GH-43. Read field in parallel.
  for (auto& field : schema.fields()) {
    ARROW_ASSIGN_OR_RAISE(auto arr, GetArray(field, batch_id, params));
    arrs.emplace_back(arr);
  }
  return ::arrow::RecordBatch::Make(schema.ToArrow(), arrs[0]->length(), arrs);
}

::arrow::Result<std::tuple<int64_t, int64_t>> FileReader::GetPageInfo(int32_t field_id,
                                                                      int32_t batch_id) const {
  auto offset = page_table_->GetPageInfo(field_id, batch_id);
  if (offset.has_value()) {
    return offset.value();
  }
  return ::arrow::Status::Invalid(
      fmt::format("Invalid access for page info: field={} batch={}", field_id, batch_id));
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetArray(
    const std::shared_ptr<lance::format::Field>& field,
    int32_t batch_id,
    const ArrayReadParams& params) const {
  auto dtype = field->type();
  auto storage_type = field->storage_type();

  std::shared_ptr<::arrow::Array> storage_arr;
  if (is_struct(storage_type)) {
    ARROW_ASSIGN_OR_RAISE(storage_arr, GetStructArray(field, batch_id, params));
  } else if (is_list(storage_type)) {
    ARROW_ASSIGN_OR_RAISE(storage_arr, GetListArray(field, batch_id, params));
  } else if (::arrow::is_dictionary(storage_type->id())) {
    ARROW_ASSIGN_OR_RAISE(storage_arr, GetDictionaryArray(field, batch_id, params));
  } else {
    ARROW_ASSIGN_OR_RAISE(auto primitive_arr, GetPrimitiveArray(field, batch_id, params));
    ARROW_ASSIGN_OR_RAISE(storage_arr, primitive_arr->View(storage_type));
  }

  if (lance::arrow::is_extension(dtype)) {
    return ::arrow::ExtensionType::WrapArray(dtype, storage_arr);
  }
  return storage_arr;
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetStructArray(
    const std::shared_ptr<lance::format::Field>& field,
    int32_t batch_id,
    const ArrayReadParams& params) const {
  ::arrow::ArrayVector children;
  std::vector<std::string> field_names;
  for (auto child : field->fields()) {
    ARROW_ASSIGN_OR_RAISE(auto arr, GetArray(child, batch_id, params));
    children.emplace_back(arr);
    field_names.emplace_back(child->name());
  }
  return ::arrow::StructArray::Make(children, field_names);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetListArray(
    const std::shared_ptr<lance::format::Field>& field,
    int batch_id,
    const ArrayReadParams& params) const {
  if (params.indices.has_value()) {
    // TODO: GH-39. We should improve the read behavior to use indices to save some I/Os.
    auto& indices = params.indices.value();
    if (indices->length() == 0) {
      return ::arrow::MakeEmptyArray(field->type());
    }
    auto start = static_cast<int32_t>(indices->Value(0));
    auto length = static_cast<int32_t>(indices->Value(indices->length() - 1) - start + 1);

    ARROW_ASSIGN_OR_RAISE(auto unfiltered_arr,
                          GetListArray(field, batch_id, ArrayReadParams(start, length)));
    ARROW_ASSIGN_OR_RAISE(auto offsets_datum,
                          ::arrow::compute::Subtract(indices, ::arrow::Datum(indices->Value(0))));
    ARROW_ASSIGN_OR_RAISE(
        auto datum,
        ::arrow::compute::CallFunction("take", {unfiltered_arr, offsets_datum.make_array()}));
    return datum.make_array();
  }

  auto length = params.length;
  auto start = params.offset.value();

  std::shared_ptr<::arrow::Array> offsets_arr;
  if (length.has_value()) {
    ARROW_ASSIGN_OR_RAISE(
        offsets_arr,
        GetPrimitiveArray(field, batch_id, ArrayReadParams(start, length.value() + 1)));
  } else {
    ARROW_ASSIGN_OR_RAISE(offsets_arr, GetPrimitiveArray(field, batch_id, ArrayReadParams(start)));
  }
  auto offsets = std::static_pointer_cast<::arrow::Int32Array>(offsets_arr);
  int32_t start_pos = offsets->Value(0);
  int32_t array_length = offsets->Value(offsets_arr->length() - 1) - start_pos;
  ARROW_ASSIGN_OR_RAISE(
      auto values,
      GetArray(field->fields()[0], batch_id, ArrayReadParams(start_pos, array_length)));
  // Realigned offsets to be zero-started
  ARROW_ASSIGN_OR_RAISE(auto shifted_offsets, ResetOffsets(offsets));
  // Setup null bitmap
  ARROW_ASSIGN_OR_RAISE(auto null_bitmap,
                        ::arrow::AllocateBitmap(shifted_offsets->length() - 1, pool_));
  for (int i = 0; i < shifted_offsets->length() - 1; i++) {
    ::arrow::bit_util::SetBitTo(
        null_bitmap->mutable_data(), i, offsets->Value(i + 1) - offsets->Value(i) > 0);
  }
  return std::make_shared<::arrow::ListArray>(field->type(),
                                              shifted_offsets->length() - 1,
                                              shifted_offsets->data()->buffers[1],
                                              values,
                                              null_bitmap);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetDictionaryArray(
    const std::shared_ptr<lance::format::Field>& field,
    int batch_id,
    const ArrayReadParams& params) const {
  assert(::arrow::is_dictionary(field->type()->id()));
  return GetPrimitiveArray(field, batch_id, params);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetPrimitiveArray(
    const std::shared_ptr<lance::format::Field>& field,
    int batch_id,
    const ArrayReadParams& params) const {
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto page_info, GetPageInfo(field_id, batch_id));
  auto [position, length] = page_info;
  ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
  decoder->Reset(position, length);
  decltype(decoder->ToArray()) result;
  if (params.indices) {
    return decoder->Take(params.indices.value());
  } else {
    return decoder->ToArray(params.offset.value(), params.length);
  }
}

FileReader::ArrayReadParams::ArrayReadParams(int32_t off, std::optional<int32_t> len)
    : offset(off), length(len) {}

FileReader::ArrayReadParams::ArrayReadParams(std::shared_ptr<::arrow::Int32Array> idx)
    : indices(idx) {}

}  // namespace lance::io