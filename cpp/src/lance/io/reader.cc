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
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>

#include "lance/arrow/type.h"
#include "lance/encodings/binary.h"
#include "lance/encodings/plain.h"
#include "lance/format/format.h"
#include "lance/format/lookup_table.h"
#include "lance/format/manifest.h"
#include "lance/format/metadata.h"
#include "lance/format/schema.h"
#include "lance/io/endian.h"

using arrow::Result;
using arrow::Status;
using std::unique_ptr;

using lance::arrow::is_list;
using lance::arrow::is_struct;

namespace pb = lance::format::pb;

typedef ::arrow::Result<std::shared_ptr<::arrow::Scalar>> ScalarResult;

namespace lance::io {

::arrow::Result<int64_t> ReadFooter(const std::shared_ptr<::arrow::Buffer>& buf) {
  assert(buf->size() >= 16);
  if (auto magic_buf = ::arrow::SliceBuffer(buf, buf->size() - 4);
      !magic_buf->Equals(::arrow::Buffer(lance::format::kMagic))) {
    return Status::IOError(
        fmt::format("Invalidate file format: MAGIC NUM is not {}", lance::format::kMagic));
  }
  // Metadata Offset
  return ReadInt<int64_t>(buf->data() + buf->size() - 16);
}

FileReader::FileReader(std::shared_ptr<::arrow::io::RandomAccessFile> in,
                       ::arrow::MemoryPool* pool) noexcept
    : file_(in), pool_(pool) {}

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
      metadata_, format::Metadata::Parse(::arrow::SliceBuffer(cached_last_page_, inbuf_offset)));

  ARROW_ASSIGN_OR_RAISE(manifest_, metadata_->GetManifest(file_));
  // TODO: Let's assume that chunk position is prefetched in memory already.
  assert(metadata_->chunk_position() >= size - kPrefetchSize);

  ARROW_ASSIGN_OR_RAISE(
      lookup_table_,
      format::LookupTable::Read(file_, metadata_->chunk_position(), metadata_->pb()));
  return Status::OK();
}

const lance::format::Schema& FileReader::schema() const { return manifest_->schema(); }

const lance::format::Manifest& FileReader::manifest() const { return *manifest_; }

const lance::format::Metadata& FileReader::metadata() const { return *metadata_; }

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const {
  if (field->logical_type() == "struct") {
    return GetStructScalar(field, chunk_id, idx);
  } else if (field->logical_type() == "list" || field->logical_type() == "list.struct") {
    return GetListScalar(field, chunk_id, idx);
  } else {
    return GetPrimitiveScalar(field, chunk_id, idx);
  }
}

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetPrimitiveScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const {
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
  ARROW_ASSIGN_OR_RAISE(auto pos, GetChunkOffset(field_id, chunk_id));
  ARROW_ASSIGN_OR_RAISE(auto length, lookup_table_->GetPageLength(field_id, chunk_id));
  decoder->Reset(pos, length);
  return decoder->GetScalar(idx);
}

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetStructScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const {
  ::arrow::StructScalar::ValueType values;
  std::vector<std::future<ScalarResult>> futures;
  for (auto& child : field->fields()) {
    futures.emplace_back(std::async(&FileReader::GetScalar, this, child, chunk_id, idx));
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
  ::arrow::Int32Builder builder;
  for (int i = 0; i < offsets->length(); i++) {
    ARROW_RETURN_NOT_OK(builder.Append(offsets->Value(i) - start_pos));
  }
  ARROW_ASSIGN_OR_RAISE(auto arr, builder.Finish());

  return std::static_pointer_cast<::arrow::Int32Array>(arr);
}

::arrow::Result<::std::shared_ptr<::arrow::Scalar>> FileReader::GetListScalar(
    const std::shared_ptr<lance::format::Field>& field, int32_t chunk_id, int32_t idx) const {
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
  ARROW_ASSIGN_OR_RAISE(auto pos, GetChunkOffset(field_id, chunk_id));
  ARROW_ASSIGN_OR_RAISE(auto length, lookup_table_->GetPageLength(field_id, chunk_id));
  decoder->Reset(pos, length);
  ARROW_ASSIGN_OR_RAISE(auto offsets_arr, decoder->ToArray(idx, 2));
  auto offsets = std::static_pointer_cast<::arrow::Int32Array>(offsets_arr);
  ARROW_ASSIGN_OR_RAISE(
      auto values,
      GetArray(
          field->fields()[0], chunk_id, offsets->Value(0), offsets->Value(1) - offsets->Value(0)));
  ARROW_ASSIGN_OR_RAISE(offsets, ResetOffsets(offsets));
  auto rst = ::arrow::ListArray::FromArrays(field->type(), *offsets, *values);
  if (!rst.ok()) {
    fmt::print(stderr, "GetListScalar error: {}\n", rst.status().message());
    return rst.status();
  }
  ARROW_ASSIGN_OR_RAISE(auto list_arr,
                        ::arrow::ListArray::FromArrays(field->type(), *offsets, *values));
  return std::make_shared<::arrow::ListScalar>(values);
}

::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> FileReader::Get(
    int32_t idx, const format::Schema& schema) {
  auto chunk_result = metadata_->LocateChunk(idx);
  if (!chunk_result.ok()) {
    return chunk_result.status();
  }
  auto [chunk_id, idx_in_chunk] = *chunk_result;
  auto row = std::vector<::std::shared_ptr<::arrow::Scalar>>();
  std::vector<std::future<::arrow::Result<::std::shared_ptr<::arrow::Scalar>>>> futures;
  for (auto& field : schema.fields()) {
    auto f = std::async(&FileReader::GetScalar, this, field, chunk_id, idx_in_chunk);
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
  auto schema = manifest_->schema();
  ARROW_ASSIGN_OR_RAISE(auto projection, schema.Project(columns));
  return Get(idx, *projection);
}

::arrow::Result<std::vector<::std::shared_ptr<::arrow::Scalar>>> FileReader::Get(int32_t idx) {
  return Get(idx, manifest_->schema());
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable() {
  std::vector<std::shared_ptr<::arrow::ChunkedArray>> columns;
  return ReadTable(manifest_->schema());
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable(
    const std::vector<std::string>& columns) {
  auto schema = manifest_->schema();
  ARROW_ASSIGN_OR_RAISE(auto projection, schema.Project(columns));
  return ReadTable(*projection);
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable(
    const lance::format::Schema& schema) const {
  std::vector<std::shared_ptr<::arrow::ChunkedArray>> columns;
  for (auto& field : schema.fields()) {
    ::arrow::ArrayVector chunks;
    for (int i = 0; i < metadata_->num_chunks(); i++) {
      ARROW_ASSIGN_OR_RAISE(auto arr, GetArray(field, i));
      chunks.emplace_back(arr);
    }
    columns.emplace_back(std::make_shared<::arrow::ChunkedArray>(chunks));
  }
  return ::arrow::Table::Make(schema.ToArrow(), columns);
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> FileReader::ReadBatch(
    int32_t offset, int32_t length, const lance::format::Schema& schema) const {
  ARROW_ASSIGN_OR_RAISE(auto chunk_and_idx, metadata_->LocateChunk(offset));
  auto [chunk_id, idx_in_chunk] = chunk_and_idx;
  std::vector<std::shared_ptr<::arrow::Array>> arrs;
  for (auto& field : schema.fields()) {
    auto len =
        static_cast<int32_t>(std::min(static_cast<int64_t>(length), metadata_->length() - offset));
    ::arrow::ArrayVector chunks;
    // Make a local copy?
    auto ckid = chunk_id;
    auto ck_index = idx_in_chunk;
    while (len > 0 && ckid < metadata_->num_chunks()) {
      auto page_length = metadata_->GetChunkLength(chunk_id);
      auto length_in_chunk = std::min(len, page_length - ck_index);
      ARROW_ASSIGN_OR_RAISE(auto arr, GetArray(field, ckid, ck_index, length_in_chunk));
      len -= length_in_chunk;
      ckid++;
      ck_index = 0;
      chunks.emplace_back(arr);
    }
    assert(!chunks.empty());
    if (chunks.size() > 1) {
      ARROW_ASSIGN_OR_RAISE(auto arr, ::arrow::Concatenate(chunks, pool_));
      arrs.emplace_back(arr);
    } else {
      arrs.emplace_back(chunks[0]);
    }
  }
  return ::arrow::RecordBatch::Make(schema.ToArrow(), arrs[0]->length(), arrs);
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> FileReader::ReadBatch(
    int32_t offset, int32_t length, const std::vector<std::string>& columns) const {
  auto schema = manifest_->schema();
  ARROW_ASSIGN_OR_RAISE(auto projection, schema.Project(columns));

  return ReadBatch(offset, length, *projection);
}

::arrow::Result<int64_t> FileReader::GetChunkOffset(int64_t field_id, int64_t chunk_id) const {
  auto offset = lookup_table_->GetOffset(field_id, chunk_id);
  if (offset.has_value()) {
    return offset.value();
  }
  return ::arrow::Status::Invalid(
      fmt::format("Invalid access for chunk offset: field={} chunk={}", field_id, chunk_id));
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetArray(
    const std::shared_ptr<lance::format::Field>& field,
    int chunk_id,
    int32_t start,
    std::optional<int32_t> length) const {
  auto dtype = field->type();
  if (is_struct(dtype)) {
    return GetStructArray(field, chunk_id, start, length);
  } else if (is_list(dtype)) {
    return GetListArray(field, chunk_id, start, length);
  } else {
    return GetPrimitiveArray(field, chunk_id, start, length);
  }
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetStructArray(
    const std::shared_ptr<lance::format::Field>& field,
    int chunk_id,
    int32_t start,
    std::optional<int32_t> length) const {
  ::arrow::ArrayVector children;
  std::vector<std::string> field_names;
  for (auto child : field->fields()) {
    ARROW_ASSIGN_OR_RAISE(auto arr, GetArray(child, chunk_id, start, length));
    children.emplace_back(arr);
    field_names.emplace_back(child->name());
  }
  auto arr = ::arrow::StructArray::Make(children, field_names);
  return arr;
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetListArray(
    const std::shared_ptr<lance::format::Field>& field,
    int chunk_id,
    int32_t start,
    std::optional<int32_t> length) const {
  /// Consolidate this with primitive Array
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto pos, GetChunkOffset(field_id, chunk_id));
  ARROW_ASSIGN_OR_RAISE(auto chunk_length, lookup_table_->GetPageLength(field_id, chunk_id));
  if (!length.has_value()) {
    length = chunk_length - start;
  }

  auto decoder = std::make_shared<lance::encodings::PlainDecoder<::arrow::Int32Type>>(
      file_, pos, chunk_length);
  // TODO: fix this
  auto l = std::min(static_cast<int64_t>(*length + 1), chunk_length - start);
  ARROW_ASSIGN_OR_RAISE(auto offsets, decoder->ToArray(start, l));
  auto offsets_arr = std::static_pointer_cast<::arrow::Int32Array>(offsets);
  int32_t start_pos = offsets_arr->Value(0);
  int32_t array_length = offsets_arr->Value(offsets_arr->length() - 1) - start_pos;
  ARROW_ASSIGN_OR_RAISE(auto values,
                        GetArray(field->fields()[0], chunk_id, start_pos, array_length));

  // Realigned offsets
  ARROW_ASSIGN_OR_RAISE(auto shiftted_offsets, ResetOffsets(offsets_arr));
  auto result = ::arrow::ListArray::FromArrays(*shiftted_offsets, *values, pool_);
  if (!result.ok()) {
    fmt::print("GetListArray: field={}, chunk_id={}, start={}, length={}\nReason:{}\n",
               field->name(),
               chunk_id,
               start,
               *length,
               result.status().message());
  }
  return result;
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FileReader::GetPrimitiveArray(
    const std::shared_ptr<lance::format::Field>& field,
    int chunk_id,
    int32_t start,
    std::optional<int32_t> length) const {
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto pos, GetChunkOffset(field_id, chunk_id));
  ARROW_ASSIGN_OR_RAISE(auto chunk_length, lookup_table_->GetPageLength(field_id, chunk_id));
  ARROW_ASSIGN_OR_RAISE(auto decoder, field->GetDecoder(file_));
  decoder->Reset(pos, chunk_length);
  auto result = decoder->ToArray(start, length);
  if (!result.ok()) {
    return result.status().WithMessage(fmt::format("FileReader::GetPrimitiveArray: field={}({}) {}",
                                                   field->name(),
                                                   field->id(),
                                                   result.status().message()));
  }
  return result;
}

}  // namespace lance::io