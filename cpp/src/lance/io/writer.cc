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

#include "lance/io/writer.h"

#include <arrow/array.h>
#include <arrow/dataset/file_base.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/type.h"
#include "lance/format/format.h"
#include "lance/format/manifest.h"
#include "lance/format/metadata.h"
#include "lance/format/schema.h"
#include "lance/io/pb.h"

namespace lance::io {

static constexpr int16_t kMajorVersion = 0;
static constexpr int16_t kMinorVersion = 1;

namespace internal {

/**
 * Write the 16 bytes footer to the end of a file.
 *
 * @param sink arrow OutputStream.
 * @param metadata_offset the absolute offset from the beginning of the file for metadata.
 *
 * @return Status::OK if success
 *
 * It writes 16 bytes in total:
 *  - 8 bytes metadata_offset
 *  - 2 bytes major version number
 *  - 2 bytes minor version number
 *  - 4 bytes magic word.
 */
::arrow::Status WriteFooter(std::shared_ptr<::arrow::io::OutputStream> sink,
                            int64_t metadata_offset) {
  ARROW_RETURN_NOT_OK(WriteInt<int64_t>(sink, metadata_offset));
  ARROW_RETURN_NOT_OK(WriteInt<int16_t>(sink, kMajorVersion));
  ARROW_RETURN_NOT_OK(WriteInt<int16_t>(sink, kMinorVersion));
  return sink->Write(lance::format::kMagic, 4);
}

}  // namespace internal

FileWriter::FileWriter(std::shared_ptr<::arrow::Schema> schema,
                       std::shared_ptr<::arrow::dataset::FileWriteOptions> options,
                       std::shared_ptr<::arrow::io::OutputStream> destination,
                       ::arrow::fs::FileLocator destination_locator)
    : ::arrow::dataset::FileWriter(schema, options, destination, destination_locator),
      lance_schema_(std::make_unique<lance::format::Schema>(schema)),
      metadata_(std::make_unique<lance::format::Metadata>()) {
  assert(schema->num_fields() > 0);
}

FileWriter::~FileWriter() {}

::arrow::Status FileWriter::Write(const std::shared_ptr<::arrow::RecordBatch>& batch) {
  metadata_->AddChunkOffset(batch->num_rows());

  for (const auto& field : lance_schema_->fields()) {
    ARROW_RETURN_NOT_OK(WriteArray(field, batch->GetColumnByName(field->name())));
  }
  chunk_id_++;
  return ::arrow::Status::OK();
}

::arrow::Status FileWriter::WriteArray(const std::shared_ptr<format::Field>& field,
                                       const std::shared_ptr<::arrow::Array>& arr) {
  assert(field->type()->id() == arr->type_id());
  if (::arrow::is_primitive(arr->type_id()) || ::arrow::is_binary_like(arr->type_id())) {
    return WritePrimitiveArray(field, arr);
  } else if (lance::arrow::is_struct(arr->type())) {
    return WriteStructArray(field, arr);
  } else if (lance::arrow::is_list(arr->type())) {
    return WriteListArray(field, arr);
  }
  assert(false);
}

::arrow::Status FileWriter::WritePrimitiveArray(const std::shared_ptr<format::Field>& field,
                                                const std::shared_ptr<::arrow::Array>& arr) {
  auto field_id = field->id();
  auto encoder = field->GetEncoder(destination_);
  lookup_table_.AddPageLength(field_id, chunk_id_, arr->length());
  ARROW_ASSIGN_OR_RAISE(auto pos, encoder->Write(arr));
  lookup_table_.AddOffset(field_id, chunk_id_, pos);
  return ::arrow::Status::OK();
}

::arrow::Status FileWriter::WriteStructArray(const std::shared_ptr<format::Field>& field,
                                             const std::shared_ptr<::arrow::Array>& arr) {
  assert(arrow::is_struct(field->type()));
  auto struct_arr = std::static_pointer_cast<::arrow::StructArray>(arr);
  assert(field->fields().size() == static_cast<size_t>(struct_arr->num_fields()));
  for (auto child : field->fields()) {
    auto child_arr = struct_arr->GetFieldByName(child->name());
    ARROW_RETURN_NOT_OK(WriteArray(child, child_arr));
  }
  return ::arrow::Status::OK();
}

::arrow::Status FileWriter::WriteListArray(const std::shared_ptr<format::Field>& field,
                                           const std::shared_ptr<::arrow::Array>& arr) {
  assert(field->logical_type() == "list" || field->logical_type() == "list.struct");
  assert(field->fields().size() == 1);
  auto list_arr = std::static_pointer_cast<::arrow::ListArray>(arr);
  auto offset_arr = list_arr->offsets();
  ARROW_RETURN_NOT_OK(WritePrimitiveArray(field, list_arr->offsets()));
  auto child_field = field->field(0);
  return WriteArray(child_field, list_arr->values());
}

::arrow::Status FileWriter::WriteFooter() {
  ARROW_ASSIGN_OR_RAISE(auto pos, lookup_table_.Write(destination_));
  metadata_->SetChunkPosition(pos);
  lookup_table_.WritePageLengthTo(&metadata_->pb());

  std::string primary_key;
  if (options_->type_name() == lance::arrow::LanceFileFormat::Make()->type_name()) {
    auto opts = std::dynamic_pointer_cast<lance::arrow::FileWriteOptions>(options_);
    primary_key = opts->primary_key;
  }
  format::Manifest manifest(primary_key, lance_schema_);
  ARROW_ASSIGN_OR_RAISE(pos, manifest.Write(destination_));
  metadata_->pb().set_manifest_position(pos);

  ARROW_ASSIGN_OR_RAISE(pos, WriteProto(destination_, metadata_->pb()));
  return internal::WriteFooter(destination_, pos);
}

::arrow::Future<> FileWriter::FinishInternal() {
  return ::arrow::Future<>::MakeFinished(this->WriteFooter());
};

}  // namespace lance::io
