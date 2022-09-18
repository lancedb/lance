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
#include <arrow/compute/exec.h>
#include <arrow/dataset/file_base.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/type.h"
#include "lance/format/format.h"
#include "lance/format/manifest.h"
#include "lance/format/metadata.h"
#include "lance/format/schema.h"
#include "lance/format/visitors.h"
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
  metadata_->AddBatchLength(batch->num_rows());
  for (const auto& field : lance_schema_->fields()) {
    ARROW_RETURN_NOT_OK(WriteArray(field, batch->GetColumnByName(field->name())));
  }
  batch_id_++;
  return ::arrow::Status::OK();
}

::arrow::Status FileWriter::WriteArray(const std::shared_ptr<format::Field>& field,
                                       const std::shared_ptr<::arrow::Array>& arr) {
  if (::lance::arrow::is_extension(arr->type())) {
    assert(field->is_extension_type());
    auto ext_array = std::static_pointer_cast<::arrow::ExtensionArray>(arr);
    return WriteArray(field, ext_array->storage());
  }

  // If field is an extension type, storage_id is the underneath storage arr's type id.
  // Otherwise, storage_id is the same as type()->id()
  assert(field->type()->storage_id() == arr->type_id());

  if (arrow::is_fixed_length(arr->type_id())) {
    return WriteFixedLengthArray(field, arr);
  } else if (lance::arrow::is_struct(arr->type())) {
    return WriteStructArray(field, arr);
  } else if (lance::arrow::is_list(arr->type())) {
    return WriteListArray(field, arr);
  } else if (::arrow::is_dictionary(arr->type_id())) {
    return WriteDictionaryArray(field, arr);
  }
  return ::arrow::Status::Invalid("FileWriter::WriteArray: unsupported data type: ",
                                  arr->type()->ToString());
}

::arrow::Status FileWriter::WriteFixedLengthArray(const std::shared_ptr<format::Field>& field,
                                                  const std::shared_ptr<::arrow::Array>& arr) {
  auto field_id = field->id();
  auto encoder = field->GetEncoder(destination_);
  auto type = field->type();

  // Physical array.
  ::arrow::Result<std::shared_ptr<::arrow::Array>> storage_arr;
  switch (type->id()) {
    case ::arrow::TimestampType::type_id:
    case ::arrow::Date64Type::type_id:
    case ::arrow::Time64Type::type_id:
      storage_arr = arr->View(::arrow::int64());
      break;
    case ::arrow::Date32Type::type_id:
    case ::arrow::Time32Type::type_id:
      storage_arr = arr->View(::arrow::int32());
      break;
    default:
      storage_arr = arr;
      break;
  }
  if (!storage_arr.ok()) {
    return storage_arr.status();
  }

  ARROW_ASSIGN_OR_RAISE(auto pos, encoder->Write(storage_arr.ValueOrDie()));
  lookup_table_.SetPageInfo(field_id, batch_id_, pos, arr->length());
  return ::arrow::Status::OK();
}

::arrow::Status FileWriter::WriteStructArray(const std::shared_ptr<format::Field>& field,
                                             const std::shared_ptr<::arrow::Array>& arr) {
  assert(arrow::is_struct(field->type()->storage_id()));
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
  auto child_field = field->field(0);

  ARROW_ASSIGN_OR_RAISE(
      auto offsets_datum,
      ::arrow::compute::CallFunction(
          "subtract", {list_arr->offsets(), list_arr->offsets()->GetScalar(0).ValueOrDie()}));
  ARROW_RETURN_NOT_OK(WriteFixedLengthArray(field, offsets_datum.make_array()));

  auto start_offset = list_arr->value_offset(0);
  auto last_offset = list_arr->value_offset(arr->length());
  auto child_length = last_offset - start_offset;
  return WriteArray(child_field, list_arr->values()->Slice(start_offset, child_length));
}

::arrow::Status FileWriter::WriteDictionaryArray(const std::shared_ptr<format::Field>& field,
                                                 const std::shared_ptr<::arrow::Array>& arr) {
  assert(field->logical_type().starts_with("dict:"));
  auto encoder = field->GetEncoder(destination_);
  auto dict_arr = std::dynamic_pointer_cast<::arrow::DictionaryArray>(arr);
  if (!field->dictionary()) {
    ARROW_RETURN_NOT_OK(field->set_dictionary(dict_arr->dictionary()));
  }
  auto field_id = field->id();
  ARROW_ASSIGN_OR_RAISE(auto pos, encoder->Write(arr));
  lookup_table_.SetPageInfo(field_id, batch_id_, pos, arr->length());
  return ::arrow::Status::OK();
}

::arrow::Status FileWriter::WriteFooter() {
  // Write dictionary values first.
  auto visitor = format::WriteDictionaryVisitor(destination_);
  ARROW_RETURN_NOT_OK(visitor.VisitSchema(*lance_schema_));

  ARROW_ASSIGN_OR_RAISE(auto pos, lookup_table_.Write(destination_));
  metadata_->SetPageTablePosition(pos);

  if (options_->type_name() == lance::arrow::LanceFileFormat::Make()->type_name()) {
    auto opts = std::dynamic_pointer_cast<lance::arrow::FileWriteOptions>(options_);
  }
  format::Manifest manifest(lance_schema_);
  ARROW_ASSIGN_OR_RAISE(pos, manifest.Write(destination_));
  metadata_->SetManifestPosition(pos);

  ARROW_ASSIGN_OR_RAISE(pos, metadata_->Write(destination_));
  return internal::WriteFooter(destination_, pos);
}

::arrow::Future<> FileWriter::FinishInternal() {
  return ::arrow::Future<>::MakeFinished(this->WriteFooter());
};

}  // namespace lance::io
