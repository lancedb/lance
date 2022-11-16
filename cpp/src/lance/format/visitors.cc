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

#include "lance/format/visitors.h"

#include <arrow/io/api.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>

#include <memory>

#include "lance/encodings/dictionary.h"
#include "lance/format/schema.h"

namespace lance::format {

::arrow::Status FieldVisitor::VisitSchema(const Schema& schema) {
  for (auto& field : schema.fields()) {
    ARROW_RETURN_NOT_OK(Visit(field));
  }
  return ::arrow::Status::OK();
}

::arrow::Status ToArrowVisitor::Visit(std::shared_ptr<Field> root) {
  for (auto& child : root->children_) {
    ARROW_ASSIGN_OR_RAISE(auto arrow_field, DoVisit(child));
    arrow_fields_.push_back(arrow_field);
  }
  return ::arrow::Status::OK();
}

std::shared_ptr<::arrow::Schema> ToArrowVisitor::Finish() { return ::arrow::schema(arrow_fields_); }

::arrow::Result<::std::shared_ptr<::arrow::Field>> ToArrowVisitor::DoVisit(
    std::shared_ptr<Field> node) {
  return node->ToArrow();
}

WriteDictionaryVisitor::WriteDictionaryVisitor(std::shared_ptr<::arrow::io::OutputStream> out)
    : out_(std::move(out)) {}

::arrow::Status WriteDictionaryVisitor::Visit(std::shared_ptr<Field> root) {
  if (::arrow::is_dictionary(root->storage_type()->id())) {
    assert(root->dictionary());
    auto decoder =
        std::dynamic_pointer_cast<lance::encodings::DictionaryEncoder>(root->GetEncoder(out_));
    ARROW_ASSIGN_OR_RAISE(auto offset, decoder->WriteValueArray(root->dictionary()));
    root->dictionary_offset_ = offset;
    root->dictionary_page_length_ = root->dictionary()->length();
  }
  for (auto& child : root->fields()) {
    ARROW_RETURN_NOT_OK(Visit(child));
  }
  return ::arrow::Status::OK();
}

/// LoadDictionaryVisitor
ReadDictionaryVisitor::ReadDictionaryVisitor(std::shared_ptr<::arrow::io::RandomAccessFile> in)
    : in_(std::move(in)) {}

::arrow::Status ReadDictionaryVisitor::Visit(std::shared_ptr<Field> root) {
  if (::arrow::is_dictionary(root->type()->id())) {
    ARROW_RETURN_NOT_OK(root->LoadDictionary(in_));
  }
  for (auto& child : root->fields()) {
    ARROW_RETURN_NOT_OK(Visit(child));
  }
  return ::arrow::Status::OK();
}

}  // namespace lance::format
