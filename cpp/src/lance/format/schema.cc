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

#include "lance/format/schema.h"

#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/string.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "lance/arrow/type.h"
#include "lance/encodings/binary.h"
#include "lance/encodings/dictionary.h"
#include "lance/encodings/plain.h"

using std::make_shared;
using std::string;
using std::vector;

namespace lance::format {

Field::Field() : id_(-1), parent_(-1), encoding_(pb::NONE) {}

Field::Field(const std::shared_ptr<::arrow::Field>& field)
    : id_(0),
      parent_(-1),
      name_(field->name()),
      logical_type_(arrow::ToLogicalType(field->type()).ValueOrDie()),
      extension_name_(arrow::GetExtensionName(field->type()).value_or("")),
      encoding_(pb::NONE) {
  if (is_extension_type()) {
    auto ext_type = std::static_pointer_cast<::arrow::ExtensionType>(field->type());
    Init(ext_type->storage_type());
  } else {
    Init(field->type());
  }
}

void Field::Init(std::shared_ptr<::arrow::DataType> dtype) {
  if (::lance::arrow::is_struct(dtype)) {
    auto struct_type = std::static_pointer_cast<::arrow::StructType>(dtype);
    for (auto& arrow_field : struct_type->fields()) {
      children_.push_back(std::shared_ptr<Field>(new Field(arrow_field)));
    }
  } else if (::lance::arrow::is_list(dtype)) {
    auto list_type = std::static_pointer_cast<::arrow::ListType>(dtype);
    children_.emplace_back(
        std::shared_ptr<Field>(new Field(::arrow::field("item", list_type->value_type()))));
    encoding_ = pb::PLAIN;
  }

  auto type_id = dtype->id();
  if (::arrow::is_binary_like(type_id) || ::arrow::is_large_binary_like(type_id)) {
    encoding_ = pb::VAR_BINARY;
  } else if (::arrow::is_primitive(type_id) || ::arrow::is_fixed_size_binary(type_id) ||
             lance::arrow::is_fixed_size_list(dtype)) {
    encoding_ = pb::PLAIN;
  } else if (::arrow::is_dictionary(type_id)) {
    encoding_ = pb::DICTIONARY;
  }
}

Field::Field(const pb::Field& pb)
    : id_(pb.id()),
      parent_(pb.parent_id()),
      name_(pb.name()),
      logical_type_(pb.logical_type()),
      extension_name_(pb.extension_name()),
      encoding_(pb.encoding()),
      dictionary_offset_(pb.dictionary_offset()),
      dictionary_page_length_(pb.dictionary_page_length()) {}

void Field::AddChild(std::shared_ptr<Field> child) { children_.emplace_back(child); }

bool Field::RemoveChild(int32_t id) {
  for (auto it = children_.begin(); it != children_.end(); ++it) {
    if ((*it)->id() == id) {
      children_.erase(it);
      return true;
    }
    if ((*it)->RemoveChild(id)) {
      return true;
    }
  }
  return false;
}

// TODO: lets use a map to speed up this. for now it is just O(n) scan.
std::shared_ptr<Field> Field::Get(int32_t id) {
  for (auto& child : children_) {
    if (child->id_ == id) {
      return child;
    }
    auto found = child->Get(id);
    if (found) {
      return found;
    }
  }
  return std::shared_ptr<Field>();
}

std::shared_ptr<Field> Field::Get(const std::vector<std::string>& field_path,
                                  std::size_t start_idx) const {
  if (start_idx >= field_path.size()) {
    return nullptr;
  }
  if (lance::arrow::is_list(type())) {
    return children_[0]->Get(field_path, start_idx);
  }
  auto child = Get(field_path[start_idx]);
  if (!child || start_idx == field_path.size() - 1) {
    return child;
  }
  return child->Get(field_path, start_idx + 1);
}

std::shared_ptr<Field> Field::Get(const std::string_view& name) const {
  if (logical_type_ == "list.struct") {
    if (children_.empty()) {
      return nullptr;
    }
    return children_[0]->Get(name);
  }
  for (const auto& child : children_) {
    if (child->name_ == name) {
      return child;
    }
  }
  return nullptr;
}

::arrow::Status Field::Add(const pb::Field& pb) {
  children_.emplace_back(std::shared_ptr<Field>(new Field(pb)));
  return ::arrow::Status::OK();
}

std::string Field::ToString() const {
  if (is_extension_type()) {
    return fmt::format("{}({}): {}, encoding={}, extension_name={}",
                       name_,
                       id_,
                       type()->ToString(),
                       encoding_,
                       extension_name_);
  }
  return fmt::format("{}({}): {}, encoding={}", name_, id_, type()->ToString(), encoding_);
}

std::string Field::name() const {
  auto pos = name_.find_last_of('.');
  if (pos != std::string::npos) {
    return name_.substr(pos + 1);
  } else {
    return name_;
  }
}

void Field::set_encoding(lance::format::pb::Encoding encoding) { encoding_ = encoding; }

const std::shared_ptr<::arrow::Array>& Field::dictionary() const { return dictionary_; }

::arrow::Status Field::set_dictionary(std::shared_ptr<::arrow::Array> dict_arr) {
  if (!dictionary_) {
    dictionary_ = dict_arr;
    return ::arrow::Status::OK();
  }
  return ::arrow::Status::Invalid("Field::dictionary has already been set");
}

::arrow::Status Field::LoadDictionary(std::shared_ptr<::arrow::io::RandomAccessFile> infile) {
  assert(::arrow::is_dictionary(type()->id()));
  auto dict_type = std::dynamic_pointer_cast<::arrow::DictionaryType>(type());
  ///
  assert(dict_type->value_type()->Equals(::arrow::utf8()));

  auto decoder = lance::encodings::VarBinaryDecoder<::arrow::StringType>(infile, ::arrow::utf8());
  decoder.Reset(dictionary_offset_, dictionary_page_length_);

  ARROW_ASSIGN_OR_RAISE(auto dict_arr, decoder.ToArray());
  return set_dictionary(dict_arr);
}

int32_t Field::GetFieldsCount() const {
  return std::accumulate(
      std::begin(children_), std::end(children_), children_.size(), [](int32_t acc, auto& f) {
        return f->GetFieldsCount() + acc;
      });
}

std::shared_ptr<lance::encodings::Encoder> Field::GetEncoder(
    std::shared_ptr<::arrow::io::OutputStream> sink) {
  switch (encoding_) {
    case pb::Encoding::PLAIN:
      return std::make_shared<lance::encodings::PlainEncoder>(sink);
    case pb::Encoding::VAR_BINARY:
      return std::make_shared<lance::encodings::VarBinaryEncoder>(sink);
    case pb::Encoding::DICTIONARY:
      return std::make_shared<lance::encodings::DictionaryEncoder>(sink);
    default:
      fmt::print(stderr, "Encoding {} is not supported\n", encoding_);
      assert(false);
  }
  // Make compiler happy.
  return nullptr;
}

::arrow::Result<std::shared_ptr<lance::encodings::Decoder>> Field::GetDecoder(
    std::shared_ptr<::arrow::io::RandomAccessFile> infile) {
  std::shared_ptr<lance::encodings::Decoder> decoder;
  auto data_type = type();
  if (encoding() == pb::Encoding::PLAIN) {
    if (logical_type_ == "list" || logical_type_ == "list.struct") {
      decoder = std::make_shared<lance::encodings::PlainDecoder>(infile, ::arrow::int32());
    } else if (data_type->id() == ::arrow::TimestampType::type_id ||
               data_type->id() == ::arrow::Time64Type::type_id ||
               data_type->id() == ::arrow::Date64Type::type_id) {
      decoder = std::make_shared<lance::encodings::PlainDecoder>(infile, ::arrow::int64());
    } else if (data_type->id() == ::arrow::Time32Type::type_id ||
               data_type->id() == ::arrow::Date32Type::type_id) {
      decoder = std::make_shared<lance::encodings::PlainDecoder>(infile, ::arrow::int32());
    } else {
      decoder = std::make_shared<lance::encodings::PlainDecoder>(infile, type());
    }
  } else if (encoding_ == pb::Encoding::VAR_BINARY) {
    if (logical_type_ == "string") {
      decoder =
          std::make_shared<lance::encodings::VarBinaryDecoder<::arrow::StringType>>(infile, type());
    } else if (logical_type_ == "binary") {
      decoder =
          std::make_shared<lance::encodings::VarBinaryDecoder<::arrow::BinaryType>>(infile, type());
    }
  } else if (encoding_ == pb::Encoding::DICTIONARY) {
    auto dict_type = std::static_pointer_cast<::arrow::DictionaryType>(type());
    if (!dictionary()) {
      {
        std::scoped_lock lock(lock_);
        if (!dictionary()) {
          /// Fetch dictionary on demand?
          ARROW_RETURN_NOT_OK(LoadDictionary(infile));
        }
      }
    }
    decoder =
        std::make_shared<lance::encodings::DictionaryDecoder>(infile, dict_type, dictionary());
  }

  if (decoder) {
    auto status = decoder->Init();
    if (!status.ok()) {
      return status;
    }
    return decoder;
  } else {
    return ::arrow::Status::NotImplemented(
        fmt::format("Field::GetDecoder(): encoding={} logic_type={} is not supported.",
                    encoding(),
                    logical_type_));
  }
}

std::shared_ptr<::arrow::Field> Field::ToArrow() const {
  if (is_extension_type()) {
    auto ext_type = ::arrow::GetExtensionType(extension_name_);
    if (ext_type != nullptr) {
      return ::arrow::field(name(), ext_type);
    }
  }
  return ::arrow::field(name(), type());
}

std::vector<lance::format::pb::Field> Field::ToProto() const {
  std::vector<lance::format::pb::Field> pb_fields;

  lance::format::pb::Field field;
  field.set_name(name_);
  field.set_parent_id(parent_);
  field.set_id(id_);
  field.set_logical_type(logical_type_);
  field.set_extension_name(extension_name_);
  field.set_encoding(encoding_);
  field.set_dictionary_offset(dictionary_offset_);
  field.set_dictionary_page_length(dictionary_page_length_);
  field.set_type(GetNodeType());

  pb_fields.emplace_back(field);

  for (auto& child : children_) {
    auto protos = child->ToProto();
    pb_fields.insert(pb_fields.end(), protos.begin(), protos.end());
  }
  return pb_fields;
};

std::shared_ptr<::arrow::DataType> Field::type() const {
  if (logical_type_ == "list") {
    assert(children_.size() == 1);
    return ::arrow::list(children_[0]->type());
  } else if (logical_type_ == "list.struct") {
    assert(children_.size() == 1);
    return ::arrow::list(children_[0]->type());
  } else if (logical_type_ == "struct") {
    std::vector<std::shared_ptr<::arrow::Field>> sub_types;
    for (auto& child : children_) {
      sub_types.emplace_back(std::make_shared<::arrow::Field>(child->name(), child->type()));
    }
    return ::arrow::struct_(sub_types);
  } else {
    return lance::arrow::FromLogicalType(logical_type_).ValueOrDie();
  }
}

int32_t Field::id() const { return id_; }

void Field::SetId(int32_t parent_id, int32_t* current_id) {
  parent_ = parent_id;
  id_ = (*current_id);
  *current_id += 1;
  for (auto& child : children_) {
    child->SetId(id_, current_id);
  }
}

std::shared_ptr<Field> Field::Copy(bool include_children) const {
  auto new_field = make_shared<Field>();
  new_field->id_ = id_;
  new_field->parent_ = parent_;
  new_field->name_ = name_;
  new_field->logical_type_ = logical_type_;
  new_field->extension_name_ = extension_name_;
  new_field->encoding_ = encoding_;
  new_field->dictionary_offset_ = dictionary_offset_;
  new_field->dictionary_page_length_ = dictionary_page_length_;

  if (include_children) {
    for (const auto& child : children_) {
      new_field->children_.emplace_back(child->Copy(include_children));
    }
  }
  return new_field;
}

std::shared_ptr<Field> Field::Project(const std::shared_ptr<::arrow::Field>& arrow_field) const {
  assert(name_ == arrow_field->name());
  auto new_field = Copy();
  auto dtype = arrow_field->type();
  if (::lance::arrow::is_extension(dtype)) {
    auto ext_type = std::static_pointer_cast<::arrow::ExtensionType>(dtype);
    dtype = ext_type->storage_type();
  }
  if (arrow::is_struct(dtype)) {
    auto struct_type = std::dynamic_pointer_cast<::arrow::StructType>(dtype);
    for (auto arrow_subfield : struct_type->fields()) {
      auto subfield = Get(arrow_subfield->name());
      assert(subfield);
      new_field->AddChild(subfield->Project(arrow_subfield));
    }
  } else if (arrow::is_list(dtype)) {
    auto list_type = std::dynamic_pointer_cast<::arrow::ListType>(dtype);
    new_field->AddChild(children_[0]->Project(list_type->value_field()));
  }
  return new_field;
}

bool Field::Equals(const Field& other, bool check_id) const {
  if (check_id && (id_ != other.id_ || parent_ != other.parent_)) {
    return false;
  }
  if (name_ != other.name_ || logical_type_ != other.logical_type_ ||
      encoding_ != other.encoding_) {
    return false;
  }
  if (children_.size() != other.children_.size()) {
    return false;
  }
  for (std::size_t i = 0; i < children_.size(); i++) {
    if (!children_[i]->Equals(other.children_[i], check_id)) {
      return false;
    }
  }
  return true;
}

bool Field::Equals(const std::shared_ptr<Field>& other, bool check_id) const {
  if (!other) {
    return false;
  }
  return Equals(*other, check_id);
}

bool Field::operator==(const Field& other) const { return Equals(other, true); }

//------ Schema

Schema::Schema(const google::protobuf::RepeatedPtrField<::lance::format::pb::Field>& pb_fields) {
  for (auto& f : pb_fields) {
    auto field = std::make_shared<Field>(f);
    if (field->parent_id() < 0) {
      fields_.emplace_back(field);
    } else {
      auto parent = GetField(f.parent_id());
      assert(parent);
      parent->AddChild(field);
    }
  }
}

Schema::Schema(std::shared_ptr<::arrow::Schema> schema) {
  for (auto f : schema->fields()) {
    fields_.emplace_back(make_shared<Field>(f));
  }
  AssignIds();
}

::arrow::Status CopyField(std::shared_ptr<Field> new_field,
                          std::shared_ptr<Field> field,
                          std::vector<std::string> components,
                          std::size_t comp_idx) {
  if (comp_idx >= components.size() || !new_field || !field) {
    return ::arrow::Status::OK();
  }

  /// If this is a list<struct> node, we push the copy field into the child / struct node.
  if (field->logical_type() == "list.struct") {
    assert(field->children_.size() == 1);
    if (new_field->children_.empty()) {
      new_field->children_.emplace_back(field->children_[0]->Copy(false));
    }
    return CopyField(new_field->children_[0], field->children_[0], components, comp_idx);
  }

  const auto& name = components[comp_idx];
  auto new_child = new_field->Get(name);
  if (!new_child) {
    auto child = field->Get(name);
    if (!child) {
      return ::arrow::Status::Invalid(fmt::format("Invalid name {}[{}]", components, comp_idx));
    }
    new_child = child->Copy(components.size() - 1 == comp_idx);
    new_field->AddChild(new_child);
  }
  return CopyField(new_field->Get(name), field->Get(name), components, comp_idx + 1);
}

::arrow::Result<std::shared_ptr<Schema>> Schema::Project(
    const std::vector<std::string>& column_names) const {
  auto view = make_shared<Schema>();
  for (auto& name : column_names) {
    auto split_strs = ::arrow::internal::SplitString(name, '.');
    std::vector<std::string> components;
    for (auto& ss : split_strs) {
      components.emplace_back(ss);
    }
    if (components.empty()) {
      return ::arrow::Status::Invalid("Column name can not be empty.");
    }

    auto actual_field = GetField(components[0]);
    if (!actual_field) {
      return ::arrow::Status::Invalid("Field {} dose not exist.", name);
    }
    auto view_field = view->GetField(components[0]);
    if (!view_field) {
      view_field = actual_field->Copy(components.size() == 1);
      view->AddField(view_field);
    }
    ARROW_RETURN_NOT_OK(CopyField(view_field, actual_field, components, 1));
  }
  return view;
}

::arrow::Result<std::shared_ptr<Schema>> Schema::Project(
    const ::arrow::Schema& arrow_schema) const {
  auto projection = make_shared<Schema>();
  for (auto& arrow_field : arrow_schema.fields()) {
    auto field = GetField(arrow_field->name());
    if (!field) {
      return ::arrow::Status::Invalid(fmt::format("Field {} dose not exist", arrow_field->name()));
    }
    auto proj_field = field->Project(arrow_field);
    projection->AddField(proj_field);
  }
  return projection;
}

::arrow::Result<std::shared_ptr<Schema>> Schema::Project(
    const ::arrow::compute::Expression& expr) const {
  if (!::arrow::compute::ExpressionHasFieldRefs(expr)) {
    /// All scalar?
    return nullptr;
  }
  std::vector<std::string> columns;
  for (auto& ref : ::arrow::compute::FieldsInExpression(expr)) {
    columns.emplace_back(std::string(*ref.name()));
  }
  return Project(columns);
}

::arrow::Result<std::shared_ptr<Schema>> Schema::Exclude(std::shared_ptr<Schema> other) const {
  /// An visitor to remove fields in place.
  class SchemaExcludeVisitor : public FieldVisitor {
   public:
    SchemaExcludeVisitor(std::shared_ptr<Schema> excluded) : excluded_(excluded) {}

    ::arrow::Status Visit(std::shared_ptr<Field> field) override {
      for (auto& field : field->fields()) {
        ARROW_RETURN_NOT_OK(Visit(field));
      }
      auto excluded_field = excluded_->GetField(field->id());
      if (!excluded_field) {
        return ::arrow::Status::OK();
      }
      if (field->fields().empty() ||
          (excluded_field->GetNodeType() != pb::Field::LEAF && excluded_field->fields().empty())) {
        excluded_->RemoveField(field->id());
      }

      return ::arrow::Status::OK();
    }

   private:
    std::shared_ptr<Schema> excluded_;
  };

  auto excluded = Copy();
  auto visitor = SchemaExcludeVisitor(excluded);
  ARROW_RETURN_NOT_OK(visitor.VisitSchema(other));
  return excluded;
}

void Schema::AddField(std::shared_ptr<Field> f) { fields_.emplace_back(f); }

std::shared_ptr<Field> Schema::GetField(int32_t id) const {
  for (auto& field : fields_) {
    if (field->id() == id) {
      return field;
    }
    auto subfield = field->Get(id);
    if (subfield) {
      return subfield;
    }
  }
  return nullptr;
};

std::shared_ptr<Field> Schema::GetField(const std::string& name) const {
  vector<string> components;
  /// Convert arrow split string to strings.
  for (auto& sv : ::arrow::internal::SplitString(name, '.')) {
    components.emplace_back(sv);
  }
  for (auto& field : fields_) {
    if (field->name() == components[0]) {
      if (components.size() == 1) {
        return field;
      }
      return field->Get(components, 1);
    }
  }
  return nullptr;
}

int32_t Schema::GetFieldsCount() const {
  return std::accumulate(
      std::begin(fields_), std::end(fields_), fields_.size(), [](int32_t acc, auto& f) {
        return f->GetFieldsCount() + acc;
      });
}

std::vector<lance::format::pb::Field> Schema::ToProto() const {
  std::vector<lance::format::pb::Field> pb_fields;

  for (auto child : fields_) {
    auto protos = child->ToProto();
    pb_fields.insert(pb_fields.end(), protos.begin(), protos.end());
  }
  return pb_fields;
}

/// Make a full copy of the schema.
std::shared_ptr<Schema> Schema::Copy() const {
  auto copy = std::make_shared<Schema>();
  for (auto& field : fields_) {
    copy->fields_.emplace_back(field->Copy(true));
  };
  return copy;
}

void Schema::AssignIds() {
  int cur_id = 0;
  for (auto& field : fields_) {
    field->SetId(-1, &cur_id);
  }
}

bool Schema::RemoveField(int32_t id) {
  for (auto it = fields_.begin(); it != fields_.end(); ++it) {
    if ((*it)->id() == id) {
      fields_.erase(it);
      return true;
    }
    if ((*it)->RemoveChild(id)) {
      return true;
    }
  }
  return false;
}

bool Schema::Equals(const Schema& other, bool check_id) const {
  if (fields_.size() != other.fields_.size()) {
    return false;
  }
  for (std::size_t i = 0; i < fields_.size(); ++i) {
    if (!fields_[i]->Equals(other.fields_[i], check_id)) {
      return false;
    }
  }
  return true;
}

bool Schema::Equals(const std::shared_ptr<Schema>& other, bool check_id) const {
  if (!other) {
    return false;
  }
  return Equals(*other, check_id);
}

bool Schema::operator==(const Schema& other) const { return Equals(other, true); };

std::string Schema::ToString() const {
  vector<string> fields_strs;
  fields_strs.reserve(fields_.size());
  for (auto& f : fields_) {
    fields_strs.emplace_back(f->ToString());
  }
  return ::arrow::internal::JoinStrings(fields_strs, "\n");
}

std::shared_ptr<::arrow::Schema> Schema::ToArrow() const {
  vector<std::shared_ptr<::arrow::Field>> arrow_fields;
  for (auto f : fields_) {
    arrow_fields.emplace_back(f->ToArrow());
  }
  return ::arrow::schema(arrow_fields);
}

pb::Field::Type Field::GetNodeType() const {
  if (logical_type_ == "struct") {
    return pb::Field::PARENT;
  } else if (logical_type_ == "list.struct" || logical_type_ == "list") {
    return pb::Field::REPEATED;
  } else {
    return pb::Field::LEAF;
  }
}

}  // namespace lance::format
