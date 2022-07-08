#include "lance/format/schema.h"

#include <arrow/util/string.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <memory>
#include <string>
#include <vector>

#include "lance/arrow/type.h"
#include "lance/encodings/binary.h"
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
      physical_type_(arrow::ToPhysicalType(field->type()).ValueOrDie()),
      logical_type_(arrow::ToLogicalType(field->type()).ValueOrDie()),
      encoding_(pb::NONE) {
  if (::lance::arrow::is_struct(field->type())) {
    auto struct_type = std::static_pointer_cast<::arrow::StructType>(field->type());
    for (auto& arrow_field : struct_type->fields()) {
      children_.push_back(std::shared_ptr<Field>(new Field(arrow_field)));
    }
  } else if (::lance::arrow::is_list(field->type())) {
    auto list_type = std::static_pointer_cast<::arrow::ListType>(field->type());
    children_.emplace_back(
        std::shared_ptr<Field>(new Field(::arrow::field("item", list_type->value_type()))));
    encoding_ = pb::PLAIN;
  }

  if (::arrow::is_binary_like(field->type()->id())) {
    encoding_ = pb::VAR_BINARY;
  } else if (::arrow::is_primitive(field->type()->id())) {
    encoding_ = pb::PLAIN;
  }
}

Field::Field(const pb::Field& pb)
    : id_(pb.id()),
      parent_(pb.parent_id()),
      name_(pb.name()),
      physical_type_(pb.data_type()),
      logical_type_(pb.logical_type()),
      encoding_(pb.encoding()) {
}

void Field::AddChild(std::shared_ptr<Field> child) { children_.emplace_back(child); }

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

const std::shared_ptr<Field> Field::Get(const std::vector<std::string>& field_path,
                                        std::size_t start_idx) const {
  if (start_idx >= field_path.size()) {
    return nullptr;
  }
  auto child = Get(field_path[start_idx]);
  if (!child || start_idx == field_path.size() - 1) {
    return child;
  }
  return child->Get(field_path, start_idx + 1);
}

const std::shared_ptr<Field> Field::Get(const std::string_view& name) const {
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

std::shared_ptr<lance::encodings::Encoder> Field::GetEncoder(
    std::shared_ptr<::arrow::io::OutputStream> sink) {
  switch (encoding_) {
    case pb::Encoding::PLAIN:
      return std::make_shared<lance::encodings::PlainEncoder>(sink);
    case pb::Encoding::VAR_BINARY:
      return std::make_shared<lance::encodings::VarBinaryEncoder>(sink);
    default:
      fmt::print(stderr, "Encoding {} is not supported", encoding_);
      assert(false);
  }
}

::arrow::Result<std::shared_ptr<lance::encodings::Decoder>> Field::GetDecoder(
    std::shared_ptr<::arrow::io::RandomAccessFile> infile) {
  if (encoding() == pb::Encoding::PLAIN) {
    if (logical_type_ == "int32" || logical_type_ == "list" || logical_type_ == "list.struct") {
      return std::make_shared<lance::encodings::PlainDecoder<::arrow::Int32Type>>(infile);
    } else if (logical_type_ == "int64") {
      return std::make_shared<lance::encodings::PlainDecoder<::arrow::Int64Type>>(infile);
    } else if (logical_type_ == "float") {
      return std::make_shared<lance::encodings::PlainDecoder<::arrow::FloatType>>(infile);
    } else if (logical_type_ == "double") {
      return std::make_shared<lance::encodings::PlainDecoder<::arrow::DoubleType>>(infile);
    }
  } else if (encoding_ == pb::Encoding::VAR_BINARY) {
    if (logical_type_ == "string") {
      return std::make_shared<lance::encodings::VarBinaryDecoder<::arrow::StringType>>(infile);
    } else if (logical_type_ == "binary") {
      return std::make_shared<lance::encodings::VarBinaryDecoder<::arrow::BinaryType>>(infile);
    }
  }
  return ::arrow::Status::NotImplemented(fmt::format(
      "Field::GetDecoder(): encoding={} logic_type={} is not supported.", "plain", logical_type_));
}

std::shared_ptr<::arrow::Field> Field::ToArrow() const { return ::arrow::field(name(), type()); }

std::vector<lance::format::pb::Field> Field::ToProto() const {
  std::vector<lance::format::pb::Field> pb_fields;

  lance::format::pb::Field field;
  field.set_name(name_);
  field.set_parent_id(parent_);
  field.set_id(id_);
  field.set_logical_type(logical_type_);
  field.set_data_type(physical_type_);
  field.set_encoding(encoding_);

  if (logical_type_ == "struct") {
    field.set_type(pb::Field::PARENT);
  } else if (logical_type_ == "list.struct" || logical_type_ == "list") {
    field.set_type(pb::Field::REPEATED);
  } else {
    field.set_type(pb::Field::LEAF);
  }
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

void Field::SetId(int32_t parent_id,
                  int32_t* current_id,
                  std::map<int32_t, std::shared_ptr<Field>>* field_map) {
  assert(field_map != nullptr);
  SetId(parent_id, current_id);
  for (auto& child : children_) {
    field_map->emplace(id_, child);
  }
}

std::shared_ptr<Field> Field::Copy(bool include_children) const {
  auto new_field = make_shared<Field>();
  new_field->id_ = id_;
  new_field->parent_ = parent_;
  new_field->name_ = name_;
  new_field->physical_type_ = physical_type_;
  new_field->logical_type_ = logical_type_;
  new_field->encoding_ = encoding_;

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
  if (arrow::is_struct(arrow_field->type())) {
    auto struct_type = std::dynamic_pointer_cast<::arrow::StructType>(arrow_field->type());
    for (auto arrow_subfield : struct_type->fields()) {
      auto subfield = Get(arrow_subfield->name());
      assert(subfield);
      new_field->AddChild(subfield->Project(arrow_subfield));
    }
  }
  return new_field;
}

bool Field::Equals(const Field& other, bool check_id) const {
  if (check_id && (id_ != other.id_ || parent_ != other.parent_)) {
    return false;
  }
  if (name_ != other.name_ || physical_type_ != other.physical_type_ ||
      logical_type_ != other.logical_type_ || encoding_ != other.encoding_) {
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
  return std::make_shared<::arrow::Field>(node->name(), node->type());
}

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

    fields_map_.emplace(field->id(), field);
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

void Schema::AddField(std::shared_ptr<Field> f) { fields_.emplace_back(f); }

std::shared_ptr<Field> Schema::GetField(int32_t id) const {
  if (auto it = fields_map_.find(id); it != fields_map_.end()) {
    return it->second;
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

std::vector<lance::format::pb::Field> Schema::ToProto() const {
  std::vector<lance::format::pb::Field> pb_fields;

  for (auto child : fields_) {
    auto protos = child->ToProto();
    pb_fields.insert(pb_fields.end(), protos.begin(), protos.end());
  }
  return pb_fields;
}

void Schema::AssignIds() {
  int cur_id = 0;
  fields_map_.clear();
  for (auto& field : fields_) {
    field->SetId(-1, &cur_id, &fields_map_);
    fields_map_[field->id()] = field;
  }
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

}  // namespace lance::format
