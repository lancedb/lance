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

#include "lance/arrow/type.h"

#include <arrow/result.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <arrow/util/string.h>
#include <fmt/format.h>

#include <memory>

#include "lance/format/schema.h"

namespace lance::arrow {

namespace pb = ::lance::format::pb;

::arrow::Result<std::string> ToLogicalType(std::shared_ptr<::arrow::DataType> dtype) {
  if (is_list(dtype)) {
    auto list_type = std::reinterpret_pointer_cast<::arrow::ListType>(dtype);
    return is_struct(list_type->value_type()) ? "list.struct" : "list";
  } else if (is_struct(dtype)) {
    return "struct";
  } else if (::arrow::is_dictionary(dtype->id())) {
    auto dict_type = std::dynamic_pointer_cast<::arrow::DictionaryType>(dtype);
    return fmt::format("dict:{}:{}:{}",
                       dict_type->value_type()->ToString(),
                       dict_type->index_type()->ToString(),
                       dict_type->ordered());
  } else {
    return dtype->ToString();
  }
}

::arrow::Result<std::shared_ptr<::arrow::DataType>> FromLogicalType(
    ::arrow::util::string_view logical_type) {
  // TODO: optimize this lookup table?
  if (logical_type == "bool") {
    return ::arrow::boolean();
  } else if (logical_type == "int8") {
    return ::arrow::int8();
  } else if (logical_type == "uint8") {
    return ::arrow::utf8();
  } else if (logical_type == "int16") {
    return ::arrow::int16();
  } else if (logical_type == "uint16") {
    return ::arrow::uint16();
  } else if (logical_type == "int32") {
    return ::arrow::int32();
  } else if (logical_type == "uint32") {
    return ::arrow::uint32();
  } else if (logical_type == "int64") {
    return ::arrow::int64();
  } else if (logical_type == "uint64") {
    return ::arrow::uint64();
  } else if (logical_type == "float") {
    return ::arrow::float32();
  } else if (logical_type == "double") {
    return ::arrow::float64();
  } else if (logical_type == "string") {
    return ::arrow::utf8();
  } else if (logical_type == "binary") {
    return ::arrow::binary();
  } else if (logical_type.starts_with("dict")) {
    auto components = ::arrow::internal::SplitString(logical_type, ':');
    if (components.size() != 4) {
      return ::arrow::Status::Invalid(
          fmt::format("Invalid dictionary type string: {}", logical_type.to_string()));
    }
    ARROW_ASSIGN_OR_RAISE(auto value_type, FromLogicalType(components[1]));
    ARROW_ASSIGN_OR_RAISE(auto index_value, FromLogicalType(components[2]));
    auto ordered = components[3] == "true";
    return ::arrow::dictionary(index_value, value_type, ordered);
  }
  return ::arrow::Status::NotImplemented(fmt::format(
      "FromLogicalType: logical_type \"{}\" is not supported yet", logical_type.to_string()));
}

::arrow::Result<std::vector<lance::format::pb::Field>> FromArrowSchema(
    std::shared_ptr<::arrow::Schema> schema) {
  auto root = format::Schema(schema);
  return root.ToProto();
}

::arrow::Result<std::shared_ptr<::arrow::Schema>> ToArrowSchema(
    const std::vector<lance::format::pb::Field>& fields) {
  // TODO: better to build schema tree first.
  std::vector<std::shared_ptr<lance::format::Field>> field_map;
  auto root = std::make_shared<lance::format::Field>();
  for (auto& pb_field : fields) {
    auto parent = root;
    if (pb_field.parent_id() >= 0) {
      parent = root->Get(pb_field.parent_id());
    }
    if (!parent) {
      return ::arrow::Status::Invalid(fmt::format("Not parent found for {}", pb_field.parent_id()));
    }
    ARROW_RETURN_NOT_OK(parent->Add(pb_field));
  }

  format::ToArrowVisitor visitor;
  ARROW_RETURN_NOT_OK(visitor.Visit(root));
  return visitor.Finish();
}

}  // namespace lance::arrow