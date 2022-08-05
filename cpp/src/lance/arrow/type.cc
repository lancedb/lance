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

const static std::map<std::string, std::shared_ptr<::arrow::DataType>> kPrimitiveTypeMap = {
    {"null", ::arrow::null()},
    {"bool", ::arrow::boolean()},
    {"int8", ::arrow::int8()},
    {"uint8", ::arrow::uint8()},
    {"int16", ::arrow::int16()},
    {"uint16", ::arrow::uint16()},
    {"int32", ::arrow::int32()},
    {"uint32", ::arrow::uint32()},
    {"int64", ::arrow::int64()},
    {"uint64", ::arrow::uint64()},
    {"halffloat", ::arrow::float16()},
    {"float", ::arrow::float32()},
    {"double", ::arrow::float64()},
    {"string", ::arrow::utf8()},
    {"binary", ::arrow::binary()},
    {"large_string", ::arrow::large_utf8()},
    {"large_binary", ::arrow::large_binary()},
};

::arrow::Result<std::shared_ptr<::arrow::DataType>> FromLogicalType(
    ::arrow::util::string_view logical_type) {
  // TODO: optimize this lookup table?
  const auto& it = kPrimitiveTypeMap.find(logical_type.to_string());
  if (it != kPrimitiveTypeMap.end()) {
    return it->second;
  }
  if (logical_type.starts_with("dict")) {
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

}  // namespace lance::arrow