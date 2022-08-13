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

namespace {

std::string ToString(::arrow::TimeUnit::type unit) {
  using TimeUnit = ::arrow::TimeUnit;
  switch (unit) {
    case TimeUnit::SECOND:
      return "s";
    case TimeUnit::MILLI:
      return "ms";
    case TimeUnit::MICRO:
      return "us";
    case TimeUnit::NANO:
      return "ns";
    default:
      assert(false);
      return "";
  }
}

}  // namespace

::arrow::Result<std::string> ToLogicalType(std::shared_ptr<::arrow::DataType> dtype) {
  if (dtype->id() == ::arrow::Type::EXTENSION) {
    auto ext_type = std::static_pointer_cast<::arrow::ExtensionType>(dtype);
    return ToLogicalType(ext_type->storage_type());
  }

  if (is_list(dtype)) {
    auto list_type = std::reinterpret_pointer_cast<::arrow::ListType>(dtype);
    return is_struct(list_type->value_type()) ? "list.struct" : "list";
  } else if (is_struct(dtype)) {
    return "struct";
  } else if (::arrow::is_fixed_size_binary(dtype->id())) {
    auto fixed_type = std::reinterpret_pointer_cast<::arrow::FixedSizeBinaryType>(dtype);
    return fmt::format("fixed_size_binary:{}", fixed_type->byte_width());
  } else if (dtype->id() == ::arrow::Date32Type::type_id) {
    return "date32:day";
  } else if (dtype->id() == ::arrow::Date64Type::type_id) {
    return "date64:ms";
  } else if (dtype->id() == ::arrow::Time32Type::type_id) {
    auto time32 = std::dynamic_pointer_cast<::arrow::Time32Type>(dtype);
    return fmt::format("time32:{}", ToString(time32->unit()));
  } else if (dtype->id() == ::arrow::Time64Type::type_id) {
    auto time64 = std::dynamic_pointer_cast<::arrow::Time64Type>(dtype);
    return fmt::format("time64:{}", ToString(time64->unit()));
  } else if (is_timestamp(dtype)) {
    auto timestamp_type = std::dynamic_pointer_cast<::arrow::TimestampType>(dtype);
    return fmt::format("timestamp:{}", ToString(timestamp_type->unit()));
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
    {"date32:day", ::arrow::date32()},
    {"date64:ms", ::arrow::date64()},
};

::arrow::Result<::arrow::TimeUnit::type> TimeUnitFromLogicalType(
    const ::arrow::util::string_view& unit) {
  using Unit = ::arrow::TimeUnit;
  if (unit == "s") {
    return Unit::SECOND;
  } else if (unit == "ms") {
    return Unit::MILLI;
  } else if (unit == "us") {
    return Unit::MICRO;
  } else if (unit == "ns") {
    return Unit::NANO;
  };
  return ::arrow::Status::Invalid(fmt::format("Unsupported TimeUnit: {}", unit.to_string()));
}

::arrow::Result<std::shared_ptr<::arrow::DataType>> TimeFromLogicalType(
    const ::arrow::util::string_view& logical_type) {
  auto components = ::arrow::internal::SplitString(logical_type, ':');
  if (components.size() != 2) {
    return ::arrow::Status::Invalid(
        fmt::format("Invalid timestamp string: {}", logical_type.to_string()));
  }
  ARROW_ASSIGN_OR_RAISE(auto unit, TimeUnitFromLogicalType(components[1]));
  if (components[0] == "timestamp") {
    return ::arrow::timestamp(unit);
  } else if (components[0] == "time32") {
    return ::arrow::time32(unit);
  } else if (components[0] == "time64") {
    return ::arrow::time64(unit);
  }
  return ::arrow::Status::Invalid(
      fmt::format("Invalid temporal logical type: {}", logical_type.to_string()));
};

::arrow::Result<std::shared_ptr<::arrow::DataType>> FromLogicalType(
    ::arrow::util::string_view logical_type) {
  const auto& it = kPrimitiveTypeMap.find(logical_type.to_string());
  if (it != kPrimitiveTypeMap.end()) {
    return it->second;
  }

  if (logical_type.starts_with("time")) {
    return TimeFromLogicalType(logical_type);
  }

  if (logical_type.starts_with("fixed_size_binary")) {
    auto components = ::arrow::internal::SplitString(logical_type, ':');
    if (components.size() != 2) {
      return ::arrow::Status::Invalid(
          fmt::format("Invalid fixed size binary string: {}", logical_type.to_string()));
    }
    auto size = std::stoi(components[1].to_string());
    if (size == 0) {
      return ::arrow::Status::Invalid(
          fmt::format("Invalid fixe size binary string: {}", logical_type.to_string()));
    }
    return ::arrow::fixed_size_binary(size);
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


std::optional<std::string> GetExtensionName(std::shared_ptr<::arrow::DataType> dtype) {
  if (dtype->id() == ::arrow::Type::EXTENSION) {
    auto ext_type = std::static_pointer_cast<::arrow::ExtensionType>(dtype);
    return ext_type->extension_name();
  }
  return std::nullopt;
}

}  // namespace lance::arrow