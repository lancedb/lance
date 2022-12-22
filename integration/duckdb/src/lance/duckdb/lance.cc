// Copyright 2022 Lance Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "lance/duckdb/lance.h"

#include <arrow/type.h>

#include <duckdb/common/exception.hpp>
#include <vector>

namespace lance::duckdb {

namespace {

inline ::duckdb::LogicalType ToLogicalType(const ::arrow::DictionaryType& dtype) {
  return lance::duckdb::ToLogicalType(*dtype.value_type());
}

inline ::duckdb::LogicalType ToLogicalType(const ::arrow::StructType& struct_type) {
  ::duckdb::child_list_t<::duckdb::LogicalType> children;
  for (auto& child : struct_type.fields()) {
    children.emplace_back(
        std::make_pair(child->name(), lance::duckdb::ToLogicalType(*child->type())));
  }
  return ::duckdb::LogicalType::STRUCT(children);
}

template <typename L>
inline ::duckdb::LogicalType ToLogicalType(const ::arrow::DataType& dtype) {
  auto& list_type = dynamic_cast<const L&>(dtype);
  auto child_type = lance::duckdb::ToLogicalType(*list_type.value_type());
  return ::duckdb::LogicalType::LIST(child_type);
}

}  // namespace

::duckdb::LogicalType ToLogicalType(const ::arrow::DataType& arrow_type) {
  switch (arrow_type.id()) {
    case ::arrow::Type::BOOL:
      return ::duckdb::LogicalType::BOOLEAN;
    case ::arrow::Type::INT8:
      return ::duckdb::LogicalType::TINYINT;
    case ::arrow::Type::UINT8:
      return ::duckdb::LogicalType::UTINYINT;
    case ::arrow::Type::INT16:
      return ::duckdb::LogicalType::SMALLINT;
    case ::arrow::Type::UINT16:
      return ::duckdb::LogicalType::USMALLINT;
    case ::arrow::Type::INT32:
      return ::duckdb::LogicalType::INTEGER;
    case ::arrow::Type::UINT32:
      return ::duckdb::LogicalType::UINTEGER;
    case ::arrow::Type::INT64:
      return ::duckdb::LogicalType::BIGINT;
    case ::arrow::Type::UINT64:
      return ::duckdb::LogicalType::UBIGINT;
    case ::arrow::Type::FLOAT:
    case ::arrow::Type::HALF_FLOAT:
      return ::duckdb::LogicalType::FLOAT;
    case ::arrow::Type::DOUBLE:
      return ::duckdb::LogicalType::DOUBLE;
    case ::arrow::Type::STRING:
    case ::arrow::Type::LARGE_STRING:
      return ::duckdb::LogicalType::VARCHAR;
    case ::arrow::Type::BINARY:
    case ::arrow::Type::LARGE_BINARY:
      return ::duckdb::LogicalType::BLOB;
    case ::arrow::Type::TIME32:
    case ::arrow::Type::TIME64:
      return ::duckdb::LogicalType::TIME;
    case ::arrow::Type::TIMESTAMP:
      return ::duckdb::LogicalType::TIMESTAMP;
    case ::arrow::Type::DATE32:
    case ::arrow::Type::DATE64:
      return ::duckdb::LogicalType::DATE;
    case ::arrow::Type::DICTIONARY:
      return ToLogicalType(dynamic_cast<const ::arrow::DictionaryType&>(arrow_type));
    case ::arrow::Type::STRUCT:
      return ToLogicalType(dynamic_cast<const ::arrow::StructType&>(arrow_type));
    case ::arrow::Type::LIST:
      return ToLogicalType<::arrow::ListType>(arrow_type);
    case ::arrow::Type::FIXED_SIZE_LIST:
      return ToLogicalType<::arrow::FixedSizeListType>(arrow_type);
    default:
      throw ::duckdb::InvalidInputException("Does not support type: %s",
                                            arrow_type.ToString().c_str());
  }
}

}  // namespace lance::duckdb