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

#pragma once

#include <arrow/scalar.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <concepts>
#include <memory>
#include <optional>
#include <string>
#include <vector>

template <typename T>
concept HasToString = requires(T t) {
                        { t.ToString() } -> std::same_as<std::string>;
                      };

template <HasToString T>
struct fmt::formatter<T> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const T& v, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", v.ToString());
  }
};

template <HasToString T>
struct fmt::formatter<std::shared_ptr<T>> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const std::shared_ptr<T>& v, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", v->ToString());
  }
};

namespace lance::arrow {

/// Returns True if the data type is a list.
inline auto is_list(const std::shared_ptr<::arrow::DataType>& dtype) {
  return dtype->id() == ::arrow::Type::LIST || dtype->id() == ::arrow::Type::LARGE_LIST;
}

/// Returns True if the data type is a struct.
inline bool is_struct(::arrow::Type::type type_id) { return type_id == ::arrow::Type::STRUCT; }

/// Returns True if the data type is a struct.
inline bool is_struct(const std::shared_ptr<::arrow::DataType>& dtype) {
  return is_struct(dtype->id());
}

/// Returns True if the data type is a map.
inline bool is_map(const std::shared_ptr<::arrow::DataType>& dtype) {
  return dtype->id() == ::arrow::Type::MAP;
}

/// Returns True if the data type is timestamp type.
inline bool is_timestamp(const std::shared_ptr<::arrow::DataType>& dtype) {
  return dtype->id() == ::arrow::TimestampType::type_id;
}

inline bool is_extension(const std::shared_ptr<::arrow::DataType>& dtype) {
  return dtype->id() == ::arrow::Type::EXTENSION;
}

inline bool is_fixed_size_list(::arrow::Type::type type_id) {
  return type_id == ::arrow::Type::FIXED_SIZE_LIST;
}

inline bool is_fixed_size_list(const std::shared_ptr<::arrow::DataType>& dtype) {
  return is_fixed_size_list(dtype->id());
}

/// Returns True if every value has the same length.
inline bool is_fixed_length(::arrow::Type::type type_id) {
  return ::arrow::is_primitive(type_id) || ::arrow::is_binary_like(type_id) ||
         ::arrow::is_large_binary_like(type_id) || ::arrow::is_fixed_size_binary(type_id) ||
         is_fixed_size_list(type_id);
}

inline bool is_fixed_length(const std::shared_ptr<::arrow::DataType>& data_type) {
  return is_fixed_length(data_type->id());
}

::arrow::Result<std::shared_ptr<::arrow::ArrayBuilder>> GetArrayBuilder(
    const std::shared_ptr<::arrow::DataType>& dtype,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

/// Convert arrow DataType to a string representation.
::arrow::Result<std::string> ToLogicalType(std::shared_ptr<::arrow::DataType> dtype);

::arrow::Result<std::shared_ptr<::arrow::DataType>> FromLogicalType(
    ::arrow::util::string_view logical_type);

std::optional<std::string> GetExtensionName(std::shared_ptr<::arrow::DataType> dtype);

}  // namespace lance::arrow
