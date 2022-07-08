#pragma once

#include <arrow/scalar.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <memory>
#include <string>
#include <vector>

#include "lance/format/format.pb.h"

template <>
struct fmt::formatter<::arrow::Scalar> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const ::arrow::Scalar& scalar, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", scalar.ToString());
  }
};

template <>
struct fmt::formatter<std::shared_ptr<::arrow::Scalar>> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const std::shared_ptr<::arrow::Scalar>& scalar, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", scalar->ToString());
  }
};

template <>
struct fmt::formatter<::arrow::Array> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const ::arrow::Array& arr, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", arr.ToString());
  }
};

template <>
struct fmt::formatter<std::shared_ptr<::arrow::Array>> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const std::shared_ptr<::arrow::Array>& arr, FormatContext& ctx) -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", arr->ToString());
  }
};

namespace lance::arrow {

/// Returns True if the data type is a list.
inline auto is_list(const std::shared_ptr<::arrow::DataType>& dtype) {
  return dtype->id() == ::arrow::Type::LIST || dtype->id() == ::arrow::Type::LARGE_LIST;
}

/// Returns True if the data type is a struct.
inline bool is_struct(const std::shared_ptr<::arrow::DataType>& dtype) {
  return dtype->id() == ::arrow::Type::STRUCT;
}

/// Returns True if the data type si a map.
inline bool is_map(std::shared_ptr<::arrow::DataType> dtype) {
  return dtype->id() == ::arrow::Type::MAP;
}

::arrow::Result<::lance::format::pb::DataType> ToPhysicalType(
    std::shared_ptr<::arrow::DataType> dtype);

::arrow::Result<std::string> ToLogicalType(std::shared_ptr<::arrow::DataType> dtype);

::arrow::Result<std::shared_ptr<::arrow::DataType>> FromLogicalType(
    const std::string& logical_type);

/// Convert arrow schema.
::arrow::Result<std::vector<lance::format::pb::Field>> FromArrowSchema(
    std::shared_ptr<::arrow::Schema> schema);

::arrow::Result<std::shared_ptr<::arrow::Schema>> ToArrowSchema(
    const std::vector<lance::format::pb::Field>& fields);

}  // namespace lance::arrow
