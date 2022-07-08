#include "lance/arrow/type.h"

#include <arrow/result.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <memory>

#include "lance/format/schema.h"

namespace lance::arrow {

namespace pb = ::lance::format::pb;

::arrow::Result<::lance::format::pb::DataType> ToPhysicalType(
    std::shared_ptr<::arrow::DataType> dtype) {
  if (dtype->Equals(::arrow::boolean())) {
    return pb::BOOLEAN;
  }
  if (::arrow::is_integer(dtype->id())) {
    switch (::arrow::bit_width(dtype->id())) {
      case 8:
      case 16:
      case 32:
        return pb::INT32;
      case 64:
        return pb::INT64;
    };
  } else if (::arrow::is_floating(dtype->id())) {
    switch (::arrow::bit_width(dtype->id())) {
      case 16:
      case 32:
        return pb::FLOAT32;
      case 64:
        return pb::FLOAT64;
    }
  } else if (::arrow::is_binary_like(dtype->id())) {
    return pb::BYTES;
  } else if (is_list(dtype)) {
    // Offset type
    return pb::INT32;
  } else if (is_struct(dtype)) {
    // We do not actually store it.
    return pb::INT32;
  } else if (::arrow::is_dictionary(dtype->id())) {
    return pb::INT32;
  }
  return ::arrow::Status::NotImplemented(
      fmt::format("data type {} is not supported yet", dtype->ToString()));
}

::arrow::Result<std::string> ToLogicalType(std::shared_ptr<::arrow::DataType> dtype) {
  if (is_list(dtype)) {
    auto list_type = std::reinterpret_pointer_cast<::arrow::ListType>(dtype);
    return is_struct(list_type->value_type()) ? "list.struct" : "list";
  } else if (is_struct(dtype)) {
    return "struct";
  } else {
    return dtype->ToString();
  }
}

::arrow::Result<std::shared_ptr<::arrow::DataType>> FromLogicalType(
    const std::string& logical_type) {
  // TODO: optimize this lookup table?
  if (logical_type == "bool") {
    return ::arrow::boolean();
  } else if (logical_type == "int32") {
    return ::arrow::int32();
  } else if (logical_type == "int64") {
    return ::arrow::int64();
  } else if (logical_type == "float") {
    return ::arrow::float32();
  } else if (logical_type == "double") {
    return ::arrow::float64();
  } else if (logical_type == "string") {
    return ::arrow::utf8();
  } else if (logical_type == "binary") {
    return ::arrow::binary();
  }
  return ::arrow::Status::NotImplemented(
      fmt::format("FromLogicalType: logical_type \"{}\" is not supported yet", logical_type));
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