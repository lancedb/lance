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

#include "lance/arrow/utils.h"

#include <arrow/builder.h>
#include <arrow/dataset/discovery.h>
#include <arrow/filesystem/api.h>
#include <arrow/result.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <uuid.h>

#include <algorithm>
#include <concepts>
#include <random>
#include <range/v3/action.hpp>
#include <range/v3/view.hpp>
#include <string>
#include <vector>

#include "lance/arrow/file_lance.h"
#include "lance/arrow/type.h"

namespace views = ranges::views;

namespace lance::arrow {

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> MergeRecordBatches(
    const std::shared_ptr<::arrow::RecordBatch>& lhs,
    const std::shared_ptr<::arrow::RecordBatch>& rhs,
    ::arrow::MemoryPool* pool) {
  ARROW_ASSIGN_OR_RAISE(auto left_struct, lhs->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto right_struct, rhs->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto struct_arr, MergeStructArrays(left_struct, right_struct, pool));
  return ::arrow::RecordBatch::FromStructArray(struct_arr);
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> MergeRecordBatches(
    const std::vector<std::shared_ptr<::arrow::RecordBatch>>& batches, ::arrow::MemoryPool* pool) {
  if (batches.empty()) {
    return nullptr;
  }
  auto batch = batches[0];
  for (auto& b : batches | views::drop(1)) {
    if (b->num_rows() != batch->num_rows()) {
      return ::arrow::Status::Invalid(
          "MergeRecordBatches: attempt to merge batches with different length: ",
          b->num_rows(),
          " != ",
          batch->num_rows());
    }
    ARROW_ASSIGN_OR_RAISE(batch, MergeRecordBatches(batch, b, pool));
  }
  return batch;
}

::arrow::Result<std::shared_ptr<::arrow::Array>> MergeListArrays(
    const std::shared_ptr<::arrow::Array>& lhs,
    const std::shared_ptr<::arrow::Array>& rhs,
    ::arrow::MemoryPool* pool) {
  assert(is_list(lhs->type()) && is_list(rhs->type()));
  auto left_list_type = std::static_pointer_cast<::arrow::ListType>(lhs->type());
  auto right_list_type = std::static_pointer_cast<::arrow::ListType>(rhs->type());
  if (!is_struct(left_list_type->value_type()) || !is_struct(right_list_type->value_type())) {
    return ::arrow::Status::Invalid(fmt::format(
        "Can only merge list of structs: left={} right={}", left_list_type, right_list_type));
  }
  auto left_list = std::static_pointer_cast<::arrow::ListArray>(lhs);
  auto right_list = std::static_pointer_cast<::arrow::ListArray>(rhs);
  auto left_values = std::static_pointer_cast<::arrow::StructArray>(left_list->values());
  auto right_values = std::static_pointer_cast<::arrow::StructArray>(right_list->values());
  ARROW_ASSIGN_OR_RAISE(auto values, MergeStructArrays(left_values, right_values, pool));
  if (!left_list->offsets()->Equals(right_list->offsets())) {
    return ::arrow::Status::Invalid("Attempt to merge two lists with different offsets");
  }
  return ::arrow::ListArray::FromArrays(*left_list->offsets(), *values, pool);
}

::arrow::Result<std::shared_ptr<::arrow::StructArray>> MergeStructArrays(
    const std::shared_ptr<::arrow::StructArray>& lhs,
    const std::shared_ptr<::arrow::StructArray>& rhs,
    ::arrow::MemoryPool* pool) {
  if (lhs->length() != rhs->length()) {
    return ::arrow::Status::Invalid("Two StructArrays have different length");
  }
  std::vector<std::string> names;
  ::arrow::ArrayVector arrays;
  auto left_type = lhs->struct_type();
  for (auto& field : left_type->fields()) {
    auto& name = field->name();
    names.emplace_back(name);
    auto left_arr = lhs->GetFieldByName(name);

    auto right_arr = rhs->GetFieldByName(name);
    if (right_arr) {
      if (is_struct(left_arr->type()) && is_struct(right_arr->type())) {
        ARROW_ASSIGN_OR_RAISE(
            left_arr,
            MergeStructArrays(std::static_pointer_cast<::arrow::StructArray>(left_arr),
                              std::static_pointer_cast<::arrow::StructArray>(right_arr),
                              pool));
      } else if (is_list(left_arr->type()) && is_list(right_arr->type())) {
        ARROW_ASSIGN_OR_RAISE(left_arr, MergeListArrays(left_arr, right_arr, pool));
      } else {
        return ::arrow::Status::Invalid(
            fmt::format("Dose not support merge between: left={} right={}",
                        left_arr->type()->ToString(),
                        right_arr->type()->ToString()));
      }
    }
    arrays.emplace_back(left_arr);
  }

  for (auto name :
       rhs->struct_type()->fields()                                                      //
           | views::filter([&lhs](auto& f) { return !lhs->GetFieldByName(f->name()); })  //
           | views::transform([](auto& f) { return f->name(); })) {
    names.emplace_back(name);
    arrays.emplace_back(rhs->GetFieldByName(name));
  }
  return ::arrow::StructArray::Make(arrays, names);
}

/// Concept of a class that has ".fields()" method.
template <typename T>
concept HasFields = (std::same_as<T, ::arrow::Schema> || std::same_as<T, ::arrow::StructType>);

// Forward Declaration.
::arrow::Result<std::shared_ptr<::arrow::Field>> MergeField(const ::arrow::Field& lhs,
                                                            const ::arrow::Field& rhs);

template <HasFields T>
::arrow::Result<std::vector<std::shared_ptr<::arrow::Field>>> MergeFieldWithChildren(const T& lhs,
                                                                                     const T& rhs) {
  std::vector<std::shared_ptr<::arrow::Field>> fields;
  for (const auto& field : lhs.fields()) {
    auto right_field = rhs.GetFieldByName(field->name());
    if (!right_field) {
      fields.emplace_back(field);
    } else {
      ARROW_ASSIGN_OR_RAISE(auto merged, MergeField(*field, *right_field));
      fields.emplace_back(merged);
    }
  }
  ranges::actions::insert(
      fields,
      std::end(fields),  //
      rhs.fields()       //
          | views::filter([&lhs](auto& f) { return !lhs.GetFieldByName(f->name()); }));
  return fields;
};

/// Var-length list type concept.
template <typename T>
concept VarLenListType = (std::same_as<T, ::arrow::ListType> ||
                          std::same_as<T, ::arrow::LargeListType>);

/// Merge two var-length list types (`::arrow::List` or `::arrow::LargeList`).
template <VarLenListType L>
::arrow::Result<std::shared_ptr<::arrow::Field>> MergeListFields(const ::arrow::Field& lhs,
                                                                 const ::arrow::Field& rhs) {
  assert(lhs.type()->id() == L::type_id);
  if (lhs.type()->id() != rhs.type()->id()) {
    return ::arrow::Status::Invalid(
        fmt::format("Attempt to merge two different lists: {} != {}", lhs, rhs));
  }
  auto left_list = std::dynamic_pointer_cast<L>(lhs.type());
  auto right_list = std::dynamic_pointer_cast<L>(rhs.type());
  ARROW_ASSIGN_OR_RAISE(auto merged_field,
                        MergeField(*left_list->value_field(), *right_list->value_field()));
  return ::arrow::field(lhs.name(), std::make_shared<L>(merged_field->type()));
}

::arrow::Result<std::shared_ptr<::arrow::Field>> MergeFixedSizeListFields(
    const ::arrow::Field& lhs, const ::arrow::Field& rhs) {
  assert(lhs.type()->id() == ::arrow::Type::FIXED_SIZE_LIST);
  if (lhs.type()->id() != rhs.type()->id()) {
    return ::arrow::Status::Invalid(
        fmt::format("Attempt to merge two different fixed_size_list lists: {} != {}", lhs, rhs));
  }
  auto left_list = std::dynamic_pointer_cast<::arrow::FixedSizeListType>(lhs.type());
  auto right_list = std::dynamic_pointer_cast<::arrow::FixedSizeListType>(rhs.type());
  if (left_list->list_size() != right_list->list_size()) {
    return ::arrow::Status::Invalid(
        fmt::format("Attempt to merge two fixed size lists with different size: {} != {}",
                    left_list->list_size(),
                    right_list->list_size()));
  }
  ARROW_ASSIGN_OR_RAISE(auto merged_field,
                        MergeField(*left_list->value_field(), *right_list->value_field()));
  return ::arrow::field(lhs.name(),
                        ::arrow::fixed_size_list(merged_field->type(), left_list->list_size()));
}

::arrow::Result<std::shared_ptr<::arrow::Field>> MergeStructFields(const ::arrow::Field& lhs,
                                                                   const ::arrow::Field& rhs) {
  if (!is_struct(rhs.type()->id())) {
    return ::arrow::Status::Invalid(
        fmt::format("Attempt to merge two structs: {} != {}", lhs, rhs));
  }
  // Merge two structs
  auto left_struct = std::dynamic_pointer_cast<::arrow::StructType>(lhs.type());
  auto right_struct = std::dynamic_pointer_cast<::arrow::StructType>(rhs.type());
  ARROW_ASSIGN_OR_RAISE(auto merged_fields, MergeFieldWithChildren(*left_struct, *right_struct));
  return ::arrow::field(lhs.name(), ::arrow::struct_(merged_fields));
}

::arrow::Result<std::shared_ptr<::arrow::Field>> MergeField(const ::arrow::Field& lhs,
                                                            const ::arrow::Field& rhs) {
  if (lhs.name() != rhs.name()) {
    return ::arrow::Status::Invalid(fmt::format(
        "Attempt to merge fields with different names: {} != {}", lhs.name(), rhs.name()));
  }

  switch (lhs.type()->id()) {
    case ::arrow::Type::LIST:
      return MergeListFields<::arrow::ListType>(lhs, rhs);
    case ::arrow::Type::LARGE_LIST:
      return MergeListFields<::arrow::LargeListType>(lhs, rhs);
    case ::arrow::Type::FIXED_SIZE_LIST:
      return MergeFixedSizeListFields(lhs, rhs);
    case ::arrow::Type::STRUCT:
      return MergeStructFields(lhs, rhs);
    default:
      // primitive types, extension types, dictionary, and etc
      break;
  }

  if (!lhs.Equals(rhs)) {
    return ::arrow::Status::Invalid(
        fmt::format("Attempt to merge two different types: {} != {}", lhs, rhs));
  }
  return lhs.MergeWith(rhs);
}

::arrow::Result<std::shared_ptr<::arrow::Schema>> MergeSchema(const ::arrow::Schema& lhs,
                                                              const ::arrow::Schema& rhs) {
  ARROW_ASSIGN_OR_RAISE(auto merged_fields, MergeFieldWithChildren(lhs, rhs));
  return ::arrow::schema(merged_fields);
}

::arrow::Result<std::shared_ptr<::arrow::dataset::FileSystemDataset>> OpenDataset(
    const std::string& uri, std::shared_ptr<::arrow::dataset::Partitioning> partitioning) {
  std::string path;
  ARROW_ASSIGN_OR_RAISE(auto fs, ::arrow::fs::FileSystemFromUriOrPath(uri, &path));
  ::arrow::fs::FileSelector selector;
  selector.base_dir = path;
  selector.recursive = true;
  selector.allow_not_found = true;
  auto format = lance::arrow::LanceFileFormat::Make();

  auto options = ::arrow::dataset::FileSystemFactoryOptions();
  if (partitioning) {
    options.partitioning = partitioning;
  }
  ARROW_ASSIGN_OR_RAISE(
      auto factory,
      ::arrow::dataset::FileSystemDatasetFactory::Make(fs, selector, format, options));
  ARROW_ASSIGN_OR_RAISE(auto dataset, factory->Finish());
  return std::dynamic_pointer_cast<::arrow::dataset::FileSystemDataset>(dataset);
}

template <typename ArrayType>
::arrow::Result<std::shared_ptr<::arrow::Array>> CreateArrayImpl(
    const std::shared_ptr<::arrow::Scalar>& scalar, int64_t length, ::arrow::MemoryPool* pool) {
  auto concrete_scalar =
      std::dynamic_pointer_cast<typename ::arrow::TypeTraits<ArrayType>::ScalarType>(scalar);
  auto builder =
      std::make_shared<typename ::arrow::TypeTraits<ArrayType>::BuilderType>(scalar->type, pool);
  ARROW_RETURN_NOT_OK(builder->Reserve(length));
  for (int64_t i = 0; i < length; i++) {
    ARROW_RETURN_NOT_OK(builder->Append(concrete_scalar->value));
  }
  return builder->Finish();
}

template <>
::arrow::Result<std::shared_ptr<::arrow::Array>> CreateArrayImpl<::arrow::StringType>(
    const std::shared_ptr<::arrow::Scalar>& scalar, int64_t length, ::arrow::MemoryPool* pool) {
  auto concrete_scalar = std::dynamic_pointer_cast<::arrow::StringScalar>(scalar);
  auto builder = ::arrow::StringBuilder(pool);
  ARROW_RETURN_NOT_OK(builder.Reserve(length));
  for (int64_t i = 0; i < length; i++) {
    ARROW_RETURN_NOT_OK(builder.Append(concrete_scalar->view()));
  }
  return builder.Finish();
}

::arrow::Result<std::shared_ptr<::arrow::Array>> CreateArray(
    const std::shared_ptr<::arrow::Scalar>& scalar, int64_t length, ::arrow::MemoryPool* pool) {
  ARROW_ASSIGN_OR_RAISE(auto builder, GetArrayBuilder(scalar->type, pool));
  switch (scalar->type->id()) {
    case ::arrow::Type::BOOL:
      return CreateArrayImpl<::arrow::BooleanType>(scalar, length, pool);
    case ::arrow::Type::UINT8:
      return CreateArrayImpl<::arrow::UInt8Type>(scalar, length, pool);
    case ::arrow::Type::INT8:
      return CreateArrayImpl<::arrow::Int8Type>(scalar, length, pool);
    case ::arrow::Type::UINT16:
      return CreateArrayImpl<::arrow::UInt16Type>(scalar, length, pool);
    case ::arrow::Type::INT16:
      return CreateArrayImpl<::arrow::Int16Type>(scalar, length, pool);
    case ::arrow::Type::UINT32:
      return CreateArrayImpl<::arrow::UInt32Type>(scalar, length, pool);
    case ::arrow::Type::INT32:
      return CreateArrayImpl<::arrow::Int32Type>(scalar, length, pool);
    case ::arrow::Type::UINT64:
      return CreateArrayImpl<::arrow::UInt64Type>(scalar, length, pool);
    case ::arrow::Type::INT64:
      return CreateArrayImpl<::arrow::Int64Type>(scalar, length, pool);
    case ::arrow::Type::HALF_FLOAT:
      return CreateArrayImpl<::arrow::HalfFloatType>(scalar, length, pool);
    case ::arrow::Type::FLOAT:
      return CreateArrayImpl<::arrow::FloatType>(scalar, length, pool);
    case ::arrow::Type::DOUBLE:
      return CreateArrayImpl<::arrow::DoubleType>(scalar, length, pool);
    case ::arrow::Type::STRING:
      return CreateArrayImpl<::arrow::StringType>(scalar, length, pool);
    default:
      return ::arrow::Status::Invalid(
          fmt::format("CreateArray: unsupported type: {}", scalar->type->ToString()));
  }
}

namespace {

class UuidGenerator {
 public:
  UuidGenerator() {
    std::generate(std::begin(seed_), std::end(seed_), std::ref(rd_));
    std::seed_seq seq(std::begin(seed_), std::end(seed_));
    random_gen_ = std::make_unique<std::mt19937>(seq);
    generator_ = std::make_unique<uuids::uuid_random_generator>(*random_gen_);
  }

  std::string operator()() const {
    auto uuid = (*generator_)();
    return uuids::to_string(uuid);
  }

 private:
  std::random_device rd_;
  std::array<int, std::mt19937::state_size> seed_;
  std::unique_ptr<std::mt19937> random_gen_;
  std::unique_ptr<uuids::uuid_random_generator> generator_;
};

const UuidGenerator uuid_generator_;

}  // namespace

std::string GetUUIDString() { return uuid_generator_(); }

}  // namespace lance::arrow