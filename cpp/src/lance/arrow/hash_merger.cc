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

#include "lance/arrow/hash_merger.h"

#include <arrow/compute/api.h>
#include <arrow/type_traits.h>

#include <optional>

#include "lance/arrow/stl.h"
#include "lance/arrow/type.h"

namespace lance::arrow {

class HashMerger::Impl {
 public:
  virtual ~Impl() = default;

  virtual void ComputeHash(const std::shared_ptr<::arrow::Array>& arr,
                           std::vector<std::optional<std::size_t>>* out) = 0;

  virtual ::arrow::Result<std::unordered_map<std::size_t, int64_t>> BuildHashChunkIndex(
      const std::shared_ptr<::arrow::ChunkedArray>& chunked_arr) = 0;
};

template <ArrowType T, typename CType = typename ::arrow::TypeTraits<T>::CType>
class TypedHashMerger : public HashMerger::Impl {
 public:
  void ComputeHash(const std::shared_ptr<::arrow::Array>& arr,
                   std::vector<std::optional<std::size_t>>* out) override {
    auto hash_func = std::hash<CType>{};
    assert(out);
    auto values = std::dynamic_pointer_cast<typename ::arrow::TypeTraits<T>::ArrayType>(arr);
    assert(values);
    out->reserve(values->length());
    out->clear();
    for (int i = 0; i < values->length(); ++i) {
      if (values->IsNull(i)) {
        out->emplace_back(std::nullopt);
      } else {
        auto value = values->Value(i);
        out->emplace_back(hash_func(value));
      }
    }
  }

  ::arrow::Result<std::unordered_map<std::size_t, int64_t>> BuildHashChunkIndex(
      const std::shared_ptr<::arrow::ChunkedArray>& chunked_arr) override {
    std::unordered_map<std::size_t, int64_t> key_to_chunk_index;
    int64_t index = 0;
    std::vector<std::optional<std::size_t>> hashes;
    for (const auto& chunk : chunked_arr->chunks()) {
      ComputeHash(chunk, &hashes);
      assert(chunk->length() == static_cast<int64_t>(hashes.size()));
      for (std::size_t i = 0; i < hashes.size(); i++) {
        const auto& key = hashes[i];
        if (key.has_value()) {
          auto ret = key_to_chunk_index.emplace(key.value(), index);
          if (!ret.second) {
            auto values =
                std::dynamic_pointer_cast<typename ::arrow::TypeTraits<T>::ArrayType>(chunk);
            return ::arrow::Status::IndexError("Duplicate key found: ", values->Value(i));
          }
        }
        index++;
      }
    }
    return key_to_chunk_index;
  }
};

HashMerger::HashMerger(std::shared_ptr<::arrow::Table> table,
                       std::string index_column,
                       ::arrow::MemoryPool* pool)
    : table_(std::move(table)), column_name_(std::move(index_column)), pool_(pool) {}

HashMerger::~HashMerger() {}

::arrow::Status HashMerger::Init() {
  auto chunked_arr = table_->GetColumnByName(column_name_);
  if (chunked_arr == nullptr) {
    return ::arrow::Status::Invalid("index column ", column_name_, " does not exist");
  }
  index_column_type_ = chunked_arr->type();

#define BUILD_IMPL(TypeId)                                                    \
  case TypeId:                                                                \
    impl_ = std::unique_ptr<Impl>(                                            \
        new TypedHashMerger<typename ::arrow::TypeIdTraits<TypeId>::Type>()); \
    break;

  switch (index_column_type_->id()) {
    BUILD_IMPL(::arrow::Type::UINT8);
    BUILD_IMPL(::arrow::Type::INT8);
    BUILD_IMPL(::arrow::Type::UINT16);
    BUILD_IMPL(::arrow::Type::INT16);
    BUILD_IMPL(::arrow::Type::UINT32);
    BUILD_IMPL(::arrow::Type::INT32);
    BUILD_IMPL(::arrow::Type::UINT64);
    BUILD_IMPL(::arrow::Type::INT64);
    case ::arrow::Type::HALF_FLOAT:
    case ::arrow::Type::FLOAT:
    case ::arrow::Type::DOUBLE:
      return ::arrow::Status::Invalid("Do not support merge on floating points");
    case ::arrow::Type::STRING:
      impl_ = std::unique_ptr<Impl>(new TypedHashMerger<::arrow::StringType, std::string_view>());
      break;
    default:
      return ::arrow::Status::Invalid("Only support primitive or string type, got: ",
                                      index_column_type_->ToString());
  }

  ARROW_ASSIGN_OR_RAISE(index_map_, impl_->BuildHashChunkIndex(chunked_arr));

  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> HashMerger::Collect(
    const std::shared_ptr<::arrow::Array>& index_arr) {
  if (!index_arr->type()->Equals(index_column_type_)) {
    return ::arrow::Status::TypeError(
        "Index column match mismatch: ", index_arr->type()->ToString(), " != ", index_column_type_);
  }
  std::vector<std::optional<std::size_t>> hashes;
  impl_->ComputeHash(index_arr, &hashes);
  ::arrow::Int64Builder indices_builder;
  ARROW_RETURN_NOT_OK(indices_builder.Reserve(index_arr->length()));
  for (const auto& hvalue : hashes) {
    if (hvalue.has_value()) {
      auto it = index_map_.find(hvalue.value());
      if (it != index_map_.end()) {
        ARROW_RETURN_NOT_OK(indices_builder.Append(it->second));
      } else {
        ARROW_RETURN_NOT_OK(indices_builder.AppendNull());
      }
    } else {
      ARROW_RETURN_NOT_OK(indices_builder.AppendNull());
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto indices_arr, indices_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto datum, ::arrow::compute::Take(table_, indices_arr));
  assert(datum.table());
  auto table = datum.table();

  // Drop the index column.
  for (int i = 0; i < table->num_columns(); ++i) {
    if (table->field(i)->name() == column_name_) {
      ARROW_ASSIGN_OR_RAISE(table, table->RemoveColumn(i));
      break;
    }
  }
  return table->CombineChunksToBatch(pool_);
}

}  // namespace lance::arrow