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

#include <arrow/type_traits.h>

namespace lance::arrow {

HashMerger::HashMerger(const ::arrow::Table& table,
                       std::string index_column,
                       ::arrow::MemoryPool* pool)
    : table_(table), column_name_(std::move(index_column)), pool_(pool) {}

namespace {

/// Build index map: key => {chunk_id, idx_in_chunk}.
///
template <typename ArrowType, typename CType = typename ::arrow::TypeTraits<ArrowType>::CType>
::arrow::Result<std::unordered_map<std::size_t, std::tuple<int64_t, int64_t>>> BuildHashChunkIndex(
    const std::shared_ptr<::arrow::ChunkedArray>& chunked_arr) {
  std::unordered_map<std::size_t, std::tuple<int64_t, int64_t>> key_to_chunk_index;
  for (int64_t chk = 0; chk < chunked_arr->num_chunks(); chk++) {
    auto arr = std::dynamic_pointer_cast<typename ::arrow::TypeTraits<ArrowType>::ArrayType>(
        chunked_arr->chunk(chk));
    for (int64_t idx = 0; idx < arr->length(); idx++) {
      auto value = arr->Value(idx);
      auto key = std::hash<CType>{}(value);
      auto ret = key_to_chunk_index.emplace(key, std::make_tuple(chk, idx));
      if (!ret.second) {
        return ::arrow::Status::IndexError("Duplicated key found: ", value);
      }
    }
  }
  return std::move(key_to_chunk_index);
}

}  // namespace

::arrow::Status HashMerger::Build() {
  auto chunked_arr = table_.GetColumnByName(column_name_);
  index_column_type_ = chunked_arr->type();

  ::arrow::Result<std::unordered_map<std::size_t, std::tuple<int64_t, int64_t>>> result;

#define BUILD_CHUNK_IDX(TypeId)                                                              \
  case TypeId:                                                                               \
    result = BuildHashChunkIndex<typename ::arrow::TypeIdTraits<TypeId>::Type>(chunked_arr); \
    break;

  switch (index_column_type_->id()) {
    BUILD_CHUNK_IDX(::arrow::Type::UINT8);
    BUILD_CHUNK_IDX(::arrow::Type::INT8);
    BUILD_CHUNK_IDX(::arrow::Type::UINT16);
    BUILD_CHUNK_IDX(::arrow::Type::INT16);
    BUILD_CHUNK_IDX(::arrow::Type::UINT32);
    BUILD_CHUNK_IDX(::arrow::Type::INT32);
    BUILD_CHUNK_IDX(::arrow::Type::UINT64);
    BUILD_CHUNK_IDX(::arrow::Type::INT64);
    //    BUILD_CHUNK_IDX(::arrow::Type::HALF_FLOAT);
    BUILD_CHUNK_IDX(::arrow::Type::FLOAT);
    BUILD_CHUNK_IDX(::arrow::Type::DOUBLE);
    case ::arrow::Type::STRING:
      result = BuildHashChunkIndex<::arrow::StringType, std::string_view>(chunked_arr);
      break;
    default:
      return ::arrow::Status::Invalid("Only support primitive or string type, got: ",
                                      index_column_type_->ToString());
  }

  if (!result.ok()) {
    return result.status();
  }
  index_map_ = std::move(result.ValueOrDie());
  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> HashMerger::Collect(
    const std::shared_ptr<::arrow::Array>& on_col) {
  if (!on_col->type()->Equals(index_column_type_)) {
    return ::arrow::Status::TypeError(
        "Index column match mismatch: ", on_col->type()->ToString(), " != ", index_column_type_);
  }
  for (int i = 0; i < table_.num_columns(); i++) {
    auto field = table_.field(i);
    if (field->name() == column_name_) {
      continue;
    }
  }
  fmt::print("{}", fmt::ptr(pool_));
  return ::arrow::Status::NotImplemented("not impl");
}

}  // namespace lance::arrow