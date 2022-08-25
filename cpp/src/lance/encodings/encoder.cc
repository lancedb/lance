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

#include "lance/encodings/encoder.h"

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/type.h>

#include <memory>

namespace lance::encodings {

Decoder::Decoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                 std::shared_ptr<::arrow::DataType> type,
                 ::arrow::MemoryPool* pool) noexcept
    : infile_(infile), type_(type), pool_(pool) {}

::arrow::Status Decoder::Init() { return ::arrow::Status::OK(); }

void Decoder::Reset(int64_t position, int32_t length) {
  position_ = position;
  length_ = length;
}

::arrow::Result<std::shared_ptr<::arrow::Scalar>> Decoder::GetScalar(int64_t idx) const {
  ARROW_ASSIGN_OR_RAISE(auto arr, ToArray(idx, 1));
  return arr->GetScalar(0);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> Decoder::Take(
    std::shared_ptr<::arrow::Int32Array> indices) const {
  ARROW_ASSIGN_OR_RAISE(auto builder, ::lance::arrow::GetArrayBuilder(type_, pool_));
  ARROW_RETURN_NOT_OK(builder->Reserve(indices->length()));
  // TODO: use arrow thread pool
  for (int64_t i = 0; i < indices->length(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto scalar, GetScalar(indices->Value(i)));
    ARROW_RETURN_NOT_OK(builder->AppendScalar(*scalar));
  }
  return builder->Finish();
}

}  // namespace lance::encodings