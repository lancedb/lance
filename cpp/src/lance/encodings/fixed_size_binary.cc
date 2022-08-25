//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/encodings/fixed_size_binary.h"

#include <arrow/builder.h>

#include "lance/arrow/type.h"
#include "lance/encodings/encoder.h"
#include "lance/encodings/plain.h"

namespace lance::encodings {

::arrow::Result<std::shared_ptr<::arrow::Array>> FixedSizedBinaryDecoder::ToFixedSizeListArray(
    int32_t start, int32_t length) const {
  auto fixed_list_type = std::dynamic_pointer_cast<::arrow::FixedSizeListType>(type_);
  auto value_type = fixed_list_type->value_field();
  assert(::arrow::is_primitive(value_type->type()->id()));
  auto plain_decoder = PlainDecoder(infile_, value_type->type());
  ARROW_RETURN_NOT_OK(plain_decoder.Init());

  auto list_size = fixed_list_type->list_size();
  plain_decoder.Reset(position_, length_ * list_size);
  ARROW_ASSIGN_OR_RAISE(auto values, plain_decoder.ToArray(start * list_size, length * list_size));
  return std::make_shared<::arrow::FixedSizeListArray>(type_, length, values);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FixedSizedBinaryDecoder::ToArray(
    int32_t start, std::optional<int32_t> length) const {
  if (!length.has_value()) {
    length = length_ - start;
  }
  if (start + length.value() > length_ || start > length_) {
    return ::arrow::Status::IndexError(
        fmt::format("{}::ToArray: out of range: start={}, length={}, page_length={}\n",
                    ToString(),
                    start,
                    length.value(),
                    length_));
  }
  if (lance::arrow::is_fixed_size_list(type_)) {
    return ToFixedSizeListArray(start, length.value());
  }
  return ::arrow::Status::Invalid("Invalid data type: {}\n", type_->ToString());
}

::arrow::Result<std::shared_ptr<::arrow::Array>> FixedSizedBinaryDecoder::Take(
    std::shared_ptr<::arrow::Int32Array> indices) const {
  std::shared_ptr<::arrow::ArrayBuilder> builder;
  if (lance::arrow::is_fixed_size_list(type_)) {
    auto list_type = std::dynamic_pointer_cast<::arrow::FixedSizeListType>(type_);
    ARROW_ASSIGN_OR_RAISE(auto value_builder,
                          ::lance::arrow::GetArrayBuilder(list_type->value_type()));
    builder = std::make_shared<::arrow::FixedSizeListBuilder>(pool_, value_builder, type_);
  } else if (::arrow::is_fixed_size_binary(type_->id())) {
    builder = std::make_shared<::arrow::FixedSizeBinaryBuilder>(type_, pool_);
  } else {
    return ::arrow::Status::Invalid("FixedSizeBuilderDecoder::Take: Invalid data type: ", type_);
  }

  // TODO: Use thread pool
  for (int i = 0; i < indices->length(); i++) {
    ARROW_ASSIGN_OR_RAISE(auto scalar, GetScalar(indices->Value(i)));
    ARROW_RETURN_NOT_OK(builder->AppendScalar(*scalar));
  }
  return builder->Finish();
}

std::string FixedSizedBinaryDecoder::ToString() const { return "FixedSizeBinaryDecoder"; }

}  // namespace lance::encodings