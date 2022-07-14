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

#include <arrow/array.h>
#include <arrow/io/api.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <memory>

#include "lance/encodings/encoder.h"

namespace lance::encodings {

class PlainEncoder : public Encoder {
 public:
  explicit PlainEncoder(std::shared_ptr<::arrow::io::OutputStream> out);

  virtual ~PlainEncoder() = default;

  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::Array> arr) override;

  std::string ToString() const override { return "Encoder(type=Plain)"; }
};

template <ArrowType T>
class PlainDecoder : public Decoder {
 public:
  using Decoder::Decoder;

  virtual ~PlainDecoder() override = default;

  /** Get a Value without scanning the full row group. */
  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override {
    CType value;
    RETURN_NOT_OK(infile_->ReadAt(position_ + idx * sizeof(CType), sizeof(CType), &value));
    return std::make_shared<ScalarType>(value);
  }

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const override {
    if (!length.has_value()) {
      length = length_ - start;
    }
    if (start + length.value() > length_ || start > length_) {
      return ::arrow::Status::IndexError(
          fmt::format("PlainDecoder::ToArray: out of range: start={}, length={}, chunk_length={}\n",
                      start,
                      length.value(),
                      length_));
    }
    ARROW_ASSIGN_OR_RAISE(
        auto buf,
        infile_->ReadAt(position_ + start * sizeof(CType), length.value() * sizeof(CType)));
    return std::make_shared<ArrayType>(length.value(), buf);
  }

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const override {
    int32_t start = indices->Value(0);
    int32_t length = indices->Value(indices->length() - 1) - start + 1;
    if (indices->length() == 0 || start < 0 || start + length > length_) {
      return ::arrow::Status::Invalid("PlainDecoder::Take: Indices array is not valid");
    }
    // For the simplicity, we read all data in batch to reduce random I/O.
    // And apply indices later.
    // We can optimize this later if the indices are sparse and making small I/Os can bring
    // benefits.
    ARROW_ASSIGN_OR_RAISE(auto raw_value_arr, ToArray(start, length));
    auto values = std::static_pointer_cast<ArrayType>(raw_value_arr);
    typename ::arrow::TypeTraits<T>::BuilderType builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(indices->length()));
    for (int64_t i = 0; i < indices->length(); i++) {
      auto index = indices->Value(i);
      ARROW_RETURN_NOT_OK(builder.Append(values->Value(index - start)));
    }
    return builder.Finish();
  };

 private:
  using CType = typename ::arrow::TypeTraits<T>::CType;
  using ScalarType = typename ::arrow::TypeTraits<T>::ScalarType;
  using ArrayType = typename ::arrow::TypeTraits<T>::ArrayType;
};

}  // namespace lance::encodings
