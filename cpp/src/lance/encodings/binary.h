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

#pragma once

#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/scalar.h>
#include <arrow/status.h>
#include <arrow/stl_iterator.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <memory>
#include <string>
#include <vector>

#include "lance/encodings/encoder.h"
#include "lance/format/format.h"

namespace lance::encodings {

/**
 * Var-length Binary Encoding.
 *
 * Layouts:
 *
 * |value1|value2|value3|value4|
 *
 * |length |
 * |offset1|offset2|offset3|offset4|
 */
class VarBinaryEncoder : public Encoder {
 public:
  /// Can we make this int32 type>
  using OffsetType = ::arrow::Int64Type;

  explicit VarBinaryEncoder(std::shared_ptr<::arrow::io::OutputStream> out) noexcept;

  virtual ~VarBinaryEncoder() = default;

  /// Write an Array, and returns the offsets to the index block.
  ::arrow::Result<int64_t> Write(const std::shared_ptr<::arrow::Array> arr) override;

  /// Debug string.
  std::string ToString() const override { return "Encoder(type=VarBinary)"; }

 private:
  ::arrow::TypeTraits<OffsetType>::BuilderType offsetBuilder_;
  std::shared_ptr<::arrow::TypeTraits<OffsetType>::ArrayType> offsetsArr;
};

/// Decode for Var-length binary encoding.
template <ArrowType T>
class VarBinaryDecoder : public Decoder {
 public:
  using Decoder::Decoder;

  virtual ~VarBinaryDecoder() = default;

  /** Get a Value without scanning the full row group. */
  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t idx = 0, std::optional<int32_t> length = std::nullopt) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const override;

 private:
  using OffsetType = VarBinaryEncoder::OffsetType;
  using OffsetCType = typename VarBinaryEncoder::OffsetType::c_type;
  using OffsetArrayType = typename ::arrow::TypeTraits<VarBinaryEncoder::OffsetType>::ArrayType;
  using ArrayType = typename ::arrow::TypeTraits<T>::ArrayType;
};

template <ArrowType T>
::arrow::Result<std::shared_ptr<::arrow::Scalar>> VarBinaryDecoder<T>::GetScalar(
    int64_t idx) const {
  ARROW_ASSIGN_OR_RAISE(
      auto offset_buf,
      infile_->ReadAt(position_ + idx * sizeof(OffsetCType), 2 * sizeof(OffsetCType)));
  auto offset_arr = OffsetArrayType(2, offset_buf);
  ARROW_ASSIGN_OR_RAISE(
      auto buf, infile_->ReadAt(offset_arr.Value(0), offset_arr.Value(1) - offset_arr.Value(0)));
  return std::make_shared<typename ::arrow::TypeTraits<T>::ScalarType>(buf);
}

template <ArrowType T>
::arrow::Result<std::shared_ptr<::arrow::Array>> VarBinaryDecoder<T>::ToArray(
    int32_t start, std::optional<int32_t> length) const {
  if (!length.has_value()) {
    length = length_ - start;
  }
  if (start + *length > length_) {
    return ::arrow::Status::IndexError(
        fmt::format("VarBinaryDecoder::ToArray: out of range: start={} length={} page_length={}\n",
                    start,
                    *length,
                    length_));
  }

  auto offsets_buf =
      infile_->ReadAt(position_ + start * sizeof(OffsetCType), (*length + 1) * sizeof(OffsetCType));
  if (!offsets_buf.ok()) {
    return ::arrow::Status::IOError(
        fmt::format("VarBinaryDecoder::ToArray: failed to read offset: start={}, length={}: {}",
                    start,
                    *length,
                    offsets_buf.status().message()));
  }
  auto positions = std::make_shared<typename ::arrow::TypeTraits<OffsetType>::ArrayType>(
      *length + 1, *offsets_buf);
  auto start_offset = positions->Value(0);

  ::arrow::Int32Builder builder;
  for (int64_t i = 0; i < positions->length(); ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(static_cast<int32_t>(positions->Value(i) - start_offset)));
  }
  auto value_offsets_buf = builder.Finish();
  if (!value_offsets_buf.ok()) {
    return value_offsets_buf.status();
  }
  auto value_offsets = std::static_pointer_cast<::arrow::Int32Array>(*value_offsets_buf);
  auto read_length = positions->Value(positions->length() - 1) - start_offset;
  ARROW_ASSIGN_OR_RAISE(auto data_buf, infile_->ReadAt(start_offset, read_length));
  return std::make_shared<ArrayType>(*length, value_offsets->values(), data_buf);
}

template <ArrowType T>
::arrow::Result<std::shared_ptr<::arrow::Array>> VarBinaryDecoder<T>::Take(
    std::shared_ptr<::arrow::Int32Array> indices) const {
  typename ::arrow::TypeTraits<T>::BuilderType builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(indices->length()));

  for (auto index : *indices) {
    assert(index.has_value());
    ARROW_ASSIGN_OR_RAISE(auto scalar, GetScalar(index.value()));
    auto binary_scalar =
        std::dynamic_pointer_cast<typename ::arrow::TypeTraits<T>::ScalarType>(scalar);
    ARROW_RETURN_NOT_OK(builder.Append(binary_scalar->view()));
  }
  return builder.Finish();
}

}  // namespace lance::encodings