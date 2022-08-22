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

#include "lance/encodings/plain.h"

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/scalar.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <memory>

#include "lance/encodings/encoder.h"

using ::arrow::Result;
using ::arrow::Status;

namespace lance::encodings {

PlainEncoder::PlainEncoder(std::shared_ptr<::arrow::io::OutputStream> out) : Encoder(out) {}

::arrow::Status WriteBooleanArray(const std::shared_ptr<::arrow::io::OutputStream>& out,
                                  const std::shared_ptr<::arrow::BooleanArray>& arr) {
  // TODO: Boolean array is not necessarily aligned with the byte boundary:
  //  See ::arrow::BooleanArray::Value(int64)
  // Is there a faster way to write / shift the boolean array?
  ::arrow::BooleanBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(arr->length()));
  for (int i = 0; i < arr->length(); i++) {
    ARROW_RETURN_NOT_OK(builder.Append(arr->Value(i)));
  }
  ARROW_ASSIGN_OR_RAISE(auto written_arr, builder.Finish());
  return out->Write(std::dynamic_pointer_cast<::arrow::BooleanArray>(written_arr)->values());
}

::arrow::Result<int64_t> PlainEncoder::Write(std::shared_ptr<::arrow::Array> arr) {
  auto data_type = arr->type();

  ARROW_ASSIGN_OR_RAISE(auto value_offset, out_->Tell());
  // TODO: support more types.
  switch (data_type->id()) {
    case ::arrow::Type::BOOL:
      ARROW_RETURN_NOT_OK(
          WriteBooleanArray(out_, std::dynamic_pointer_cast<::arrow::BooleanArray>(arr)));
      break;
    case ::arrow::Type::INT8:
      ARROW_RETURN_NOT_OK(out_->Write(std::static_pointer_cast<::arrow::Int8Array>(arr)->values()));
      break;
    case ::arrow::Type::UINT8:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::static_pointer_cast<::arrow::UInt8Array>(arr)->values()));
      break;
    case ::arrow::Type::INT16:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::static_pointer_cast<::arrow::Int16Array>(arr)->values()));
      break;
    case ::arrow::Type::UINT16:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::static_pointer_cast<::arrow::UInt16Array>(arr)->values()));
      break;
    case ::arrow::Type::INT32:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::reinterpret_pointer_cast<::arrow::Int32Array>(arr)->values()));
      break;
    case ::arrow::Type::UINT32:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::static_pointer_cast<::arrow::UInt32Array>(arr)->values()));
      break;
    case ::arrow::Type::INT64:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::reinterpret_pointer_cast<::arrow::Int64Array>(arr)->values()));
      break;
    case ::arrow::Type::UINT64:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::static_pointer_cast<::arrow::UInt64Array>(arr)->values()));
      break;
    case ::arrow::Type::FLOAT:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::reinterpret_pointer_cast<::arrow::FloatArray>(arr)->values()));
      break;
    case ::arrow::Type::DOUBLE:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::reinterpret_pointer_cast<::arrow::DoubleArray>(arr)->values()));
      break;
    default:
      return Status::Invalid(
          fmt::format("PlainEncoder:: does not support data type {}", data_type->ToString()));
  }
  return value_offset;
}

namespace {

template <ArrowType T>
class PlainDecoderImpl : public Decoder {
 public:
  using Decoder::Decoder;

  /// Get one single scalar value from the column.
  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override {
    CType value;
    ARROW_RETURN_NOT_OK(infile_->ReadAt(position_ + idx * sizeof(value), sizeof(value), &value));
    return std::make_shared<typename ::arrow::TypeTraits<T>::ScalarType>(value);
  }

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start, std::optional<int32_t> length) const override {
    if (!length.has_value()) {
      length = length_ - start;
    }
    if (start + length.value() > length_ || start > length_) {
      return ::arrow::Status::IndexError(
          fmt::format("PlainDecoder::ToArray: out of range: start={}, length={}, page_length={}\n",
                      start,
                      length.value(),
                      length_));
    }
    auto bytes = std::max(1, ::arrow::bit_width(type_->id()) / 8);
    ARROW_ASSIGN_OR_RAISE(auto buf,
                          infile_->ReadAt(position_ + start * bytes, length.value() * bytes));
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
    BuilderType builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(indices->length()));
    for (int64_t i = 0; i < indices->length(); i++) {
      auto index = indices->Value(i);
      ARROW_RETURN_NOT_OK(builder.Append(values->Value(index - start)));
    }
    return builder.Finish();
  }

 private:
  using CType = typename ::arrow::TypeTraits<T>::CType;
  using ScalarType = typename ::arrow::TypeTraits<T>::ScalarType;
  using ArrayType = typename ::arrow::TypeTraits<T>::ArrayType;
  using BuilderType = typename ::arrow::TypeTraits<T>::BuilderType;
};

class BooleanPlainDecoderImpl : public PlainDecoderImpl<::arrow::BooleanType> {
 public:
  using PlainDecoderImpl::PlainDecoderImpl;

  /// Get one single scalar value from the column.
  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override {
    int64_t offset = idx / 8;
    uint8_t byte;
    ARROW_RETURN_NOT_OK(infile_->ReadAt(position_ + offset, 1, &byte));
    return std::make_shared<::arrow::BooleanScalar>(
        ::arrow::bit_util::GetBitFromByte(byte, idx % 8));
  }

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start, std::optional<int32_t> length) const override {
    if (!length.has_value()) {
      length = length_ - start;
    }
    if (start + length.value() > length_ || start > length_) {
      return ::arrow::Status::IndexError(
          fmt::format("PlainDecoder::ToArray: out of range: start={}, length={}, page_length={}\n",
                      start,
                      length.value(),
                      length_));
    }
    int64_t byte_length = ::arrow::bit_util::BytesForBits(length.value());
    ARROW_ASSIGN_OR_RAISE(auto buf, infile_->ReadAt(position_ + start / 8, byte_length));
    return std::make_shared<::arrow::BooleanArray>(length.value(), buf);
  }
};

}  // namespace

PlainDecoder::PlainDecoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                           std::shared_ptr<::arrow::DataType> type)
    : Decoder(infile, type) {}

PlainDecoder::~PlainDecoder() {}

::arrow::Status PlainDecoder::Init() {
  switch (type_->id()) {
    case ::arrow::Type::BOOL:
      impl_.reset(new BooleanPlainDecoderImpl(infile_, type_));
      break;
    case ::arrow::Type::INT8:
      impl_.reset(new PlainDecoderImpl<::arrow::Int8Type>(infile_, type_));
      break;
    case ::arrow::Type::UINT8:
      impl_.reset(new PlainDecoderImpl<::arrow::UInt8Type>(infile_, type_));
      break;
    case ::arrow::Type::INT16:
      impl_.reset(new PlainDecoderImpl<::arrow::Int16Type>(infile_, type_));
      break;
    case ::arrow::Type::UINT16:
      impl_.reset(new PlainDecoderImpl<::arrow::UInt16Type>(infile_, type_));
      break;
    case ::arrow::Type::INT32:
      impl_.reset(new PlainDecoderImpl<::arrow::Int32Type>(infile_, type_));
      break;
    case ::arrow::Type::UINT32:
      impl_.reset(new PlainDecoderImpl<::arrow::UInt32Type>(infile_, type_));
      break;
    case ::arrow::Type::INT64:
      impl_.reset(new PlainDecoderImpl<::arrow::Int64Type>(infile_, type_));
      break;
    case ::arrow::Type::UINT64:
      impl_.reset(new PlainDecoderImpl<::arrow::UInt64Type>(infile_, type_));
      break;
    case ::arrow::Type::FLOAT:
      impl_.reset(new PlainDecoderImpl<::arrow::FloatType>(infile_, type_));
      break;
    case ::arrow::Type::DOUBLE:
      impl_.reset(new PlainDecoderImpl<::arrow::DoubleType>(infile_, type_));
      break;
    default:
      return ::arrow::Status::Invalid(fmt::format("Unsupported type: {}", type_->ToString()));
  }
  return ::arrow::Status::OK();
}

void PlainDecoder::Reset(int64_t position, int32_t length) {
  Decoder::Reset(position, length);
  impl_->Reset(position, length);
}

::arrow::Result<std::shared_ptr<::arrow::Scalar>> PlainDecoder::GetScalar(int64_t idx) const {
  return impl_->GetScalar(idx);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> PlainDecoder::ToArray(
    int32_t start, std::optional<int32_t> length) const {
  return impl_->ToArray(start, length);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> PlainDecoder::Take(
    std::shared_ptr<::arrow::Int32Array> indices) const {
  return impl_->Take(indices);
}

}  // namespace lance::encodings
