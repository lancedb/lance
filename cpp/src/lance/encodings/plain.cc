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

::arrow::Result<int64_t> PlainEncoder::WriteFixedSizeListArray(
    const std::shared_ptr<::arrow::FixedSizeListArray>& arr) {
  assert(::arrow::is_primitive(arr->values()->type_id()));
  return Write(arr->values());
}

template <ArrowType T>
::arrow::Status WriteArray(const std::shared_ptr<::arrow::io::OutputStream>& out,
                           const std::shared_ptr<::arrow::Array>& arr) {
  auto width = arr->type()->byte_width();
  assert(width > 0);
  return out->Write(
      std::dynamic_pointer_cast<typename ::arrow::TypeTraits<T>::ArrayType>(arr)->raw_values(),
      arr->length() * width);
}

::arrow::Result<int64_t> PlainEncoder::Write(const std::shared_ptr<::arrow::Array>& arr) {
  auto data_type = arr->type();

  ARROW_ASSIGN_OR_RAISE(auto value_offset, out_->Tell());
  // TODO: support more types.
  switch (data_type->id()) {
    case ::arrow::Type::BOOL:
      ARROW_RETURN_NOT_OK(
          WriteBooleanArray(out_, std::dynamic_pointer_cast<::arrow::BooleanArray>(arr)));
      break;
    case ::arrow::Type::INT8:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::Int8Type>(out_, arr));
      break;
    case ::arrow::Type::UINT8:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::UInt8Type>(out_, arr));
      break;
    case ::arrow::Type::INT16:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::Int16Type>(out_, arr));
      break;
    case ::arrow::Type::UINT16:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::UInt16Type>(out_, arr));
      break;
    case ::arrow::Type::INT32:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::Int32Type>(out_, arr));
      break;
    case ::arrow::Type::UINT32:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::UInt32Type>(out_, arr));
      break;
    case ::arrow::Type::INT64:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::Int64Type>(out_, arr));
      break;
    case ::arrow::Type::UINT64:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::UInt64Type>(out_, arr));
      break;
    case ::arrow::Type::FLOAT:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::FloatType>(out_, arr));
      break;
    case ::arrow::Type::DOUBLE:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::DoubleType>(out_, arr));
      break;
    case ::arrow::Type::FIXED_SIZE_BINARY:
      ARROW_RETURN_NOT_OK(WriteArray<::arrow::FixedSizeBinaryType>(out_, arr));
      break;
    case ::arrow::Type::FIXED_SIZE_LIST:
      ARROW_RETURN_NOT_OK(
          WriteFixedSizeListArray(std::dynamic_pointer_cast<::arrow::FixedSizeListArray>(arr)));
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

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start, std::optional<int32_t> length) const override {
    auto len = GetReadLength(start, length);
    if (len <= 0) {
      return ::arrow::Status::IndexError(
          fmt::format("{}::ToArray: out of range: start={}, length={}, page_length={}\n",
                      ToString(),
                      start,
                      length.value_or(-1),
                      length_));
    }
    auto byte_width = type_->byte_width();
    ARROW_ASSIGN_OR_RAISE(auto buf,
                          infile_->ReadAt(position_ + start * byte_width, len * byte_width));
    return std::make_shared<ArrayType>(type_, len, buf);
  }

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const override {
    // Simple heuristic to:
    // Use batch scan for primitive data
    // Use parallel reads for large(r) blob
    if (!::arrow::is_primitive(type_->id())) {
      return Decoder::Take(indices);
    }

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
    auto values = std::dynamic_pointer_cast<ArrayType>(raw_value_arr);
    BuilderType builder(type_, pool_);
    ARROW_RETURN_NOT_OK(builder.Reserve(indices->length()));
    for (int64_t i = 0; i < indices->length(); i++) {
      auto index = indices->Value(i);
      ARROW_RETURN_NOT_OK(builder.Append(values->Value(index - start)));
    }
    return builder.Finish();
  }

  [[nodiscard]] std::string ToString() const {
    return fmt::format("PlainEncoder({})", type_->ToString());
  }

 protected:
  /// @brief Get the actual length to read in the page.
  /// @param start the start offset
  /// @param length proposed length to read
  /// @return the actual length to read
  int32_t GetReadLength(int32_t start, std::optional<int32_t> length) const {
    return std::min(length.value_or(length_), length_ - start);
  }

 private:
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
    auto len = GetReadLength(start, length);
    if (len < 0) {
      return ::arrow::Status::IndexError(fmt::format(
          "PlainDecoder(bool)::ToArray: out of range: start={}, length={}, page_length={}\n",
          start,
          length.value_or(-1),
          length_));
    }
    int64_t byte_length = ::arrow::bit_util::BytesForBits(len);
    ARROW_ASSIGN_OR_RAISE(auto buf, infile_->ReadAt(position_ + start / 8, byte_length));
    return std::make_shared<::arrow::BooleanArray>(len, buf);
  }
};

class FixedSizeListPlainDecoderImpl : public Decoder {
 public:
  FixedSizeListPlainDecoderImpl(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                                std::shared_ptr<::arrow::FixedSizeListType> type,
                                ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : Decoder(infile, type, pool),
        plain_decoder_(infile, type->value_type(), pool),
        list_type_(type) {}

  ::arrow::Status Init() override { return plain_decoder_.Init(); }

  void Reset(int64_t position, int32_t length) override {
    Decoder::Reset(position, length);
    plain_decoder_.Reset(position, length * list_type_->list_size());
  }

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const override {
    if (!length.has_value()) {
      length = length_ - start;
    }

    auto list_size = list_type_->list_size();
    ARROW_ASSIGN_OR_RAISE(auto values,
                          plain_decoder_.ToArray(start * list_size, length.value() * list_size));
    return std::make_shared<::arrow::FixedSizeListArray>(type_, length.value(), values);
  }

 private:
  PlainDecoder plain_decoder_;
  std::shared_ptr<::arrow::FixedSizeListType> list_type_;
};

}  // namespace

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
    case ::arrow::Type::FIXED_SIZE_BINARY:
      impl_.reset(new PlainDecoderImpl<::arrow::FixedSizeBinaryType>(infile_, type_));
      break;
    case ::arrow::Type::FIXED_SIZE_LIST:
      impl_.reset(new FixedSizeListPlainDecoderImpl(
          infile_, std::dynamic_pointer_cast<::arrow::FixedSizeListType>(type_)));
      break;
    default:
      return ::arrow::Status::Invalid(fmt::format("Unsupported type: {}", type_->ToString()));
  }
  return impl_->Init();
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
