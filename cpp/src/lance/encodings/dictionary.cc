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

#include "lance/encodings/dictionary.h"

#include <arrow/array.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/scalar.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <memory>

#include "lance/encodings/binary.h"
#include "lance/encodings/plain.h"

namespace lance::encodings {

DictionaryEncoder::DictionaryEncoder(std::shared_ptr<::arrow::io::OutputStream> out)
    : Encoder(out), plain_encoder_(std::make_unique<PlainEncoder>(out)) {}

::arrow::Result<int64_t> DictionaryEncoder::Write(const std::shared_ptr<::arrow::Array>& arr) {
  assert(::arrow::is_dictionary(arr->type_id()));
  auto dict_arr = std::dynamic_pointer_cast<::arrow::DictionaryArray>(arr);
  return plain_encoder_->Write(dict_arr->indices());
}

::arrow::Result<int64_t> DictionaryEncoder::WriteValueArray(std::shared_ptr<::arrow::Array> arr) {
  if (::arrow::is_primitive(arr->type_id())) {
    return PlainEncoder(out_).Write(arr);
  } else if (arr->type_id() == ::arrow::Type::STRING) {
    return VarBinaryEncoder(out_).Write(arr);
  }
  return ::arrow::Status::Invalid(
      fmt::format("Does not support dictionary with value type: {}", arr->type()->ToString()));
}

std::string DictionaryEncoder::ToString() const { return "Encoder(type=dictionary)"; }

DictionaryDecoder::DictionaryDecoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                                     std::shared_ptr<::arrow::DictionaryType> type,
                                     std::shared_ptr<::arrow::Array> dict)
    : Decoder(infile, type),
      dict_(dict),
      plain_decoder_(std::make_unique<PlainDecoder>(infile, type->index_type())) {
  assert(dict);
}

::arrow::Status DictionaryDecoder::Init() {
  return plain_decoder_->Init();
}

void DictionaryDecoder::Reset(int64_t position, int32_t length) {
  Decoder::Reset(position, length);
  plain_decoder_->Reset(position, length);
}

::arrow::Result<std::shared_ptr<::arrow::Scalar>> DictionaryDecoder::GetScalar(int64_t idx) const {
  ARROW_ASSIGN_OR_RAISE(auto index_scalar, plain_decoder_->GetScalar(idx));
  return ::arrow::DictionaryScalar::Make(index_scalar, dict_);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> DictionaryDecoder::ToArray(
    int32_t start, std::optional<int32_t> length) const {
  ARROW_ASSIGN_OR_RAISE(auto index_arr, plain_decoder_->ToArray(start, length));
  return ::arrow::DictionaryArray::FromArrays(index_arr, dict_);
}

::arrow::Result<std::shared_ptr<::arrow::Array>> DictionaryDecoder::Take(
    std::shared_ptr<::arrow::Int32Array> indices) const {
  ARROW_ASSIGN_OR_RAISE(auto index_arr, plain_decoder_->Take(indices));
  return ::arrow::DictionaryArray::FromArrays(index_arr, dict_);
}

}  // namespace lance::encodings
