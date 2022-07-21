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

#include <memory>
#include <string>

#include "lance/encodings/encoder.h"
#include "lance/encodings/plain.h"

namespace lance::encodings {

/// Dictionary Encoder.
class DictionaryEncoder : public Encoder {
 public:
  DictionaryEncoder(std::shared_ptr<::arrow::io::OutputStream> out);

  virtual ~DictionaryEncoder() = default;

  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::Array> arr) override;

  /// Write value array.
  ///
  /// It should be only called once per dataset / file.
  ::arrow::Result<int64_t> WriteValueArray(std::shared_ptr<::arrow::Array> arr);

  std::string ToString() const override;

 private:
  /// A plain encoder is used to write index values.
  std::unique_ptr<PlainEncoder> plain_encoder_;
};

/// Dictionary Decoder.
class DictionaryDecoder : public Decoder {
 public:
  /// Constructor for DictionaryDecoder.
  ///
  /// \param infile input file.
  /// \param type data type.
  /// \param dict the dictionary array.
  ///
  /// See https://arrow.apache.org/docs/cpp/api/array.html#dictionary-encoded for details w.r.t
  /// of DictionaryType.
  DictionaryDecoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                    std::shared_ptr<::arrow::DictionaryType> type,
                    std::shared_ptr<::arrow::Array> dict);

  ~DictionaryDecoder() override = default;

  ::arrow::Status Init() override;

  void Reset(int64_t position, int32_t length) override;

  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const override;

 private:
  std::shared_ptr<::arrow::Array> dict_;
  std::unique_ptr<PlainDecoder> plain_decoder_;
};

}  // namespace lance::encodings