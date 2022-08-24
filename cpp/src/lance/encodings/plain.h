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

/// Plain Encoder.
///
/// Encoding fixed sized values in an plain array.
class PlainEncoder : public Encoder {
 public:
  explicit PlainEncoder(std::shared_ptr<::arrow::io::OutputStream> out);

  virtual ~PlainEncoder() = default;

  ::arrow::Result<int64_t> Write(const std::shared_ptr<::arrow::Array>& arr) override;

  std::string ToString() const override { return "Encoder(type=Plain)"; }
};

class PlainDecoder : public Decoder {
 public:
  PlainDecoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
               std::shared_ptr<::arrow::DataType> type);

  ~PlainDecoder() override;

  /// Initialize PlainDecoder.
  ::arrow::Status Init() override;

  void Reset(int64_t position, int32_t length) override;

  /// Get one single scalar from the page.
  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override;

  /// Read the buffer as array.
  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const override;

 private:
  std::unique_ptr<Decoder> impl_;
};

}  // namespace lance::encodings
