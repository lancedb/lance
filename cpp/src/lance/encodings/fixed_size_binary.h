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

#include <arrow/array.h>
#include <arrow/io/api.h>

#include <memory>

#include "lance/encodings/encoder.h"

namespace lance::encodings {

/// Fixed size binary decoder.
class FixedSizedBinaryDecoder : public Decoder {
 public:
  using Decoder::Decoder;

  virtual ~FixedSizedBinaryDecoder() = default;

  ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const override;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const override;

  [[nodiscard]] std::string ToString() const;

 private:
  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToFixedSizeBinaryArray(int32_t start,
                                                                          int32_t length) const;

  ::arrow::Result<std::shared_ptr<::arrow::Array>> ToFixedSizeListArray(int32_t start,
                                                                        int32_t length) const;
};

}  // namespace lance::encodings