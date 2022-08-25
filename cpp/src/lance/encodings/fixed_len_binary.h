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

#include "lance/encodings/encoder.h"

namespace lance::encodings {

/// Fixed length binary encoder
class FixedLenBinaryEncoder : public Encoder {
 public:
  FixedLenBinaryEncoder(const std::shared_ptr<::arrow::io::OutputStream>& out,
                        uint32_t width) noexcept;

  virtual ~FixedLenBinaryEncoder() = default;

  /// Write an fixed-size array, and returns the offsets to the index block.
  ::arrow::Result<int64_t> Write(const std::shared_ptr<::arrow::Array>& arr) override;

 private:
  uint32_t width_;
};

}  // namespace lance::encodings