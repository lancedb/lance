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

#include "lance/encodings/fixed_len_binary.h"

namespace lance::encodings {

FixedLenBinaryEncoder::FixedLenBinaryEncoder(const std::shared_ptr<::arrow::io::OutputStream>& out,
                                             uint32_t width) noexcept
    : Encoder(out), width_(width) {}

::arrow::Result<int64_t> FixedLenBinaryEncoder::Write(const std::shared_ptr<::arrow::Array>& arr) {
  ARROW_ASSIGN_OR_RAISE(auto values_position, out_->Tell());
  auto a = std::static_pointer_cast<::arrow::BinaryArray>(arr);
  return 0;
}

}  // namespace lance::encodings