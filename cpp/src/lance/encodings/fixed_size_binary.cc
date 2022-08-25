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

#include "lance/encodings/fixed_size_binary.h"

#include "lance/arrow/type.h"
#include "lance/encodings/encoder.h"

namespace lance::encodings {

FixedSizeBinaryEncoder::FixedSizeBinaryEncoder(
    const std::shared_ptr<::arrow::io::OutputStream>& out) noexcept
    : Encoder(out) {}

FixedSizeBinaryEncoder::~FixedSizeBinaryEncoder() {}

::arrow::Result<int64_t> FixedSizeBinaryEncoder::Write(const std::shared_ptr<::arrow::Array>& arr) {
  assert(::arrow::is_fixed_size_binary(arr->type_id()) ||
         lance::arrow::is_fixed_size_list(arr->type()));

  ARROW_ASSIGN_OR_RAISE(auto values_position, out_->Tell());
  auto a = std::static_pointer_cast<::arrow::FixedSizeBinaryArray>(arr);
  ARROW_RETURN_NOT_OK(out_->Write(a->raw_values(), a->length() * a->byte_width()));

  return values_position;
}

std::string FixedSizeBinaryEncoder::ToString() const { return "FixedSizeBinaryEncoder"; }

}  // namespace lance::encodings