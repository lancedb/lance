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

#include "lance/encodings/binary.h"

#include <arrow/array.h>
#include <arrow/array/array_binary.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>
#include <vector>

using arrow::Result;
using arrow::Status;
using std::shared_ptr;
using std::vector;

namespace lance::encodings {

VarBinaryEncoder::VarBinaryEncoder(std::shared_ptr<::arrow::io::OutputStream> out) noexcept
    : Encoder(out) {}

Result<int64_t> VarBinaryEncoder::Write(const std::shared_ptr<::arrow::Array> data) {
  ARROW_ASSIGN_OR_RAISE(auto start_offset, out_->Tell());
  auto arr = std::static_pointer_cast<::arrow::BinaryArray>(data);
  ARROW_RETURN_NOT_OK(out_->Write(arr->value_data()));

  ARROW_ASSIGN_OR_RAISE(auto offsets_position, out_->Tell());
  offsetBuilder_.Reset();
  assert(arr->length() > 0);
  for (int64_t i = 0; i < arr->length(); ++i) {
    ARROW_RETURN_NOT_OK(offsetBuilder_.Append(start_offset + arr->value_offset(i)));
  }
  ARROW_RETURN_NOT_OK(offsetBuilder_.Append(offsets_position));
  ARROW_RETURN_NOT_OK(offsetBuilder_.Finish(&offsetsArr));
  ARROW_RETURN_NOT_OK(out_->Write(offsetsArr->values()));
  return offsets_position;
}

}  // namespace lance::encodings