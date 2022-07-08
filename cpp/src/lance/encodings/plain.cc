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

::arrow::Result<int64_t> PlainEncoder::Write(std::shared_ptr<::arrow::Array> arr) {
  auto data_type = arr->type();

  ARROW_ASSIGN_OR_RAISE(auto value_offset, out_->Tell());
  // TODO: support more types.
  switch (data_type->id()) {
    case ::arrow::Type::INT32:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::reinterpret_pointer_cast<::arrow::Int32Array>(arr)->values()));
      break;
    case ::arrow::Type::INT64:
      ARROW_RETURN_NOT_OK(
          out_->Write(std::reinterpret_pointer_cast<::arrow::Int64Array>(arr)->values()));
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

}  // namespace lance::encodings
