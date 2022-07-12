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

namespace lance::encodings {

class PlainEncoder;

/// Dictionary Encoder.
class DictionaryEncoder : public Encoder {
 public:
  DictionaryEncoder(std::shared_ptr<::arrow::io::OutputStream> out);

  virtual ~DictionaryEncoder() = default;

  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::Array> arr) override;

  std::string ToString() const override;

 private:
  /// A plain encoder is used to write index values.
  std::unique_ptr<PlainEncoder> plain_encoder_;
};

}  // namespace lance::encodings