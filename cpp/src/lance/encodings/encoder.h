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

#include <arrow/result.h>
#include <arrow/status.h>

#include <concepts>
#include <memory>
#include <optional>

namespace arrow::io {
class RandomAccessFile;
class OutputStream;
}  // namespace arrow::io

namespace lance::encodings {

template <typename T>
concept ArrowType = std::is_base_of<::arrow::DataType, T>::value;

/// Encoder. Encodes an array and write it to the output stream.
///
class Encoder {
 public:
  Encoder(std::shared_ptr<::arrow::io::OutputStream> out) : out_(out) {}

  /// Write an Arrow Array and returns the start offset of the column metadata.
  ///
  /// \param arr an array to write with the encoding.
  /// \return offset of metadata
  virtual ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::Array> arr) = 0;

  /// Debug String
  virtual std::string ToString() const = 0;

 protected:
  std::shared_ptr<::arrow::io::OutputStream> out_;
};

/// Decoder base class.
/// Array / column decoder.
///
class Decoder {
 public:
  inline Decoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                 int64_t position,
                 int32_t length) noexcept
      : infile_(infile), position_(position), length_(length){};

  inline Decoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile) noexcept
      : infile_(infile), position_(-1), length_(-1) {}

  virtual ~Decoder() = default;

  virtual void Reset(int64_t position, int32_t length) {
    position_ = position;
    length_ = length;
  }

  /// Get a Value without scanning the full row group.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const = 0;

  /// Read the array.
  ///
  /// \param start the start index to read. Must be smaller than the size of the array.
  /// \param length the length of the array to read
  /// \return an array if success.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const = 0;

 protected:
  std::shared_ptr<::arrow::io::RandomAccessFile> infile_;
  int64_t position_;
  int32_t length_;
};

}  // namespace lance::encodings
