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
#include <arrow/type_fwd.h>

#include <concepts>
#include <memory>
#include <optional>
#include <string>

#include "lance/arrow/type.h"
#include "lance/format/format.pb.h"

namespace arrow::io {
class RandomAccessFile;
class OutputStream;
}  // namespace arrow::io

namespace lance::encodings {

template <typename T>
concept ArrowType = std::is_base_of<::arrow::DataType, T>::value;

/// Encoding type Enum
enum Encoding {
  NONE = 0,
  PLAIN = 1,
  VAR_BINARY = 2,
  DICTIONARY = 3,
};

/// Convert protobuf to Encoding type
Encoding FromProto(lance::format::pb::Encoding pb);
/// Convert encoding type to protobuf.
lance::format::pb::Encoding ToProto(Encoding encoding);
/// Convert Encoding to string.
std::string ToString(Encoding encoding);

/// Encoder. Encodes an array and write it to the output stream.
///
class Encoder {
 public:
  Encoder(std::shared_ptr<::arrow::io::OutputStream> out) : out_(out) {}

  virtual ~Encoder() = default;

  /// Write an Arrow Array and returns the start offset of the column metadata.
  ///
  /// \param arr an array to write with the encoding.
  /// \return offset of metadata
  virtual ::arrow::Result<int64_t> Write(const std::shared_ptr<::arrow::Array>& arr) = 0;

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
  /// Constructor of Decoder
  ///
  /// \param infile the opened input file.
  /// \param type the data type of the data on disk. Note that it can not be extension type.
  /// \param pool memory pool.
  Decoder(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
          std::shared_ptr<::arrow::DataType> type,
          ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) noexcept;

  virtual ~Decoder() = default;

  /// Initialize the decoder.
  virtual ::arrow::Status Init();

  virtual void Reset(int64_t position, int32_t length);

  /// Get a Value without scanning the full row group.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Scalar>> GetScalar(int64_t idx) const;

  /// Read the array.
  ///
  /// \param start the start index to read. Must be smaller than the size of the array.
  /// \param length the length of the array to read
  /// \return an array if success.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Array>> ToArray(
      int32_t start = 0, std::optional<int32_t> length = std::nullopt) const = 0;

  /// Take the values by the indices.
  ///
  /// \param indices. The sorted array of indices within the page.
  /// \return an array of value if success.
  ///
  /// The naive implementation is taking each element in parallel.
  /// It is up to the concrete Decoder to override this method to offer higher performance
  /// implementation.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Array>> Take(
      std::shared_ptr<::arrow::Int32Array> indices) const;

 protected:
  /// Make empty array.
  virtual ::arrow::Result<std::shared_ptr<::arrow::Array>> MakeEmpty() const;

  std::shared_ptr<::arrow::io::RandomAccessFile> infile_;
  std::shared_ptr<::arrow::DataType> type_;
  int64_t position_ = -1;
  int32_t length_ = -1;

  ::arrow::MemoryPool* pool_;
};

}  // namespace lance::encodings
