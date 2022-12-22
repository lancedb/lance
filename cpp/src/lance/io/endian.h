#pragma once

#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/endian.h>

#include <concepts>
#include <memory>

namespace lance::io {

template <std::integral T>
T ReadInt(const uint8_t* data) {
  T val;
  std::memcpy(&val, data, sizeof(T));
  return ::arrow::bit_util::FromLittleEndian(val);
}

template <std::integral T>
T ReadInt(const ::arrow::Buffer& buf) {
  return ReadInt<T>(buf.data());
}

template <std::integral T>
T ReadInt(const std::shared_ptr<::arrow::Buffer>& buf) {
  return ReadInt<T>(*buf);
}

template <std::integral T>
::arrow::Result<T> ReadInt(const std::shared_ptr<::arrow::io::RandomAccessFile>& source,
                           int64_t offset) {
  T val;
  auto result = source->ReadAt(offset, sizeof(val), &val);
  if (!result.ok()) {
    return result.status();
  }
  return ::arrow::bit_util::FromLittleEndian(val);
}

/// Write an integer to the output stream.
template <std::integral T>
::arrow::Status WriteInt(const std::shared_ptr<::arrow::io::OutputStream>& sink, T value) {
  T v = ::arrow::bit_util::ToLittleEndian(value);
  return sink->Write(&v, sizeof(T));
}

}  // namespace lance::io