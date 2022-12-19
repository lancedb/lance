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

#include "lance/io/pb.h"

#include <google/protobuf/message.h>

#include <chrono>

namespace lance::io {

/// Write the protobuf message and returns the offset the message was written at.
::arrow::Result<int64_t> WriteProto(const std::shared_ptr<::arrow::io::OutputStream>& sink,
                                    const google::protobuf::Message& pb) {
  ARROW_ASSIGN_OR_RAISE(auto offset, sink->Tell());
  int32_t pb_length = pb.ByteSizeLong();
  ARROW_RETURN_NOT_OK(WriteInt<int32_t>(sink, pb_length));
  ARROW_RETURN_NOT_OK(sink->Write(pb.SerializeAsString()));
  return offset;
}

std::chrono::time_point<std::chrono::system_clock> FromProto(
    const google::protobuf::Timestamp& proto) {
  auto micro_secs = google::protobuf::util::TimeUtil::TimestampToMicroseconds(proto);
  auto dur = std::chrono::microseconds(micro_secs);
  return std::chrono::system_clock::time_point{dur};
}

google::protobuf::Timestamp ToProto(const std::chrono::time_point<std::chrono::system_clock>& ts) {
  auto micro_secs = std::chrono::time_point_cast<std::chrono::microseconds>(ts);
  return google::protobuf::util::TimeUtil::MicrosecondsToTimestamp(
      micro_secs.time_since_epoch().count());
};

}  // namespace lance::io
