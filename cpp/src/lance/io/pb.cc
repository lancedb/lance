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

}  // namespace lance::io
