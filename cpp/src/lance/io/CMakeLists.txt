# Copyright 2022 Lance Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Internal I/O

add_subdirectory(exec)

add_library(
  io OBJECT
  endian.h
  pb.cc
  pb.h
  reader.cc
  reader.h
  record_batch_reader.cc
  record_batch_reader.h
  writer.cc
  writer.h)
target_include_directories(io SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
# Depend on lance::format to generate protobuf
add_dependencies(io format lance_io_exec)

add_lance_test(reader_test)
add_lance_test(record_batch_reader_test)
