#  Copyright 2022 Lance Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# Arrow integration

add_library(
        arrow
        OBJECT
        file_lance.cc
        reader.cc
        type.cc
        type.h
        writer.cc
)
target_include_directories(arrow SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})

add_lance_test(api_test)
add_lance_test(arrow_dataset_test)
add_lance_test(writer_test)
add_lance_test(type_test)
