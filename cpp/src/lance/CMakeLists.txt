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

add_subdirectory(arrow)
add_subdirectory(encodings)
add_subdirectory(format)
add_subdirectory(io)
add_subdirectory(util)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_subdirectory(testing)
endif ()

if (LANCE_BUILD_CMD)
    add_subdirectory(cmd)
endif ()