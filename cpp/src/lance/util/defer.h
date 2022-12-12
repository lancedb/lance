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

#pragma once

#include <functional>

namespace lance::util {

/// Defer the execution of a lambda function until out of scope.
///
/// It offers RAII to general function calls. It is modeled after Golang's Defer statement.
///
/// \example
///
/// void WriteData(const std::string& path) {
///   auto fd = fopen(path.c_str());
///   Defer auto_closer([&]() { fclose(fd); }
///   for (int i = 0; i < 10; i++) {
///       fprint(fd, "line: %d\n", i);
///   }
///   // fd is closed when out of scope.
/// }
///
class Defer {
 public:
  Defer() = delete;

  explicit Defer(std::function<void(void)>&& cb);

  /// Move constructor.
  Defer(Defer&& other);

  ~Defer();

 private:
  std::function<void()> callback_;
};

}  // namespace lance::util