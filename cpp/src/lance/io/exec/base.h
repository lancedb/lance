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

#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>
#include <string>
#include <vector>

namespace lance::io::exec {

struct ScanBatch {
  std::shared_ptr<::arrow::RecordBatch> batch;
  int32_t batch_id;

  bool eof() const { return !batch; }
};

/// I/O execute base node.
///
/// TODO: investigate to adapt Arrow Acero.
/// https://arrow.apache.org/docs/cpp/streaming_execution.html
class ExecNode {
 public:
  enum Type {
    kScan = 0,
    kProject = 1,
    kFilter = 2,
    kLimit = 3,
    kTake = 4,
    kTableScan = 256,
  };

  ExecNode() = default;

  virtual ~ExecNode() = default;

  virtual constexpr Type type() const = 0;

  /// Returns the next batch of rows, returns nullptr if EOF.
  virtual ::arrow::Result<ScanBatch> Next() = 0;

  [[nodiscard]] virtual std::string ToString() const = 0;
};

}  // namespace lance::io::exec
