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

/// Emitted results from each ExecNode
struct ScanBatch {
  /// The resulted RecordBatch.
  ///
  /// If it is zero-sized batch, there is no valid values in this batch.
  /// It it is nullptr, it reaches the end of the scan.
  std::shared_ptr<::arrow::RecordBatch> batch;

  /// The Id of the batch this result belongs to.
  int32_t batch_id = -1;

  /// Indices returned from the filter.
  std::shared_ptr<::arrow::Int32Array> indices;

  /// Return a null ScanBatch indicates EOF.
  static ScanBatch Null();

  /// Constructor with a record batch and batch id.
  ///
  /// \param records A record batch of values to return
  /// \param batch_id the id of the batch
  static ScanBatch Filtered(std::shared_ptr<::arrow::RecordBatch> records,
                            int32_t batch_id,
                            std::shared_ptr<::arrow::Int32Array> indices);

  /// Construct an empty response.
  ScanBatch() = default;

  /// Constructor with a record batch and batch id.
  ///
  /// \param records A record batch of values to return
  /// \param batch_id the id of the batch
  ScanBatch(std::shared_ptr<::arrow::RecordBatch> records, int32_t batch_id);

  /// Returns True if the end of file is reached.
  bool eof() const { return !batch; }
};

/// I/O execute base node.
///
/// TODO: investigate to adapt Arrow Acero.
/// https://arrow.apache.org/docs/cpp/streaming_execution.html
///
/// A exec plan is usually starts with Project and ends with Scan.
///
/// \example
///  A few examples of the exec plan tree for common queries.
///
/// SELECT * FROM dataset
///    Project (*) --> Scan (*)
///
/// SELECT a, b FROM dataset WHERE c = 123
///    Project (a, b) -> Take(a,b) -> Filter(c=123) -> Scan(c)
///
/// SELECT a, b FROM dataset LIMIT 200 OFFSET 5000
///    Project (a, b) -> Limit(200, 5000) -> Scan(a, b)
///
/// SELECT a, b, c FROM dataset WHERE c = 123 LIMIT 50 OFFSET 200
///    Project (a, b, c) -> Take(a, b) -> Limit(50, 200) -> Filter(c=123) -> Scan(c)
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
