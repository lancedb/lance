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

#include <arrow/result.h>
#include <arrow/table.h>

#include <memory>
#include <string>

#include "lance/io/exec/base.h"
#include "lance/io/reader.h"

namespace lance::testing {

/// Make lance::io::FileReader from an Arrow Table.
::arrow::Result<std::shared_ptr<lance::io::FileReader>> MakeReader(
    const std::shared_ptr<::arrow::Table>& table);

/// A Dummy ExecNode.
///
/// Used as place holder to replace ExecNode.child.
class DummyNode : lance::io::exec::ExecNode {
 public:
  DummyNode() = default;

  DummyNode(DummyNode&&) = default;

  static std::unique_ptr<io::exec::ExecNode> Make() {
    return std::unique_ptr<io::exec::ExecNode>(new DummyNode());
  }

  ::arrow::Result<io::exec::ScanBatch> Next() override {
    return ::arrow::Status::NotImplemented("DummyNode::Next not implemented");
  }

  std::string ToString() const override { return "Dummy"; }
};

}  // namespace lance::testing
