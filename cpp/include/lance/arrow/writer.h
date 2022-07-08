// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <arrow/io/type_fwd.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>
#include <lance/arrow/file_lance.h>

#include <optional>

namespace lance::arrow {

/// Write an Arrow Table into the destination.
///
/// \param table arrow table.
/// \param sink the output stream to write it to.
/// \param primary_key the primary key of the dataset.
/// \param options File write options, optional.
///
/// \return Status::OK() if succeed.
::arrow::Status WriteTable(const ::arrow::Table& table,
                           std::shared_ptr<::arrow::io::OutputStream> sink,
                           const std::string& primary_key,
                           std::optional<FileWriteOptions> options = std::nullopt);

}  // namespace nft::arrow