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

#include <arrow/dataset/scanner.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

/// Read all the content of the file
void ReadAll(const std::string& uri, bool ignore_error = false);

/// Read all the uris at once.
void ReadAll(const std::vector<std::string>& uris,
             std::size_t num_workers = 8,
             bool ignore_error = false);

std::shared_ptr<::arrow::io::RandomAccessFile> OpenUri(const std::string& uri,
                                                       bool ignore_error = false);

/// Open Dataset from the URI.
std::shared_ptr<::arrow::dataset::Dataset> OpenDataset(const std::string& uri,
                                                       const std::string& format = "lance");

std::shared_ptr<::arrow::dataset::Scanner> OpenScanner(
    const std::string& uri,
    const std::vector<std::string>& columns,
    std::optional<arrow::compute::Expression> expr = std::nullopt,
    std::optional<int> batch_size = std::nullopt);
