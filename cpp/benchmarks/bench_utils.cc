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

#include "bench_utils.h"

#include <arrow/dataset/discovery.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/io/api.h>
#include <arrow/util/string.h>
#include <fmt/format.h>
#include <lance/arrow/file_lance.h>

#include <filesystem>
#include <future>

std::shared_ptr<::arrow::dataset::Scanner> OpenScanner(
    const std::string& uri,
    const std::vector<std::string>& columns,
    std::optional<arrow::compute::Expression> filter,
    std::optional<int> batch_size) {
  auto format = std::shared_ptr<arrow::dataset::FileFormat>();
  if (uri.ends_with(".lance")) {
    format = lance::arrow::LanceFileFormat::Make();
  } else if (uri.ends_with(".parquet") || uri.ends_with(".parq")) {
    format.reset(new arrow::dataset::ParquetFileFormat());
  }
  auto factory = ::arrow::dataset::FileSystemDatasetFactory::Make(
                     uri, format, arrow::dataset::FileSystemFactoryOptions())
                     .ValueOrDie();
  auto dataset = factory->Finish().ValueOrDie();
  // fmt::print("{} dataset: {} groups={}\n", format->type_name(), uri,
  // dataset->schema()->ToString());
  auto scan_builder = dataset->NewScan().ValueOrDie();
  if (batch_size.has_value()) {
    fmt::print("Setting batch size: {}\n", batch_size.value());
    assert(scan_builder->BatchSize(batch_size.value()).ok());
  }
  if (filter.has_value()) {
    assert(scan_builder->Filter(filter.value()).ok());
  }
  assert(scan_builder->Project(columns).ok());
  assert(scan_builder->UseThreads().ok());
  auto scanner = scan_builder->Finish().ValueOrDie();
  /// TODO: somehow scanner builder does not honor batch size??
  scanner->options()->batch_size = batch_size.value();
  if (format->type_name() == "lance") {
    /// set to 16 will crash EC2 instance.
    scanner->options()->batch_readahead = 8;
  }
  return scan_builder->Finish().ValueOrDie();
}

std::shared_ptr<::arrow::io::RandomAccessFile> OpenUri(const std::string& uri, bool ignore_error) {
  if (uri.starts_with("s3")) {
    auto fs = arrow::fs::FileSystemFromUriOrPath(uri).ValueOrDie();
    auto file = fs->OpenInputFile(uri.substr(std::string("s3://").size()));
    if (!file.ok() && ignore_error) {
      return nullptr;
    }
    return file.ValueOrDie();
  } else {
    auto path = std::filesystem::absolute(uri);
    auto fs = arrow::fs::FileSystemFromUriOrPath(path).ValueOrDie();
    auto file = fs->OpenInputFile(path);
    if (!file.ok() && ignore_error) {
      return nullptr;
    }
    return file.ValueOrDie();
  }
}

/// Open Dataset from the URI.
std::shared_ptr<::arrow::dataset::FileSystemDataset> OpenDataset(const std::string& uri,
                                                                 const std::string& format) {
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(uri, &path).ValueOrDie();
  std::shared_ptr<arrow::dataset::FileFormat> file_format;
  if (format == "lance") {
    file_format.reset(new lance::arrow::LanceFileFormat());
  } else if (format == "parquet") {
    file_format.reset(new arrow::dataset::ParquetFileFormat());
  } else {
    fmt::print(stderr, "Unsupported file format: {}\n", format);
    assert(false);
  }
  auto selector = ::arrow::fs::FileSelector();
  selector.base_dir = path;
  selector.recursive = true;
  auto factory = arrow::dataset::FileSystemDatasetFactory::Make(
                     fs, selector, file_format, arrow::dataset::FileSystemFactoryOptions())
                     .ValueOrDie();
  auto dataset = factory->Finish().ValueOrDie();
  return std::dynamic_pointer_cast<::arrow::dataset::FileSystemDataset>(dataset);
}

void ReadAll(const std::string& uri, bool ignore_error) {
  auto infile = OpenUri(uri, ignore_error);
  if (infile) {
    auto buf = infile->Read(infile->GetSize().ValueOrDie()).ValueOrDie();
  }
}

void ReadAll(const std::vector<std::string>& uris, size_t num_workers, bool ignore_error) {
  auto total = uris.size();
  for (decltype(total) start = 0; start < total; start += num_workers) {
    std::vector<std::future<void>> futures;
    auto len = std::min(num_workers, total - start);
    for (size_t i = 0; i < len; i++) {
      auto& uri = uris[start + i];
      futures.emplace_back(std::async(
          std::launch::async,
          [](const std::string& uri, bool e) {
            auto infile = OpenUri(uri, e);
            if (infile) {
              auto buf = infile->Read(infile->GetSize().ValueOrDie()).ValueOrDie();
            }
            return;
          },
          uri,
          ignore_error));
    }
    for (auto& f : futures) {
      f.wait();
    }
  }
}