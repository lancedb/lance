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

#include <duckdb.hpp>
#include <map>
#include <memory>
#include <mutex>

namespace lance::duckdb::ml {

class ModelEntry {
 public:
  virtual ~ModelEntry() = default;

  /// Model name
  const std::string& name() const { return name_; }

  /// Model URL
  const std::string& uri() const { return uri_; }

  /// Target model device
  const std::string& device() const { return device_; }

  /// Returns the type of the model.
  virtual std::string type() const = 0;

  /// Execute the model over a batch of duckdb data.
  virtual void Execute(::duckdb::DataChunk& args,
                       ::duckdb::ExpressionState& state,
                       ::duckdb::Vector& result) = 0;

 protected:
  /// Construct model entry
  ModelEntry(std::string name, std::string uri, std::string device);

  std::string name_;
  std::string uri_;
  std::string device_;
};

/// Machine Learning Model Catalog.
///
/// Maintain the models available in the session.
class ModelCatalog {
 public:
  /// Get the global singleton of PyTorchCatalog.
  static ModelCatalog* Get();

  /// Load a Model from URI.
  bool Load(const std::string& name, const std::string& uri, const std::string& device);

  /// Get the model by name. Return `nullptr` if not found.
  [[nodiscard]] ModelEntry* Get(const std::string& name) const;

  /// Drop a model.
  void Drop(const std::string& name);

  /// Get the references of all models.
  const std::map<std::string, std::unique_ptr<ModelEntry>>& models() const;

 private:
  ModelCatalog() = default;

  // Singleton
  static std::unique_ptr<ModelCatalog> catalog_;
  static std::mutex mutex_;

  std::map<std::string, std::unique_ptr<ModelEntry>> models_;
};

}  // namespace lance::duckdb::ml
