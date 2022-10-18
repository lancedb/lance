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

#include "lance/duckdb/ml/catalog.h"

#include <mutex>
#include <utility>

#include "lance/duckdb/ml/pytorch.h"

namespace lance::duckdb::ml {

ModelEntry::ModelEntry(std::string name, std::string uri)
    : name_(std::move(name)), uri_(std::move(uri)) {}

std::unique_ptr<ModelCatalog> ModelCatalog::catalog_;
std::mutex ModelCatalog::mutex_;

ModelCatalog* ModelCatalog::Get() {
  if (catalog_) {
    return catalog_.get();
  }
  std::lock_guard guard(mutex_);
  auto c = std::unique_ptr<ModelCatalog>(new ModelCatalog());
  catalog_ = std::move(c);
  return catalog_.get();
}

bool ModelCatalog::Load(const std::string& name, const std::string& uri) {
  if (models_.contains(name)) {
    return false;
  }
  auto entry = PyTorchModelEntry::Make(name, uri);
  models_.emplace(name, std::move(entry));
  return true;
}

ModelEntry* ModelCatalog::Get(const std::string& name) const {
  auto it = models_.find(name);
  if (it == models_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void ModelCatalog::Drop(const std::string& name) {
  models_.erase(name);
}

const std::map<std::string, std::unique_ptr<ModelEntry>>& ModelCatalog::models() const {
  return models_;
}

}  // namespace lance::duckdb::ml