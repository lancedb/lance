// Copyright 2023 Lance Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Callbacks for duckdb to load lance (rust) code.

#include "extension.h"

const char* lance_version_rust(void);
void lance_init_rust(void* db);

DUCKDB_EXTENSION_API const char* lance_version() {
    return lance_version_rust();
}

DUCKDB_EXTENSION_API void lance_init(void* db) {
    lance_init_rust(db);
}
