/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance;

import com.google.common.base.MoreObjects;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public final class BasePath {
  private final int id;
  private final Optional<String> name;
  private final String path;
  private final boolean isDatasetRoot;
  private final Map<String, String> storageOptions;

  public BasePath(int id, Optional<String> name, String path, boolean isDatasetRoot) {
    this(id, name, path, isDatasetRoot, Collections.emptyMap());
  }

  public BasePath(
      int id,
      Optional<String> name,
      String path,
      boolean isDatasetRoot,
      Map<String, String> storageOptions) {
    this.id = id;
    this.name = name;
    this.path = path;
    this.isDatasetRoot = isDatasetRoot;
    this.storageOptions = Collections.unmodifiableMap(new HashMap<>(storageOptions));
  }

  public int getId() {
    return id;
  }

  public Optional<String> getName() {
    return name;
  }

  public String getPath() {
    return path;
  }

  public boolean isDatasetRoot() {
    return isDatasetRoot;
  }

  public Map<String, String> getStorageOptions() {
    return storageOptions;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("name", name)
        .add("path", path)
        .add("isDatasetRoot", isDatasetRoot)
        .add("storageOptions", storageOptions)
        .toString();
  }
}
