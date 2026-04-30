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

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public class Tag {
  private final String name;
  private final Optional<String> branch;
  private final long version;
  private final int manifestSize;
  private final Optional<Instant> createdAt;
  private final Optional<Instant> updatedAt;
  private final Map<String, String> metadata;

  public Tag(String name, String branch, long version, int manifestSize) {
    this(name, branch, version, manifestSize, null, null, Collections.emptyMap());
  }

  public Tag(
      String name, String branch, long version, int manifestSize, Map<String, String> metadata) {
    this(name, branch, version, manifestSize, null, null, metadata);
  }

  public Tag(
      String name,
      String branch,
      long version,
      int manifestSize,
      Instant createdAt,
      Instant updatedAt,
      Map<String, String> metadata) {
    this.name = name;
    this.branch = Optional.ofNullable(branch);
    this.version = version;
    this.manifestSize = manifestSize;
    this.createdAt = Optional.ofNullable(createdAt);
    this.updatedAt = Optional.ofNullable(updatedAt);
    this.metadata = Collections.unmodifiableMap(new HashMap<>(metadata));
  }

  public String getName() {
    return name;
  }

  public Optional<String> getBranch() {
    return branch;
  }

  public long getVersion() {
    return version;
  }

  public int getManifestSize() {
    return manifestSize;
  }

  public Optional<Instant> getCreatedAt() {
    return createdAt;
  }

  public Optional<Instant> getUpdatedAt() {
    return updatedAt;
  }

  public Map<String, String> getMetadata() {
    return metadata;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("name", name)
        .add("branch", branch)
        .add("version", version)
        .add("manifestSize", manifestSize)
        .add("createdAt", createdAt)
        .add("updatedAt", updatedAt)
        .add("metadata", metadata)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Tag tag = (Tag) o;
    return version == tag.version
        && Objects.equals(branch, tag.branch)
        && manifestSize == tag.manifestSize
        && Objects.equals(createdAt, tag.createdAt)
        && Objects.equals(updatedAt, tag.updatedAt)
        && Objects.equals(metadata, tag.metadata)
        && Objects.equals(name, tag.name);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, branch, version, manifestSize, createdAt, updatedAt, metadata);
  }
}
