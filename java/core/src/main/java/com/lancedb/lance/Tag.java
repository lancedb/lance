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
package com.lancedb.lance;

import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.Objects;

public class Tag {
  private final String name;
  private final long version;
  private final int manifestSize;

  public Tag(String name, long version, int manifestSize) {
    this.name = name;
    this.version = version;
    this.manifestSize = manifestSize;
  }

  public String getName() {
    return name;
  }

  public long getVersion() {
    return version;
  }

  public int getManifestSize() {
    return manifestSize;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("name", name)
        .append("version", version)
        .append("manifestSize", manifestSize)
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
        && manifestSize == tag.manifestSize
        && Objects.equals(name, tag.name);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, version, manifestSize);
  }
}
