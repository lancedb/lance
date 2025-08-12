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
package com.lancedb.lance.operation;

/** Restore operation to revert a dataset to a previous version. */
public class Restore implements Operation {
  private final long version;

  private Restore(long version) {
    this.version = version;
  }

  @Override
  public String name() {
    return "Restore";
  }

  /**
   * Get the version to restore to.
   *
   * @return the version number
   */
  public long version() {
    return version;
  }

  @Override
  public String toString() {
    return "Restore{" + "version=" + version + '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Restore that = (Restore) o;
    return version == that.version;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private long version;

    public Builder() {}

    public Builder version(long version) {
      this.version = version;
      return this;
    }

    public Restore build() {
      return new Restore(version);
    }
  }
}
