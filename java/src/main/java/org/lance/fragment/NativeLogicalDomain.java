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
package org.lance.fragment;

import java.io.Serializable;
import java.util.Objects;

/** Stable logical domain created with a storage-version-2.3 fragment. */
public final class NativeLogicalDomain implements Serializable {
  private final int logicalFragmentId;
  private final long creationVersion;

  /** Create immutable native logical-domain metadata. */
  public NativeLogicalDomain(int logicalFragmentId, long creationVersion) {
    this.logicalFragmentId = logicalFragmentId;
    this.creationVersion = creationVersion;
  }

  /** Return the dataset-unique logical fragment ID. */
  public int getLogicalFragmentId() {
    return logicalFragmentId;
  }

  /** Return the dataset version that allocated the logical domain. */
  public long getCreationVersion() {
    return creationVersion;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof NativeLogicalDomain)) {
      return false;
    }
    NativeLogicalDomain that = (NativeLogicalDomain) other;
    return logicalFragmentId == that.logicalFragmentId && creationVersion == that.creationVersion;
  }

  @Override
  public int hashCode() {
    return Objects.hash(logicalFragmentId, creationVersion);
  }
}
