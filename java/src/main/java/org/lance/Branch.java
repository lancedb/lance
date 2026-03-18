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

import java.util.Objects;
import java.util.Optional;

/**
 * Branch metadata aligned with Rust's BranchContents. name is the branch name, parentBranch may be
 * null (indicating main), parentVersion is the version on which the branch was created, createAt is
 * the unix timestamp (seconds), and manifestSize is the size of the referenced manifest file in
 * bytes.
 */
public class Branch {
  private final String name;
  private final Optional<String> parentBranch;
  private final long parentVersion;
  private final long createAt;
  private final int manifestSize;

  public Branch(
      String name, String parentBranch, long parentVersion, long createAt, int manifestSize) {
    this.name = name;
    this.parentBranch = Optional.ofNullable(parentBranch);
    this.parentVersion = parentVersion;
    this.createAt = createAt;
    this.manifestSize = manifestSize;
  }

  public String getName() {
    return name;
  }

  public Optional<String> getParentBranch() {
    return parentBranch;
  }

  public long getParentVersion() {
    return parentVersion;
  }

  public long getCreateAt() {
    return createAt;
  }

  public int getManifestSize() {
    return manifestSize;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("name", name)
        .add("parentBranch", parentBranch)
        .add("parentVersion", parentVersion)
        .add("createAt", createAt)
        .add("manifestSize", manifestSize)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Branch branch = (Branch) o;
    return parentVersion == branch.parentVersion
        && createAt == branch.createAt
        && manifestSize == branch.manifestSize
        && Objects.equals(name, branch.name)
        && Objects.equals(parentBranch, branch.parentBranch);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, parentBranch, parentVersion, createAt, manifestSize);
  }
}
