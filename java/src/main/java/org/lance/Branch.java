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
import com.google.common.collect.ImmutableList;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Branch metadata aligned with Rust's BranchContents. name is the branch name, parentBranch may be
 * null (indicating main), branchIdentifier is the lineage chain {@code [(version, uuid), ...]},
 * parentVersion is the version on which the branch was created, createAt is the unix timestamp
 * (seconds), and manifestSize is the size of the referenced manifest file in bytes.
 */
public class Branch {
  /** A single lineage hop in the Rust BranchIdentifier.version_mapping vector. */
  public static class BranchVersionMapping {
    private final long version;
    private final String uuid;

    public BranchVersionMapping(long version, String uuid) {
      this.version = version;
      this.uuid = Objects.requireNonNull(uuid);
    }

    public long getVersion() {
      return version;
    }

    public String getUuid() {
      return uuid;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("version", version).add("uuid", uuid).toString();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      BranchVersionMapping that = (BranchVersionMapping) o;
      return version == that.version && Objects.equals(uuid, that.uuid);
    }

    @Override
    public int hashCode() {
      return Objects.hash(version, uuid);
    }
  }

  private final String name;
  private final Optional<String> parentBranch;
  private final ImmutableList<BranchVersionMapping> branchIdentifier;
  private final long parentVersion;
  private final long createAt;
  private final int manifestSize;
  private final Map<String, String> metadata;

  public Branch(
      String name, String parentBranch, long parentVersion, long createAt, int manifestSize) {
    this(
        name,
        parentBranch,
        ImmutableList.of(),
        parentVersion,
        createAt,
        manifestSize,
        Collections.emptyMap());
  }

  public Branch(
      String name,
      String parentBranch,
      List<BranchVersionMapping> branchIdentifier,
      long parentVersion,
      long createAt,
      int manifestSize,
      Map<String, String> metadata) {
    this.name = name;
    this.parentBranch = Optional.ofNullable(parentBranch);
    this.branchIdentifier = ImmutableList.copyOf(Objects.requireNonNull(branchIdentifier));
    this.parentVersion = parentVersion;
    this.createAt = createAt;
    this.manifestSize = manifestSize;
    this.metadata = Collections.unmodifiableMap(new HashMap<>(metadata));
  }

  public String getName() {
    return name;
  }

  public Optional<String> getParentBranch() {
    return parentBranch;
  }

  public ImmutableList<BranchVersionMapping> getBranchIdentifier() {
    return branchIdentifier;
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

  public Map<String, String> getMetadata() {
    return metadata;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("name", name)
        .add("parentBranch", parentBranch)
        .add("branchIdentifier", branchIdentifier)
        .add("parentVersion", parentVersion)
        .add("createAt", createAt)
        .add("manifestSize", manifestSize)
        .add("metadata", metadata)
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
        && Objects.equals(parentBranch, branch.parentBranch)
        && Objects.equals(branchIdentifier, branch.branchIdentifier)
        && Objects.equals(metadata, branch.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        name, parentBranch, branchIdentifier, parentVersion, createAt, manifestSize, metadata);
  }
}
