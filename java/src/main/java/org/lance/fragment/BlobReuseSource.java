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
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 * Maps blob identifiers local to one data file to physical sidecars owned by another data file.
 *
 * <p>Identifiers are represented as {@code long} because the on-disk contract uses unsigned 32-bit
 * values.
 */
public final class BlobReuseSource implements Serializable {
  private static final long serialVersionUID = 1L;

  private final Integer baseId;
  private final String blobDir;
  private final long[] localIds;
  private final long[] physicalIds;

  /** Creates one source mapping. */
  public BlobReuseSource(Integer baseId, String blobDir, long[] localIds, long[] physicalIds) {
    this.baseId = baseId;
    this.blobDir = Objects.requireNonNull(blobDir, "blobDir");
    this.localIds = Objects.requireNonNull(localIds, "localIds").clone();
    this.physicalIds = Objects.requireNonNull(physicalIds, "physicalIds").clone();
  }

  /** Returns the external base, or empty when the containing data file's base is used. */
  public Optional<Integer> getBaseId() {
    return Optional.ofNullable(baseId);
  }

  /** Returns the source data file stem that owns the physical sidecars. */
  public String getBlobDir() {
    return blobDir;
  }

  /** Returns a defensive copy of the local blob identifiers. */
  public long[] getLocalIds() {
    return localIds.clone();
  }

  /** Returns a defensive copy of the corresponding physical blob identifiers. */
  public long[] getPhysicalIds() {
    return physicalIds.clone();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    BlobReuseSource that = (BlobReuseSource) o;
    return Objects.equals(baseId, that.baseId)
        && Objects.equals(blobDir, that.blobDir)
        && Arrays.equals(localIds, that.localIds)
        && Arrays.equals(physicalIds, that.physicalIds);
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(baseId, blobDir);
    result = 31 * result + Arrays.hashCode(localIds);
    result = 31 * result + Arrays.hashCode(physicalIds);
    return result;
  }
}
