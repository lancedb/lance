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
package org.lance.operation;

import org.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public class Update implements Operation {
  private final List<Long> removedFragmentIds;
  private final List<FragmentMetadata> updatedFragments;
  private final List<FragmentMetadata> newFragments;
  private final long[] fieldsModified;
  private final long[] fieldsForPreservingFragBitmap;
  private final Optional<UpdateMode> updateMode;
  private final Optional<Map<Long, long[]>> updatedRowOffsets;

  private Update(
      List<Long> removedFragmentIds,
      List<FragmentMetadata> updatedFragments,
      List<FragmentMetadata> newFragments,
      long[] fieldsModified,
      long[] fieldsForPreservingFragBitmap,
      Optional<UpdateMode> updateMode,
      Optional<Map<Long, long[]>> updatedRowOffsets) {
    this.removedFragmentIds = removedFragmentIds;
    this.updatedFragments = updatedFragments;
    this.newFragments = newFragments;
    this.fieldsModified = fieldsModified;
    this.fieldsForPreservingFragBitmap = fieldsForPreservingFragBitmap;
    this.updateMode = updateMode;
    this.updatedRowOffsets = updatedRowOffsets;
  }

  public static Builder builder() {
    return new Builder();
  }

  public List<Long> removedFragmentIds() {
    return removedFragmentIds;
  }

  public List<FragmentMetadata> updatedFragments() {
    return updatedFragments;
  }

  public List<FragmentMetadata> newFragments() {
    return newFragments;
  }

  public long[] fieldsModified() {
    return fieldsModified;
  }

  public long[] fieldsForPreservingFragBitmap() {
    return fieldsForPreservingFragBitmap;
  }

  public Optional<UpdateMode> updateMode() {
    return updateMode;
  }

  /**
   * Optional per-fragment map of updated row offsets (physical row indices within each fragment).
   * Used for RewriteColumns mode to selectively update _row_last_updated_at_version.
   * Key: fragment ID, Value: sorted, deduplicated local physical row offsets that were updated.
   * If empty, all rows in updated_fragments are assumed to be updated.
   */
  public Optional<Map<Long, long[]>> updatedRowOffsets() {
    return updatedRowOffsets;
  }

  @Override
  public String name() {
    return "Update";
  }

  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("removedFragmentIds", removedFragmentIds)
        .add("updatedFragments", updatedFragments)
        .add("newFragments", newFragments)
        .add("fieldsModified", fieldsModified)
        .add("fieldsForPreservingFragBitmap", fieldsForPreservingFragBitmap)
        .add("updateMode", updateMode)
        .add("updatedRowOffsets", updatedRowOffsets)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Update that = (Update) o;
    return Objects.equals(removedFragmentIds, that.removedFragmentIds)
        && Objects.equals(updatedFragments, that.updatedFragments)
        && Objects.equals(newFragments, that.newFragments)
        && Arrays.equals(fieldsModified, that.fieldsModified)
        && Arrays.equals(fieldsForPreservingFragBitmap, that.fieldsForPreservingFragBitmap)
        && Objects.equals(updateMode, that.updateMode)
        && Objects.equals(updatedRowOffsets, that.updatedRowOffsets);
  }

  public enum UpdateMode {
    RewriteRows,
    RewriteColumns;
  }

  public static class Builder {
    private List<Long> removedFragmentIds = Collections.emptyList();
    private List<FragmentMetadata> updatedFragments = Collections.emptyList();
    private List<FragmentMetadata> newFragments = Collections.emptyList();
    private long[] fieldsModified = new long[0];
    private long[] fieldsForPreservingFragBitmap = new long[0];
    private Optional<UpdateMode> updateMode = Optional.empty();
    private Optional<Map<Long, long[]>> updatedRowOffsets = Optional.empty();

    private Builder() {}

    public Builder removedFragmentIds(List<Long> removedFragmentIds) {
      this.removedFragmentIds = removedFragmentIds;
      return this;
    }

    public Builder updatedFragments(List<FragmentMetadata> updatedFragments) {
      this.updatedFragments = updatedFragments;
      return this;
    }

    public Builder newFragments(List<FragmentMetadata> newFragments) {
      this.newFragments = newFragments;
      return this;
    }

    public Builder fieldsModified(long[] fieldsModified) {
      this.fieldsModified = fieldsModified;
      return this;
    }

    public Builder fieldsForPreservingFragBitmap(long[] fieldsForPreservingFragBitmap) {
      this.fieldsForPreservingFragBitmap = fieldsForPreservingFragBitmap;
      return this;
    }

    public Builder updateMode(Optional<UpdateMode> updateMode) {
      this.updateMode = updateMode;
      return this;
    }

    /**
     * Set per-fragment updated row offsets for RewriteColumns mode.
     * @param updatedRowOffsets Map from fragment ID to sorted physical row offsets that were updated.
     */
    public Builder updatedRowOffsets(Optional<Map<Long, long[]>> updatedRowOffsets) {
      this.updatedRowOffsets = updatedRowOffsets;
      return this;
    }

    public Update build() {
      return new Update(
          removedFragmentIds,
          updatedFragments,
          newFragments,
          fieldsModified,
          fieldsForPreservingFragBitmap,
          updateMode,
          updatedRowOffsets);
    }
  }
}
