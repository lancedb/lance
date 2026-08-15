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
import org.lance.memwal.CompactedSsTable;

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

  /**
   * Per-fragment matched row offsets serialized as portable RoaringBitmap bytes (little-endian,
   * spec-compliant). Keys are fragment ids; values are the serialized bitmap for the local physical
   * row offsets (0-based) within the fragment whose columns were rewritten. Empty map means the
   * caller did not supply offsets and the partial last_updated refresh in build_manifest will not
   * activate.
   */
  private final Map<Long, byte[]> updatedFragmentOffsets;

  /** MemWAL SSTables to mark as compacted after this transaction. */
  private final List<CompactedSsTable> compactedSstables;

  /**
   * Serialized protobuf bytes of {@code lance.table.transaction.KeyExistenceFilter} for detecting
   * conflicts on inserted row keys during merge insert.
   */
  private final byte[] insertedRowsFilter;

  private Update(
      List<Long> removedFragmentIds,
      List<FragmentMetadata> updatedFragments,
      List<FragmentMetadata> newFragments,
      long[] fieldsModified,
      long[] fieldsForPreservingFragBitmap,
      Optional<UpdateMode> updateMode,
      Map<Long, byte[]> updatedFragmentOffsets,
      List<CompactedSsTable> compactedSstables,
      byte[] insertedRowsFilter) {
    this.removedFragmentIds =
        removedFragmentIds != null ? removedFragmentIds : Collections.emptyList();
    this.updatedFragments = updatedFragments != null ? updatedFragments : Collections.emptyList();
    this.newFragments = newFragments != null ? newFragments : Collections.emptyList();
    this.fieldsModified = fieldsModified != null ? fieldsModified : new long[0];
    this.fieldsForPreservingFragBitmap =
        fieldsForPreservingFragBitmap != null ? fieldsForPreservingFragBitmap : new long[0];
    this.updateMode = updateMode != null ? updateMode : Optional.empty();
    this.updatedFragmentOffsets =
        updatedFragmentOffsets != null ? updatedFragmentOffsets : Collections.emptyMap();
    this.compactedSstables =
        compactedSstables != null ? compactedSstables : Collections.emptyList();
    this.insertedRowsFilter = insertedRowsFilter;
  }

  private Update(
      List<Long> removedFragmentIds,
      List<FragmentMetadata> updatedFragments,
      List<FragmentMetadata> newFragments,
      long[] fieldsModified,
      long[] fieldsForPreservingFragBitmap,
      Optional<UpdateMode> updateMode,
      Map<Long, byte[]> updatedFragmentOffsets) {
    this(
        removedFragmentIds,
        updatedFragments,
        newFragments,
        fieldsModified,
        fieldsForPreservingFragBitmap,
        updateMode,
        updatedFragmentOffsets,
        Collections.emptyList(),
        null);
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

  public Map<Long, byte[]> updatedFragmentOffsets() {
    return updatedFragmentOffsets;
  }

  public List<CompactedSsTable> compactedSstables() {
    return compactedSstables;
  }

  public List<CompactedSsTable> getCompactedSstables() {
    return compactedSstables;
  }

  public byte[] insertedRowsFilter() {
    return insertedRowsFilter;
  }

  public byte[] getInsertedRowsFilter() {
    return insertedRowsFilter;
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
        .add("updatedFragmentOffsets", updatedFragmentOffsets)
        .add("compactedSstables", compactedSstables)
        .add("hasInsertedRowsFilter", insertedRowsFilter != null)
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
        && offsetMapsEqual(updatedFragmentOffsets, that.updatedFragmentOffsets)
        && Objects.equals(compactedSstables, that.compactedSstables)
        && Arrays.equals(insertedRowsFilter, that.insertedRowsFilter);
  }

  /** Deep-equality for {@code Map<Long, byte[]>}: keys by value, arrays by content. */
  private static boolean offsetMapsEqual(Map<Long, byte[]> a, Map<Long, byte[]> b) {
    if (a == b) return true;
    if (a.size() != b.size()) return false;
    for (Map.Entry<Long, byte[]> entry : a.entrySet()) {
      if (!Arrays.equals(entry.getValue(), b.get(entry.getKey()))) return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    int h =
        Objects.hash(
            removedFragmentIds, updatedFragments, newFragments, updateMode, compactedSstables);
    h = 31 * h + Arrays.hashCode(fieldsModified);
    h = 31 * h + Arrays.hashCode(fieldsForPreservingFragBitmap);
    h = 31 * h + Arrays.hashCode(insertedRowsFilter);
    // Sum entry hashes (XOR key ^ array-content hash) so result is insertion-order-independent.
    int mapHash = 0;
    for (Map.Entry<Long, byte[]> entry : updatedFragmentOffsets.entrySet()) {
      mapHash += Long.hashCode(entry.getKey()) ^ Arrays.hashCode(entry.getValue());
    }
    h = 31 * h + mapHash;
    return h;
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
    private Map<Long, byte[]> updatedFragmentOffsets = Collections.emptyMap();
    private List<CompactedSsTable> compactedSstables = Collections.emptyList();
    private byte[] insertedRowsFilter = null;

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
     * Set the per-fragment matched row offsets for a RewriteColumns commit.
     *
     * <p>Keys are fragment ids; values are portable RoaringBitmap bytes (little-endian,
     * spec-compliant serialization) encoding the local physical row offsets (0-based) within the
     * fragment that matched the update_columns hash join. When non-empty and update mode is
     * RewriteColumns with stable row IDs enabled, build_manifest will call the partial last_updated
     * refresh for those offsets only.
     */
    public Builder updatedFragmentOffsets(Map<Long, byte[]> updatedFragmentOffsets) {
      this.updatedFragmentOffsets = updatedFragmentOffsets;
      return this;
    }

    public Builder compactedSstables(List<CompactedSsTable> compactedSstables) {
      this.compactedSstables = compactedSstables;
      return this;
    }

    public Builder insertedRowsFilter(byte[] insertedRowsFilter) {
      this.insertedRowsFilter = insertedRowsFilter;
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
          updatedFragmentOffsets,
          compactedSstables,
          insertedRowsFilter);
    }
  }
}
