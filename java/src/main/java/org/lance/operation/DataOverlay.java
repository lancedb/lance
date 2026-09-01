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

import org.lance.fragment.DataFile;

import com.google.common.base.MoreObjects;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Attach overlay files that update selected cells without rewriting fragment base files. */
public final class DataOverlay implements Operation {
  private final List<DataOverlayGroup> groups;

  private DataOverlay(List<DataOverlayGroup> groups) {
    this.groups = Objects.requireNonNull(groups);
  }

  public List<DataOverlayGroup> getGroups() {
    return groups;
  }

  @Override
  public String name() {
    return "DataOverlay";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("groups", groups).toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private List<DataOverlayGroup> groups = Collections.emptyList();

    public Builder groups(List<DataOverlayGroup> groups) {
      this.groups = groups;
      return this;
    }

    public DataOverlay build() {
      return new DataOverlay(groups);
    }
  }

  /** Overlay files appended to a single fragment, ordered oldest to newest. */
  public static final class DataOverlayGroup {
    private final long fragmentId;
    private final List<DataOverlayFile> overlays;

    public DataOverlayGroup(long fragmentId, List<DataOverlayFile> overlays) {
      this.fragmentId = fragmentId;
      this.overlays = Objects.requireNonNull(overlays);
    }

    public long getFragmentId() {
      return fragmentId;
    }

    public List<DataOverlayFile> getOverlays() {
      return overlays;
    }
  }

  /** A data file together with the physical-row coverage it supplies. */
  public static final class DataOverlayFile {
    private final DataFile dataFile;
    private final OverlayCoverage coverage;
    private final long committedVersion;

    public DataOverlayFile(DataFile dataFile, OverlayCoverage coverage, long committedVersion) {
      this.dataFile = Objects.requireNonNull(dataFile);
      this.coverage = Objects.requireNonNull(coverage);
      this.committedVersion = committedVersion;
    }

    public DataFile getDataFile() {
      return dataFile;
    }

    public OverlayCoverage getCoverage() {
      return coverage;
    }

    public long getCommittedVersion() {
      return committedVersion;
    }
  }

  /** Portable RoaringBitmap bytes for shared or per-field overlay coverage. */
  public static final class OverlayCoverage {
    private final boolean shared;
    private final List<byte[]> bitmaps;

    private OverlayCoverage(boolean shared, List<byte[]> bitmaps) {
      this.shared = shared;
      this.bitmaps = Objects.requireNonNull(bitmaps);
      if (shared && bitmaps.size() != 1) {
        throw new IllegalArgumentException("shared overlay coverage requires exactly one bitmap");
      }
    }

    public static OverlayCoverage shared(byte[] bitmap) {
      return new OverlayCoverage(true, Collections.singletonList(bitmap));
    }

    public static OverlayCoverage perField(List<byte[]> bitmaps) {
      return new OverlayCoverage(false, new ArrayList<>(bitmaps));
    }

    public boolean isShared() {
      return shared;
    }

    public List<byte[]> getBitmaps() {
      return bitmaps;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      OverlayCoverage that = (OverlayCoverage) o;
      if (shared != that.shared || bitmaps.size() != that.bitmaps.size()) return false;
      for (int i = 0; i < bitmaps.size(); i++) {
        if (!Arrays.equals(bitmaps.get(i), that.bitmaps.get(i))) return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      int result = Boolean.hashCode(shared);
      for (byte[] bitmap : bitmaps) result = 31 * result + Arrays.hashCode(bitmap);
      return result;
    }
  }
}
