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
package org.lance.ipc;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Summary counts collected after executing a scan plan.
 *
 * <p>These statistics are populated when the scan stream is fully consumed and closed.
 */
public final class ScanStats {
  private final long iops;
  private final long requests;
  private final long bytesRead;
  private final long indicesLoaded;
  private final long partsLoaded;
  private final long indexComparisons;
  private final Map<String, Long> allCounts;
  private final Map<String, Long> allTimes;

  public ScanStats(
      long iops,
      long requests,
      long bytesRead,
      long indicesLoaded,
      long partsLoaded,
      long indexComparisons,
      Map<String, Long> allCounts,
      Map<String, Long> allTimes) {
    this.iops = iops;
    this.requests = requests;
    this.bytesRead = bytesRead;
    this.indicesLoaded = indicesLoaded;
    this.partsLoaded = partsLoaded;
    this.indexComparisons = indexComparisons;
    this.allCounts = freezeMap(allCounts);
    this.allTimes = freezeMap(allTimes);
  }

  private static <K, V> Map<K, V> freezeMap(Map<K, V> map) {
    if (map == null || map.isEmpty()) {
      return Collections.emptyMap();
    }
    return Collections.unmodifiableMap(new HashMap<>(map));
  }

  public long getIops() {
    return iops;
  }

  public long getRequests() {
    return requests;
  }

  public long getBytesRead() {
    return bytesRead;
  }

  public long getIndicesLoaded() {
    return indicesLoaded;
  }

  public long getPartsLoaded() {
    return partsLoaded;
  }

  public long getIndexComparisons() {
    return indexComparisons;
  }

  public Map<String, Long> getAllCounts() {
    return allCounts;
  }

  public Map<String, Long> getAllTimes() {
    return allTimes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof ScanStats)) {
      return false;
    }
    ScanStats that = (ScanStats) o;
    return iops == that.iops
        && requests == that.requests
        && bytesRead == that.bytesRead
        && indicesLoaded == that.indicesLoaded
        && partsLoaded == that.partsLoaded
        && indexComparisons == that.indexComparisons
        && Objects.equals(allCounts, that.allCounts)
        && Objects.equals(allTimes, that.allTimes);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        iops,
        requests,
        bytesRead,
        indicesLoaded,
        partsLoaded,
        indexComparisons,
        allCounts,
        allTimes);
  }

  @Override
  public String toString() {
    return "ScanStats{"
        + "iops="
        + iops
        + ", requests="
        + requests
        + ", bytesRead="
        + bytesRead
        + ", indicesLoaded="
        + indicesLoaded
        + ", partsLoaded="
        + partsLoaded
        + ", indexComparisons="
        + indexComparisons
        + ", allCounts="
        + allCounts
        + ", allTimes="
        + allTimes
        + '}';
  }
}
