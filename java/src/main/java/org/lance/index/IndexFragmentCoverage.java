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
package org.lance.index;

import java.util.OptionalDouble;

/** Fragment coverage summary for a logical index across all of its physical segments. */
public final class IndexFragmentCoverage {

  private final long coveredFragmentCount;
  private final long currentFragmentCount;
  private final long missingFragmentCount;
  private final long staleFragmentCount;
  private final long fragmentBitmapSizeBytes;

  public IndexFragmentCoverage(
      long coveredFragmentCount,
      long currentFragmentCount,
      long missingFragmentCount,
      long staleFragmentCount,
      long fragmentBitmapSizeBytes) {
    this.coveredFragmentCount = coveredFragmentCount;
    this.currentFragmentCount = currentFragmentCount;
    this.missingFragmentCount = missingFragmentCount;
    this.staleFragmentCount = staleFragmentCount;
    this.fragmentBitmapSizeBytes = fragmentBitmapSizeBytes;
  }

  /** Number of current dataset fragments covered by the index. */
  public long getCoveredFragmentCount() {
    return coveredFragmentCount;
  }

  /** Number of fragments in the current dataset. */
  public long getCurrentFragmentCount() {
    return currentFragmentCount;
  }

  /** Number of current dataset fragments not covered by the index. */
  public long getMissingFragmentCount() {
    return missingFragmentCount;
  }

  /** Number of indexed fragments that are no longer in the current dataset. */
  public long getStaleFragmentCount() {
    return staleFragmentCount;
  }

  /** Serialized size of all physical segment fragment bitmaps in bytes. */
  public long getFragmentBitmapSizeBytes() {
    return fragmentBitmapSizeBytes;
  }

  /**
   * Ratio of covered current fragments to all current fragments.
   *
   * @return empty when the current dataset has no fragments
   */
  public OptionalDouble getCoverageRatio() {
    if (currentFragmentCount == 0) {
      return OptionalDouble.empty();
    }
    return OptionalDouble.of((double) coveredFragmentCount / currentFragmentCount);
  }
}
