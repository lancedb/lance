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

/** Aggregate statistics for the fragments in a dataset version. */
public final class FragmentSummary {
  private final long fragmentCount;
  private final long minRowsPerFragment;
  private final long maxRowsPerFragment;
  private final long minDataFilesPerFragment;
  private final long maxDataFilesPerFragment;

  FragmentSummary(
      long fragmentCount,
      long minRowsPerFragment,
      long maxRowsPerFragment,
      long minDataFilesPerFragment,
      long maxDataFilesPerFragment) {
    this.fragmentCount = fragmentCount;
    this.minRowsPerFragment = minRowsPerFragment;
    this.maxRowsPerFragment = maxRowsPerFragment;
    this.minDataFilesPerFragment = minDataFilesPerFragment;
    this.maxDataFilesPerFragment = maxDataFilesPerFragment;
  }

  /** Number of fragments. */
  public long getFragmentCount() {
    return fragmentCount;
  }

  /** Minimum number of live rows in a fragment, or 0 when there are no fragments. */
  public long getMinRowsPerFragment() {
    return minRowsPerFragment;
  }

  /** Maximum number of live rows in a fragment, or 0 when there are no fragments. */
  public long getMaxRowsPerFragment() {
    return maxRowsPerFragment;
  }

  /** Minimum number of data files in a fragment, or 0 when there are no fragments. */
  public long getMinDataFilesPerFragment() {
    return minDataFilesPerFragment;
  }

  /** Maximum number of data files in a fragment, or 0 when there are no fragments. */
  public long getMaxDataFilesPerFragment() {
    return maxDataFilesPerFragment;
  }
}
