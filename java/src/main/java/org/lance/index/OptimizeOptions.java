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

import java.util.List;
import java.util.Optional;

/**
 * Options for optimizing indices on a dataset.
 *
 * <p>This mirrors the behavior of {@code lance_index::optimize::OptimizeOptions} in Rust.
 *
 * <p>All fields are optional on the Java side except {@code retrain}. Defaults are delegated to the
 * Rust implementation.
 */
public class OptimizeOptions {

  private final Optional<Integer> numIndicesToMerge;
  private final Optional<List<String>> indexNames;
  private final boolean retrain;

  private OptimizeOptions(
      Optional<Integer> numIndicesToMerge, Optional<List<String>> indexNames, boolean retrain) {
    this.numIndicesToMerge = numIndicesToMerge;
    this.indexNames = indexNames;
    this.retrain = retrain;
  }

  /** Number of indices to merge per index name. */
  public Optional<Integer> getNumIndicesToMerge() {
    return numIndicesToMerge;
  }

  /**
   * Names of indices to optimize. If empty, all user indices will be considered (system indices are
   * always excluded).
   */
  public Optional<List<String>> getIndexNames() {
    return indexNames;
  }

  /** Whether to retrain the index instead of performing an incremental merge. */
  public boolean isRetrain() {
    return retrain;
  }

  /** Create a new builder for {@link OptimizeOptions}. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link OptimizeOptions}. */
  public static class Builder {
    private Optional<Integer> numIndicesToMerge = Optional.empty();
    private Optional<List<String>> indexNames = Optional.empty();
    private boolean retrain = false;

    private Builder() {}

    /**
     * Set the number of indices to merge.
     *
     * @param numIndicesToMerge number of indices to merge per index name
     */
    public Builder numIndicesToMerge(int numIndicesToMerge) {
      this.numIndicesToMerge = Optional.of(numIndicesToMerge);
      return this;
    }

    /**
     * Restrict optimization to a subset of index names.
     *
     * @param indexNames index names to optimize
     */
    public Builder indexNames(List<String> indexNames) {
      this.indexNames = Optional.ofNullable(indexNames);
      return this;
    }

    /**
     * Whether to retrain the index.
     *
     * @param retrain if true, retrain instead of incremental merge
     */
    public Builder retrain(boolean retrain) {
      this.retrain = retrain;
      return this;
    }

    public OptimizeOptions build() {
      return new OptimizeOptions(numIndicesToMerge, indexNames, retrain);
    }
  }
}
