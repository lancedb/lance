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
package com.lancedb.lance;

import com.google.common.base.MoreObjects;

import java.util.Optional;

/** Options for compacting a dataset. */
public class CompactionOptions {
  private final int targetRowsPerFragment;
  private final int maxRowsPerGroup;
  private final Optional<Integer> maxBytesPerFile;
  private final boolean materializeDeletions;
  private final float materializeDeletionsThreshold;
  private final Optional<Integer> numThreads;
  private final Optional<Integer> batchSize;
  private final boolean deferIndexRemap;

  private CompactionOptions(Builder builder) {
    this.targetRowsPerFragment = builder.targetRowsPerFragment;
    this.maxRowsPerGroup = builder.maxRowsPerGroup;
    this.maxBytesPerFile = builder.maxBytesPerFile;
    this.materializeDeletions = builder.materializeDeletions;
    this.materializeDeletionsThreshold = builder.materializeDeletionsThreshold;
    this.numThreads = builder.numThreads;
    this.batchSize = builder.batchSize;
    this.deferIndexRemap = builder.deferIndexRemap;
  }

  /**
   * Create a new builder for CompactionOptions.
   *
   * @return a new Builder instance
   */
  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private int targetRowsPerFragment = 1_000_000;
    private int maxRowsPerGroup = 1024;
    private Optional<Integer> maxBytesPerFile = Optional.empty();
    private boolean materializeDeletions = true;
    private float materializeDeletionsThreshold = 0.1f;
    private Optional<Integer> numThreads = Optional.empty();
    private Optional<Integer> batchSize = Optional.empty();
    private boolean deferIndexRemap = false;

    private Builder() {}

    /**
     * Set the target number of rows per fragment.
     *
     * @param targetRowsPerFragment target number of rows per fragment (default: 1,000,000)
     * @return this builder
     */
    public Builder setTargetRowsPerFragment(int targetRowsPerFragment) {
      this.targetRowsPerFragment = targetRowsPerFragment;
      return this;
    }

    /**
     * Set the maximum number of rows per group.
     *
     * @param maxRowsPerGroup maximum number of rows per group (default: 1024)
     * @return this builder
     */
    public Builder setMaxRowsPerGroup(int maxRowsPerGroup) {
      this.maxRowsPerGroup = maxRowsPerGroup;
      return this;
    }

    /**
     * Set the maximum number of bytes per file.
     *
     * @param maxBytesPerFile maximum number of bytes per file (optional)
     * @return this builder
     */
    public Builder setMaxBytesPerFile(int maxBytesPerFile) {
      this.maxBytesPerFile = Optional.of(maxBytesPerFile);
      return this;
    }

    /**
     * Set whether to materialize deletions.
     *
     * @param materializeDeletions whether to compact fragments with deletions (default: true)
     * @return this builder
     */
    public Builder setMaterializeDeletions(boolean materializeDeletions) {
      this.materializeDeletions = materializeDeletions;
      return this;
    }

    /**
     * Set the threshold for materializing deletions.
     *
     * @param materializeDeletionsThreshold fraction of rows that need to be deleted before
     *     materializing deletions (default: 0.1)
     * @return this builder
     */
    public Builder setMaterializeDeletionsThreshold(float materializeDeletionsThreshold) {
      this.materializeDeletionsThreshold = materializeDeletionsThreshold;
      return this;
    }

    /**
     * Set the number of threads to use for compaction.
     *
     * @param numThreads number of threads (optional, defaults to CPU count)
     * @return this builder
     */
    public Builder setNumThreads(int numThreads) {
      this.numThreads = Optional.of(numThreads);
      return this;
    }

    /**
     * Set the batch size for scanning input fragments.
     *
     * @param batchSize batch size (optional, uses default if not specified)
     * @return this builder
     */
    public Builder setBatchSize(int batchSize) {
      this.batchSize = Optional.of(batchSize);
      return this;
    }

    /**
     * Set whether to defer index remapping during compaction.
     *
     * @param deferIndexRemap whether to defer index remapping (default: false)
     * @return this builder
     */
    public Builder setDeferIndexRemap(boolean deferIndexRemap) {
      this.deferIndexRemap = deferIndexRemap;
      return this;
    }

    public CompactionOptions build() {
      return new CompactionOptions(this);
    }
  }

  public int getTargetRowsPerFragment() {
    return targetRowsPerFragment;
  }

  public int getMaxRowsPerGroup() {
    return maxRowsPerGroup;
  }

  public Optional<Integer> getMaxBytesPerFile() {
    return maxBytesPerFile;
  }

  public boolean isMaterializeDeletions() {
    return materializeDeletions;
  }

  public float getMaterializeDeletionsThreshold() {
    return materializeDeletionsThreshold;
  }

  public Optional<Integer> getNumThreads() {
    return numThreads;
  }

  public Optional<Integer> getBatchSize() {
    return batchSize;
  }

  public boolean isDeferIndexRemap() {
    return deferIndexRemap;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("targetRowsPerFragment", targetRowsPerFragment)
        .add("maxRowsPerGroup", maxRowsPerGroup)
        .add("maxBytesPerFile", maxBytesPerFile.orElse(null))
        .add("materializeDeletions", materializeDeletions)
        .add("materializeDeletionsThreshold", materializeDeletionsThreshold)
        .add("numThreads", numThreads.orElse(null))
        .add("batchSize", batchSize.orElse(null))
        .add("deferIndexRemap", deferIndexRemap)
        .toString();
  }
}
