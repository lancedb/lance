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
package com.lancedb.lance.compaction;

import com.google.common.base.MoreObjects;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Optional;

/**
 * Compaction options. All fields are optional in Java side. We do not set default value for them to
 * avoid conflicting with those default values in Rust side. Please check the <a
 * href="https://docs.rs/lance/latest/src/lance/dataset/optimize.rs.html#118">rust code</a> for all
 * default values.
 */
public class CompactionOptions implements Serializable {
  // these fields are effectively final, but not marked as final for de/ser
  private Optional<Long> targetRowsPerFragment;
  private Optional<Long> maxRowsPerGroup;
  private Optional<Long> maxBytesPerFile;
  private Optional<Boolean> materializeDeletions;
  private Optional<Float> materializeDeletionsThreshold;
  private Optional<Long> numThreads;
  private Optional<Long> batchSize;
  private Optional<Boolean> deferIndexRemap;

  private CompactionOptions(
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap) {
    this.targetRowsPerFragment = targetRowsPerFragment;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
    this.materializeDeletions = materializeDeletions;
    this.materializeDeletionsThreshold = materializeDeletionsThreshold;
    this.numThreads = numThreads;
    this.batchSize = batchSize;
    this.deferIndexRemap = deferIndexRemap;
  }

  public Optional<Boolean> getDeferIndexRemap() {
    return deferIndexRemap;
  }

  public Optional<Boolean> getMaterializeDeletions() {
    return materializeDeletions;
  }

  public Optional<Float> getMaterializeDeletionsThreshold() {
    return materializeDeletionsThreshold;
  }

  public Optional<Long> getBatchSize() {
    return batchSize;
  }

  public Optional<Long> getMaxBytesPerFile() {
    return maxBytesPerFile;
  }

  public Optional<Long> getMaxRowsPerGroup() {
    return maxRowsPerGroup;
  }

  public Optional<Long> getNumThreads() {
    return numThreads;
  }

  public Optional<Long> getTargetRowsPerFragment() {
    return targetRowsPerFragment;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("targetRowsPerFragment", targetRowsPerFragment.orElse(null))
        .add("maxRowsPerGroup", maxRowsPerGroup.orElse(null))
        .add("maxBytesPerFile", maxBytesPerFile.orElse(null))
        .add("materializeDeletions", materializeDeletions.orElse(null))
        .add("materializeDeletionsThreshold", materializeDeletionsThreshold.orElse(null))
        .add("numThreads", numThreads.orElse(null))
        .add("batchSize", batchSize.orElse(null))
        .add("deferIndexRemap", deferIndexRemap.orElse(null))
        .toString();
  }

  private void writeObject(ObjectOutputStream output) throws IOException {
    output.writeObject(targetRowsPerFragment.orElse(null));
    output.writeObject(maxRowsPerGroup.orElse(null));
    output.writeObject(maxBytesPerFile.orElse(null));
    output.writeObject(materializeDeletions.orElse(null));
    output.writeObject(materializeDeletionsThreshold.orElse(null));
    output.writeObject(numThreads.orElse(null));
    output.writeObject(batchSize.orElse(null));
    output.writeObject(deferIndexRemap.orElse(null));
  }

  private void readObject(ObjectInputStream input) throws IOException, ClassNotFoundException {
    this.targetRowsPerFragment = Optional.ofNullable((Long) input.readObject());
    this.maxRowsPerGroup = Optional.ofNullable((Long) input.readObject());
    this.maxBytesPerFile = Optional.ofNullable((Long) input.readObject());
    this.materializeDeletions = Optional.ofNullable((Boolean) input.readObject());
    this.materializeDeletionsThreshold = Optional.ofNullable((Float) input.readObject());
    this.numThreads = Optional.ofNullable((Long) input.readObject());
    this.batchSize = Optional.ofNullable((Long) input.readObject());
    this.deferIndexRemap = Optional.ofNullable((Boolean) input.readObject());
  }

  /** Builder for CompactionOptions. */
  public static class Builder {
    private Optional<Long> targetRowsPerFragment = Optional.empty();
    private Optional<Long> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
    private Optional<Boolean> materializeDeletions = Optional.empty();
    private Optional<Float> materializeDeletionsThreshold = Optional.empty();
    private Optional<Long> numThreads = Optional.empty();
    private Optional<Long> batchSize = Optional.empty();
    private Optional<Boolean> deferIndexRemap = Optional.empty();

    private Builder() {}

    public Builder withTargetRowsPerFragment(long targetRowsPerFragment) {
      this.targetRowsPerFragment = Optional.of(targetRowsPerFragment);
      return this;
    }

    public Builder withMaxRowsPerGroup(long maxRowsPerGroup) {
      this.maxRowsPerGroup = Optional.of(maxRowsPerGroup);
      return this;
    }

    public Builder withMaxBytesPerFile(long maxBytesPerFile) {
      this.maxBytesPerFile = Optional.of(maxBytesPerFile);
      return this;
    }

    public Builder withMaterializeDeletions(boolean materializeDeletions) {
      this.materializeDeletions = Optional.of(materializeDeletions);
      return this;
    }

    public Builder withMaterializeDeletionsThreshold(float materializeDeletionsThreshold) {
      this.materializeDeletionsThreshold = Optional.of(materializeDeletionsThreshold);
      return this;
    }

    public Builder withNumThreads(long numThreads) {
      this.numThreads = Optional.of(numThreads);
      return this;
    }

    public Builder withBatchSize(long batchSize) {
      this.batchSize = Optional.of(batchSize);
      return this;
    }

    public Builder withDeferIndexRemap(boolean deferIndexRemap) {
      this.deferIndexRemap = Optional.of(deferIndexRemap);
      return this;
    }

    public CompactionOptions build() {
      return new CompactionOptions(
          targetRowsPerFragment,
          maxRowsPerGroup,
          maxBytesPerFile,
          materializeDeletions,
          materializeDeletionsThreshold,
          numThreads,
          batchSize,
          deferIndexRemap);
    }
  }
}
