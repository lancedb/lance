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

package com.lancedb.lance.index.vector;

import org.apache.commons.lang3.builder.ToStringBuilder;

public class IvfBuildParams {
  private final int numPartitions;
  private final int maxIters;
  private final int sampleRate;
  private final int shufflePartitionBatches;
  private final int shufflePartitionConcurrency;
  private final boolean useResidual;

  private IvfBuildParams(Builder builder) {
    this.numPartitions = builder.numPartitions;
    this.maxIters = builder.maxIters;
    this.sampleRate = builder.sampleRate;
    this.shufflePartitionBatches = builder.shufflePartitionBatches;
    this.shufflePartitionConcurrency = builder.shufflePartitionConcurrency;
    this.useResidual = builder.useResidual;
  }

  public static class Builder {
    private int numPartitions = 32;
    private int maxIters = 50;
    private int sampleRate = 256;
    private int shufflePartitionBatches = 1024 * 10;
    private int shufflePartitionConcurrency = 2;
    private boolean useResidual = true;

    public Builder() {}

    public Builder setNumPartitions(int numPartitions) {
      this.numPartitions = numPartitions;
      return this;
    }

    public Builder setMaxIters(int maxIters) {
      this.maxIters = maxIters;
      return this;
    }

    public Builder setSampleRate(int sampleRate) {
      this.sampleRate = sampleRate;
      return this;
    }

    public Builder setShufflePartitionBatches(int shufflePartitionBatches) {
      this.shufflePartitionBatches = shufflePartitionBatches;
      return this;
    }

    public Builder setShufflePartitionConcurrency(int shufflePartitionConcurrency) {
      this.shufflePartitionConcurrency = shufflePartitionConcurrency;
      return this;
    }

    public Builder setUseResidual(boolean useResidual) {
      this.useResidual = useResidual;
      return this;
    }

    public IvfBuildParams build() {
      return new IvfBuildParams(this);
    }
  }

  // Getter methods
  public int getNumPartitions() {
    return numPartitions;
  }

  public int getMaxIters() {
    return maxIters;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  public int getShufflePartitionBatches() {
    return shufflePartitionBatches;
  }

  public int getShufflePartitionConcurrency() {
    return shufflePartitionConcurrency;
  }

  public boolean useResidual() {
    return useResidual;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
      .append("numPartitions", numPartitions)
      .append("maxIters", maxIters)
      .append("sampleRate", sampleRate)
      .append("shufflePartitionBatches", shufflePartitionBatches)
      .append("shufflePartitionConcurrency", shufflePartitionConcurrency)
      .append("useResidual", useResidual)
      .toString();
  }
}
