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

/**
 * Train a PQ model for a given column.
 *
 * <p>This will run k-means clustering on each subvector to determine the centroids that will be
 * used to quantize the subvectors. This step runs against a randomly chosen sample of the data. The
 * sample size is typically quite small and PQ training is relatively fast regardless of dataset
 * scale. As a result, accelerators are not needed here.
 */
public class PQBuildParams {
  private final int numSubVectors;
  private final int numBits;
  private final int maxIters;
  private final int kmeansRedos;
  private final int sampleRate;

  private PQBuildParams(Builder builder) {
    this.numSubVectors = builder.numSubVectors;
    this.numBits = builder.numBits;
    this.maxIters = builder.maxIters;
    this.kmeansRedos = builder.kmeansRedos;
    this.sampleRate = builder.sampleRate;
  }

  public static class Builder {
    private int numSubVectors = 16;
    private int numBits = 8;
    private int maxIters = 50;
    private int kmeansRedos = 1;
    private int sampleRate = 256;

    /** Create a new builder for training a PQ model. */
    public Builder() {}

    /**
     * The number of subvectors to divide the source vectors into. This must be a divisor of the
     * vector dimension.
     *
     * @param numSubVectors the number of subvectors
     * @return Builder
     */
    public Builder setNumSubVectors(int numSubVectors) {
      this.numSubVectors = numSubVectors;
      return this;
    }

    /**
     * @param numBits the number of bits to present one PQ centroid
     * @return Builder
     */
    public Builder setNumBits(int numBits) {
      this.numBits = numBits;
      return this;
    }

    /**
     * @param maxIters the max number of iterations for kmeans training.
     * @return Builder
     */
    public Builder setMaxIters(int maxIters) {
      this.maxIters = maxIters;
      return this;
    }

    /**
     * @param kmeansRedos run kmeans `REDOS` times and take the best result. Default to 1
     * @return Builder
     */
    public Builder setKmeansRedos(int kmeansRedos) {
      this.kmeansRedos = kmeansRedos;
      return this;
    }

    /**
     * @param sampleRate sample rate to train PQ codebook
     * @return Builder
     */
    public Builder setSampleRate(int sampleRate) {
      this.sampleRate = sampleRate;
      return this;
    }

    public PQBuildParams build() {
      return new PQBuildParams(this);
    }
  }

  public int getNumSubVectors() {
    return numSubVectors;
  }

  public int getNumBits() {
    return numBits;
  }

  public int getMaxIters() {
    return maxIters;
  }

  public int getKmeansRedos() {
    return kmeansRedos;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("numSubVectors", numSubVectors)
        .append("numBits", numBits)
        .append("maxIters", maxIters)
        .append("kmeansRedos", kmeansRedos)
        .append("sampleRate", sampleRate)
        .toString();
  }
}
