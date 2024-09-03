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

    public Builder() {
    }

    public Builder setNumSubVectors(int numSubVectors) {
      this.numSubVectors = numSubVectors;
      return this;
    }

    public Builder setNumBits(int numBits) {
      this.numBits = numBits;
      return this;
    }

    public Builder setMaxIters(int maxIters) {
      this.maxIters = maxIters;
      return this;
    }

    public Builder setKmeansRedos(int kmeansRedos) {
      this.kmeansRedos = kmeansRedos;
      return this;
    }

    public Builder setSampleRate(int sampleRate) {
      this.sampleRate = sampleRate;
      return this;
    }

    public PQBuildParams build() {
      return new PQBuildParams(this);
    }
  }

  // Getter methods
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