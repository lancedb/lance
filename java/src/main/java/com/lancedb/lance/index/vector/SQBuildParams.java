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

/** Parameters for using SQ quantizer. */
public class SQBuildParams {
  private final short numBits;
  private final int sampleRate;

  private SQBuildParams(Builder builder) {
    this.numBits = builder.numBits;
    this.sampleRate = builder.sampleRate;
  }

  public static class Builder {
    private short numBits = 8;
    private int sampleRate = 256;

    public Builder() {}

    /**
     * @param numBits number of bits of scaling range.
     * @return Builder
     */
    public Builder setNumBits(short numBits) {
      this.numBits = numBits;
      return this;
    }

    /**
     * @param sampleRate sample rate for training
     * @return Builder
     */
    public Builder setSampleRate(int sampleRate) {
      this.sampleRate = sampleRate;
      return this;
    }

    public SQBuildParams build() {
      return new SQBuildParams(this);
    }
  }

  // Getter methods
  public short getNumBits() {
    return numBits;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("numBits", numBits)
        .append("sampleRate", sampleRate)
        .toString();
  }
}
