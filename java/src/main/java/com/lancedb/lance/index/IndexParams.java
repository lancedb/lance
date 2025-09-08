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
package com.lancedb.lance.index;

import com.lancedb.lance.index.vector.VectorIndexParams;

import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.Optional;

/** Parameters for creating an index. */
public class IndexParams {
  private final DistanceType distanceType;
  private final Optional<VectorIndexParams> vectorIndexParams;

  private IndexParams(Builder builder) {
    this.distanceType = builder.distanceType;
    this.vectorIndexParams = builder.vectorIndexParams;
  }

  public static class Builder {
    private DistanceType distanceType = DistanceType.L2;
    private Optional<VectorIndexParams> vectorIndexParams = Optional.empty();

    public Builder() {}

    /**
     * Set the distance type for calculating the distance between vectors. Default to L2.
     *
     * @param distanceType distance type
     * @return this builder
     */
    public Builder setDistanceType(DistanceType distanceType) {
      this.distanceType = distanceType;
      return this;
    }

    /**
     * Vector index parameters for creating a vector index.
     *
     * @param vectorIndexParams vector index parameters
     * @return this builder
     */
    public Builder setVectorIndexParams(VectorIndexParams vectorIndexParams) {
      this.vectorIndexParams = Optional.of(vectorIndexParams);
      return this;
    }

    public IndexParams build() {
      return new IndexParams(this);
    }
  }

  public String getDistanceType() {
    return distanceType.toString();
  }

  public Optional<VectorIndexParams> getVectorIndexParams() {
    return vectorIndexParams;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("distanceType", distanceType)
        .append("vectorIndexParams", vectorIndexParams.orElse(null))
        .toString();
  }
}
