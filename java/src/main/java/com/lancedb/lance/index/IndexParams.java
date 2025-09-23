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

import com.lancedb.lance.index.scalar.ScalarIndexParams;
import com.lancedb.lance.index.vector.VectorIndexParams;

import com.google.common.base.MoreObjects;

import java.util.Optional;

/** Parameters for creating an index. */
public class IndexParams {
  private final Optional<VectorIndexParams> vectorIndexParams;
  private final Optional<ScalarIndexParams> scalarIndexParams;

  private IndexParams(Builder builder) {
    this.vectorIndexParams = builder.vectorIndexParams;
    this.scalarIndexParams = builder.scalarIndexParams;
  }

  /**
   * Create a new builder for IndexParams.
   *
   * @return a new Builder instance
   */
  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private Optional<VectorIndexParams> vectorIndexParams = Optional.empty();
    private Optional<ScalarIndexParams> scalarIndexParams = Optional.empty();

    private Builder() {}

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

    /**
     * Scalar index parameters for creating a scalar index.
     *
     * @param scalarIndexParams scalar index parameters
     * @return this builder
     */
    public Builder setScalarIndexParams(ScalarIndexParams scalarIndexParams) {
      this.scalarIndexParams = Optional.of(scalarIndexParams);
      return this;
    }

    public IndexParams build() {
      return new IndexParams(this);
    }
  }

  public Optional<VectorIndexParams> getVectorIndexParams() {
    return vectorIndexParams;
  }

  public Optional<ScalarIndexParams> getScalarIndexParams() {
    return scalarIndexParams;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("vectorIndexParams", vectorIndexParams.orElse(null))
        .add("scalarIndexParams", scalarIndexParams.orElse(null))
        .toString();
  }
}
