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
package com.lancedb.lance.index.scalar;

import com.google.common.base.MoreObjects;

import java.util.Optional;

/** Parameters for creating scalar indices. */
public class ScalarIndexParams {
  private final String indexType;
  private final Optional<String> jsonParams;

  private ScalarIndexParams(Builder builder) {
    this.indexType = builder.indexType;
    this.jsonParams = builder.jsonParams;
  }

  /**
   * Create a new ScalarIndexParams with the given index type and no parameters.
   *
   * @param indexType the index type (e.g., "btree", "zonemap", "bitmap", "inverted", "labellist",
   *     "ngram")
   * @return ScalarIndexParams
   */
  public static ScalarIndexParams create(String indexType) {
    return new Builder(indexType).build();
  }

  /**
   * Create a new ScalarIndexParams with the given index type and JSON parameters.
   *
   * @param indexType the index type (e.g., "btree", "zonemap", "bitmap", "inverted", "labellist",
   *     "ngram")
   * @param jsonParams JSON string containing index-specific parameters
   * @return ScalarIndexParams
   */
  public static ScalarIndexParams create(String indexType, String jsonParams) {
    return new Builder(indexType).setJsonParams(jsonParams).build();
  }

  public static class Builder {
    private final String indexType;
    private Optional<String> jsonParams = Optional.empty();

    /**
     * Create a new builder for scalar index parameters.
     *
     * @param indexType the index type (e.g., "btree", "zonemap", "bitmap", "inverted", "labellist",
     *     "ngram")
     */
    public Builder(String indexType) {
      this.indexType = indexType;
    }

    /**
     * Set the parameters for the index as a JSON string.
     *
     * @param jsonParams JSON string containing index-specific parameters
     * @return this builder
     */
    public Builder setJsonParams(String jsonParams) {
      this.jsonParams = Optional.of(jsonParams);
      return this;
    }

    public ScalarIndexParams build() {
      return new ScalarIndexParams(this);
    }
  }

  public String getIndexType() {
    return indexType;
  }

  public Optional<String> getJsonParams() {
    return jsonParams;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("indexType", indexType)
        .add("jsonParams", jsonParams.orElse(null))
        .toString();
  }
}
