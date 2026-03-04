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
package org.lance.index.scalar;

import org.lance.util.JsonUtils;

import java.util.HashMap;
import java.util.Map;

/** Builder-style configuration for B-Tree scalar index parameters. */
public final class BTreeIndexParams {

  private static final String INDEX_TYPE = "btree";

  private BTreeIndexParams() {}

  /**
   * Create a new builder for B-Tree index parameters.
   *
   * @return a new {@link Builder}
   */
  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private Long zoneSize;
    private Integer rangeId;

    /**
     * Configure the number of rows per zone.
     *
     * @param zoneSize number of rows per zone, must be positive
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder zoneSize(long zoneSize) {
      if (zoneSize <= 0) {
        throw new IllegalArgumentException("zoneSize must be positive");
      }
      this.zoneSize = zoneSize;
      return this;
    }

    /**
     * Configure the ordinal ID of a data partition for building a large, distributed BTree index.
     *
     * @param rangeId non-negative range identifier
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder rangeId(int rangeId) {
      if (rangeId < 0) {
        throw new IllegalArgumentException("rangeId must be non-negative");
      }
      this.rangeId = rangeId;
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for a B-Tree index. */
    public ScalarIndexParams build() {
      Map<String, Object> params = new HashMap<>();
      if (zoneSize != null) {
        params.put("zone_size", zoneSize);
      }
      if (rangeId != null) {
        params.put("range_id", rangeId);
      }

      if (params.isEmpty()) {
        return ScalarIndexParams.create(INDEX_TYPE);
      }

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
