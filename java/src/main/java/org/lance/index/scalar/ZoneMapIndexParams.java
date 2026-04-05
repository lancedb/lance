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

/** Builder-style configuration for ZoneMap scalar index parameters. */
public final class ZoneMapIndexParams {

  private static final String INDEX_TYPE = "zonemap";

  private ZoneMapIndexParams() {}

  /**
   * Create a new builder for ZoneMap index parameters.
   *
   * @return a new {@link Builder}
   */
  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private Long rowsPerZone;

    /**
     * Configure the approximate number of rows per zone.
     *
     * @param rowsPerZone number of rows per zone, must be positive
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder rowsPerZone(long rowsPerZone) {
      if (rowsPerZone <= 0) {
        throw new IllegalArgumentException("rowsPerZone must be positive");
      }
      this.rowsPerZone = rowsPerZone;
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for a ZoneMap index. */
    public ScalarIndexParams build() {
      Map<String, Object> params = new HashMap<>();
      if (rowsPerZone != null) {
        params.put("rows_per_zone", rowsPerZone);
      }

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
