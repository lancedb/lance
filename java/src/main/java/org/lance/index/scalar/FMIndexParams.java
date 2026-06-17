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

/**
 * Builder-style configuration for FM-Index scalar index parameters.
 *
 * <p>An FM-Index supports exact substring search (the {@code contains} function) over a string or
 * binary column and returns exact row ids.
 */
public final class FMIndexParams {

  private static final String INDEX_TYPE = "fm";

  private FMIndexParams() {}

  /**
   * Create a new builder for FM-Index parameters.
   *
   * @return a new {@link Builder}
   */
  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private Integer numSegments;

    /**
     * Configure the number of independent index segments to distribute the dataset fragments
     * across. Each segment is a complete FM-Index covering a disjoint set of fragments, enabling
     * incremental indexing and segment merge. Defaults to a single segment when not set.
     *
     * @param numSegments number of segments, must be positive
     * @return this builder
     * @throws IllegalArgumentException if {@code numSegments} is not positive
     */
    public Builder numSegments(int numSegments) {
      if (numSegments <= 0) {
        throw new IllegalArgumentException("numSegments must be positive");
      }
      this.numSegments = numSegments;
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for an FM-Index. */
    public ScalarIndexParams build() {
      Map<String, Object> params = new HashMap<>();
      if (numSegments != null) {
        params.put("num_segments", numSegments);
      }

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
