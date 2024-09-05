/*
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.lancedb.lance.index;

import org.apache.commons.lang3.builder.ToStringBuilder;
import java.util.Optional;

/**
 * Parameters for building a Scalar Index. This determines how the index is constructed and what
 * information it stores.
 */
public class ScalarIndexParams {
  private final Optional<ScalarIndexType> forceIndexType;

  private ScalarIndexParams(Builder builder) {
    this.forceIndexType = builder.forceIndexType;
  }

  public static class Builder {
    private Optional<ScalarIndexType> forceIndexType = Optional.empty();

    /**
     * Create a new builder for Scalar Index parameters.
     */
    public Builder() {}

    /**
     * @param forceIndexType if set, always use the given index type and skip auto-detection.
     * @return Builder
     */
    public Builder setForceIndexType(ScalarIndexType forceIndexType) {
      this.forceIndexType = Optional.ofNullable(forceIndexType);
      return this;
    }

    public ScalarIndexParams build() {
      return new ScalarIndexParams(this);
    }
  }

  public Optional<ScalarIndexType> getForceIndexType() {
    return forceIndexType;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this).append("forceIndexType", forceIndexType.orElse(null))
        .toString();
  }

  public enum ScalarIndexType {
    BTREE, BITMAP, LABEL_LIST, INVERTED
  }
}
