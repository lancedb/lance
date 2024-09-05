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

/**
 * Parameters for building an Inverted Index. This determines how the index is constructed and what
 * information it stores.
 */
public class InvertedIndexParams {
  private final boolean withPosition;

  private InvertedIndexParams(Builder builder) {
    this.withPosition = builder.withPosition;
  }

  public static class Builder {
    private boolean withPosition = true;

    /**
     * Create a new builder for Inverted Index parameters.
     */
    public Builder() {}

    /**
     * @param withPosition if true, store the position of the term in the document. This can
     *        significantly increase the size of the index. If false, only store the frequency of
     *        the term in the document.
     * @return Builder
     */
    public Builder setWithPosition(boolean withPosition) {
      this.withPosition = withPosition;
      return this;
    }

    public InvertedIndexParams build() {
      return new InvertedIndexParams(this);
    }
  }

  public boolean isWithPosition() {
    return withPosition;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this).append("withPosition", withPosition).toString();
  }
}
