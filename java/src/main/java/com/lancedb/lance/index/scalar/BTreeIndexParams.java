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

/** Parameters for creating B-Tree scalar indices. */
public class BTreeIndexParams {
  private final long batchSize;

  private BTreeIndexParams(Builder builder) {
    this.batchSize = builder.batchSize;
  }

  public static class Builder {
    private long batchSize = 4096L; // DEFAULT_BTREE_BATCH_SIZE from Rust

    public Builder() {}

    /**
     * Set the batch size for B-Tree index pages.
     *
     * @param batchSize batch size
     * @return this builder
     */
    public Builder setBatchSize(long batchSize) {
      this.batchSize = batchSize;
      return this;
    }

    public BTreeIndexParams build() {
      return new BTreeIndexParams(this);
    }
  }

  public long getBatchSize() {
    return batchSize;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("batchSize", batchSize).toString();
  }
}
