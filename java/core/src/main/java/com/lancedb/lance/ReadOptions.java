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

package com.lancedb.lance;

import org.apache.commons.lang3.builder.ToStringBuilder;
import java.util.Optional;

/**
 * Read options for reading from a dataset.
 */
public class ReadOptions {

  private Optional<Integer> version;
  private Optional<Integer> blockSize;
  private int indexCacheSize;
  private int metadataCacheSize;

  private ReadOptions(Builder builder) {
    this.version = Optional.ofNullable(builder.version);
    this.blockSize = Optional.ofNullable(builder.blockSize);
    this.indexCacheSize = builder.indexCacheSize;
    this.metadataCacheSize = builder.metadataCacheSize;
  }

  public Optional<Integer> getVersion() {
    return version;
  }

  public Optional<Integer> getBlockSize() {
    return blockSize;
  }

  public int getIndexCacheSize() {
    return indexCacheSize;
  }

  public int getMetadataCacheSize() {
    return metadataCacheSize;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("version", version.orElse(null))
        .append("blockSize", blockSize.orElse(null))
        .append("indexCacheSize", indexCacheSize)
        .append("metadataCacheSize", metadataCacheSize)
        .toString();
  }

  public static class Builder {

    private Integer version;
    private Integer blockSize;
    private int indexCacheSize = 256;
    private int metadataCacheSize = 256;

    public Builder setVersion(int version) {
      this.version = version;
      return this;
    }

    public Builder setBlockSize(int blockSize) {
      this.blockSize = blockSize;
      return this;
    }

    public Builder setIndexCacheSize(int indexCacheSize) {
      this.indexCacheSize = indexCacheSize;
      return this;
    }

    public Builder setMetadataCacheSize(int metadataCacheSize) {
      this.metadataCacheSize = metadataCacheSize;
      return this;
    }

    public ReadOptions build() {
      return new ReadOptions(this);
    }
  }
}
