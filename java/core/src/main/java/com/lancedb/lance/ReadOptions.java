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

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/** Read options for reading from a dataset. */
public class ReadOptions {

  private final Optional<Integer> version;
  private final Optional<Integer> blockSize;
  private final long indexCacheSizeBytes;
  private final long metadataCacheSizeBytes;
  private final Map<String, String> storageOptions;

  private ReadOptions(Builder builder) {
    this.version = builder.version;
    this.blockSize = builder.blockSize;
    this.indexCacheSizeBytes = builder.indexCacheSizeBytes;
    this.metadataCacheSizeBytes = builder.metadataCacheSizeBytes;
    this.storageOptions = builder.storageOptions;
  }

  public Optional<Integer> getVersion() {
    return version;
  }

  public Optional<Integer> getBlockSize() {
    return blockSize;
  }

  public long getIndexCacheSizeBytes() {
    return indexCacheSizeBytes;
  }

  public long getMetadataCacheSizeBytes() {
    return metadataCacheSizeBytes;
  }

  public Map<String, String> getStorageOptions() {
    return storageOptions;
  }

  @Override
  public String toString() {
    return new ToStringBuilder(this)
        .append("version", version.orElse(null))
        .append("blockSize", blockSize.orElse(null))
        .append("indexCacheSizeBytes", indexCacheSizeBytes)
        .append("metadataCacheSizeBytes", metadataCacheSizeBytes)
        .append("storageOptions", storageOptions)
        .toString();
  }

  public static class Builder {

    private Optional<Integer> version = Optional.empty();
    private Optional<Integer> blockSize = Optional.empty();
    private long indexCacheSizeBytes = 6 * 1024 * 1024 * 1024; // Default to 6 GiB like Rust
    private long metadataCacheSizeBytes = 1024 * 1024 * 1024; // Default to 1 GiB like Rust
    private Map<String, String> storageOptions = new HashMap<>();

    /**
     * Set the version of the dataset to read. If not set, read from latest version.
     *
     * @param version the version of the dataset
     * @return this builder
     */
    public Builder setVersion(int version) {
      this.version = Optional.of(version);
      return this;
    }

    /**
     * Block size in bytes. Provide a hint for the size of the minimal I/O request. Recommended to
     * be set to value bigger than 4KB.
     *
     * @param blockSize the block size in bytes
     * @return this builder
     */
    public Builder setBlockSize(int blockSize) {
      this.blockSize = Optional.of(blockSize);
      return this;
    }

    /**
     * Index cache size in bytes. Index cache is a LRU cache with TTL. This number specifies the
     * size of the index cache in bytes. Default is 6 GiB.
     *
     * @param indexCacheSizeBytes the index cache size in bytes
     * @return this builder
     */
    public Builder setIndexCacheSizeBytes(long indexCacheSizeBytes) {
      this.indexCacheSizeBytes = indexCacheSizeBytes;
      return this;
    }

    /**
     * Index cache size. Index cache is a LRU cache with TTL. This number specifies the number of
     * index pages, for example, IVF partitions, to be cached in the host memory. Roughly, for an
     * IVF_PQ partition with n rows, the size of each index page equals the combination of the pq
     * code (nd.array([n,pq], dtype=uint8)) and the row ids (nd.array([n], dtype=uint64)).
     * Approximately, n = Total Rows / number of IVF partitions. pq = number of PQ sub-vectors.
     * Default is 256.
     *
     * @param indexCacheSize the index cache size
     * @return this builder
     */
    @Deprecated
    public Builder setIndexCacheSize(int indexCacheSize) {
      long assumedEntrySize = 20 * 1024 * 1024; // 20MB per entry
      this.indexCacheSizeBytes = indexCacheSize * assumedEntrySize;
      return this;
    }

    /**
     * Size of the metadata cache in bytes. This cache stores metadata in memory for faster open
     * table and scans. If it is zero, metadata cache is disabled. Default is 1 GiB.
     *
     * @param metadataCacheSizeBytes the metadata cache size in bytes
     * @return this builder
     */
    public Builder setMetadataCacheSizeBytes(long metadataCacheSizeBytes) {
      this.metadataCacheSizeBytes = metadataCacheSizeBytes;
      return this;
    }

    /**
     * Metadata cache size for the fragment metadata. If it is zero, metadata cache is disabled.
     * This method is deprecated. Use {@link #setMetadataCacheSizeBytes(long)} instead.
     *
     * @param metadataCacheSize the metadata cache size (entry count)
     * @return this builder
     * @deprecated Use {@link #setMetadataCacheSizeBytes(long)} instead
     */
    @Deprecated
    public Builder setMetadataCacheSize(int metadataCacheSize) {
      long assumedEntrySize = 4 * 1024 * 1024; // 4MB per entry
      this.metadataCacheSizeBytes = metadataCacheSize * assumedEntrySize;
      return this;
    }

    /**
     * Set storage options. Extra options that make sense for a particular storage connection. This
     * is used to store connection parameters like credentials, endpoint, etc.
     *
     * @param storageOptions the storage options
     * @return this builder
     */
    public Builder setStorageOptions(Map<String, String> storageOptions) {
      this.storageOptions = storageOptions;
      return this;
    }

    public ReadOptions build() {
      return new ReadOptions(this);
    }
  }
}
