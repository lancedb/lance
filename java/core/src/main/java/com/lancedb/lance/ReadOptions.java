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

import java.util.Optional;

/**
 * Read options for reading from a dataset.
 */
public class ReadOptions {
  private Optional<Integer> version;
  private Optional<Integer> blockSize;
  private int indexCacheSize;
  private int metadataCacheSize;

  public ReadOptions() {
    this.version = Optional.empty();
    this.blockSize = Optional.empty();
    this.indexCacheSize = 256;
    this.metadataCacheSize = 256;
  }

  public ReadOptions version(int version) {
    this.version = Optional.of(version);
    return this;
  }

  public ReadOptions blockSize(int blockSize) {
    this.blockSize = Optional.of(blockSize);
    return this;
  }

  public ReadOptions indexCacheSize(int indexCacheSize) {
    this.indexCacheSize = indexCacheSize;
    return this;
  }

  public ReadOptions metadataCacheSize(int metadataCacheSize) {
    this.metadataCacheSize = metadataCacheSize;
    return this;
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
}