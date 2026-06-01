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
package org.lance.memwal;

import com.google.common.base.Preconditions;

/**
 * HNSW build parameters for a maintained vector index, used with {@link
 * ShardWriterConfig#withHnswParams}.
 *
 * <p>Defaults match the Lance defaults; override only the fields you need. {@code m} is the graph
 * degree (level 0 retains {@code 2*m}), equivalent to FAISS's {@code M}.
 */
public class MemWalHnswParams {
  private final String indexName;
  private int m = 20;
  private int efConstruction = 150;
  private int maxLevel = 7;

  /** @param indexName name of the maintained vector index these parameters apply to. */
  public MemWalHnswParams(String indexName) {
    Preconditions.checkNotNull(indexName, "indexName must not be null");
    this.indexName = indexName;
  }

  /** HNSW graph degree: max neighbors retained per node on upper levels (level 0 retains 2*m). */
  public MemWalHnswParams withM(int m) {
    Preconditions.checkArgument(m > 0, "m must be greater than 0");
    this.m = m;
    return this;
  }

  /** Beam width (candidate list size) used while inserting nodes at build time. */
  public MemWalHnswParams withEfConstruction(int efConstruction) {
    Preconditions.checkArgument(efConstruction > 0, "efConstruction must be greater than 0");
    this.efConstruction = efConstruction;
    return this;
  }

  /** Maximum number of graph levels. */
  public MemWalHnswParams withMaxLevel(int maxLevel) {
    Preconditions.checkArgument(maxLevel > 0, "maxLevel must be greater than 0");
    this.maxLevel = maxLevel;
    return this;
  }

  public String indexName() {
    return indexName;
  }

  public int m() {
    return m;
  }

  public int efConstruction() {
    return efConstruction;
  }

  public int maxLevel() {
    return maxLevel;
  }
}
