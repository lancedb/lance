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

package com.lancedb.lance.index;

import com.lancedb.lance.Dataset;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

/** Builder for creating Index on lance dataset. */
public class IndexBuilder {
  private final Dataset dataset;
  private final List<String> columns;

  private Optional<IndexParams> params;

  private DistanceType distanceType = DistanceType.L2;

  /** Set true to replace existing index. */
  private boolean replace = false;

  /** Constructor. */
  public IndexBuilder(Dataset dataset, List<String> columns) {
    this.dataset = dataset;
    this.columns = columns;
  }

  /** Set true to replace existing index. */
  public IndexBuilder replace() {
    return this.replace(true);
  }

  /** Set true to replace existing index. */
  public IndexBuilder replace(boolean flag) {
    this.replace = flag;
    return this;
  }

  /**
   * Build Ivf_PQ index.
   *
   * @param numPartitions Number of IVF Partitions.
   * @param numSubVectors Number of PQ sub-vectors.
   */
  public IndexBuilder ivfPq(int numPartitions, int numSubVectors) throws IOException {
    if (this.params.isPresent()) {
      throw new IOException("A different index parameter already set.");
    }
    this.params = Optional.of(new IvfPqParams(numPartitions, numSubVectors));
    return this;
  }

  /**
   * Set the distance type.
   *
   * <p>Default type is {@link DistanceType.L2}
   */
  public IndexBuilder distanceType(DistanceType dt) {
    this.distanceType = dt;
    return this;
  }

  /** Build a Scalar index. */
  public IndexBuilder scalar() throws IOException {
    if (this.params.isPresent()) {
      throw new IOException("A different index parameter already set.");
    }
    this.params = Optional.of(new ScalarParams());
    return this;
  }

  /** Build the index. */
  public void build() throws IOException {
    if (this.params.isEmpty()) {
      throw new IOException("Index parameters are not set");
    }
    var params = this.params.get();
    if (params instanceof IvfPqParams) {
      if (columns.size() != 1) {
        throw new IOException("Can only create IVF_PQ on one column, got: " + columns);
      }
      createIvfPq(
          dataset,
          columns.get(0),
          ((IvfPqParams) params).getNumPartitions(),
          ((IvfPqParams) params).getNumSubVectors(),
          distanceType.getValue(),
          this.replace);

    } else {
      throw new IOException("Unsupported Index Parameter");
    }
  }

  /**
   * Explicitly call create ivf pq with primitive parameters for simplicity.
   *
   * @param dataset the dataset instance.
   * @param column the column name.
   * @param numPartitions the number of IVF Partition.
   * @param numSubVectors the number of PQ sub vectors.
   * @param replace whether to replace existing index.
   */
  private native void createIvfPq(
      Dataset dataset,
      String column,
      int numPartitions,
      int numSubVectors,
      int distanceType,
      boolean replace)
      throws IOException;
}
