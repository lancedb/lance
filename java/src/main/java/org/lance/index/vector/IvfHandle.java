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
package org.lance.index.vector;

import org.lance.index.DistanceType;

import java.util.Optional;

/**
 * IVF view of a {@link VectorIndexHandle}. Does not own native resources independently; relies on
 * the parent handle staying open until reads complete.
 */
public final class IvfHandle {
  private final VectorIndexHandle parent;

  IvfHandle(VectorIndexHandle parent) {
    this.parent = parent;
  }

  /** Distance metric used to build this vector index. */
  public DistanceType getDistanceType() {
    return parent.getDistanceType();
  }

  /** Indexed vector dimension. */
  public int getDimension() {
    return parent.getDimension();
  }

  /**
   * Read the trained IVF centroids of this index (concatenated across segments in commit order).
   *
   * @throws IllegalStateException if the parent handle has been closed
   */
  public IvfCentroids readCentroids() {
    return nativeReadCentroids(parent.nativeHandlePtr());
  }

  /**
   * Read the trained PQ codebook of this index, or {@link Optional#empty()} if the index does not
   * use product quantization.
   *
   * @throws IllegalStateException if the parent handle has been closed
   */
  public Optional<PqCodebook> readPqCodebook() {
    return Optional.ofNullable(nativeReadPqCodebook(parent.nativeHandlePtr()));
  }

  private static native IvfCentroids nativeReadCentroids(long handle);

  private static native PqCodebook nativeReadPqCodebook(long handle);
}
