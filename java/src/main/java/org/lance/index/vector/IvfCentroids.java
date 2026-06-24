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

import org.apache.arrow.util.Preconditions;

/**
 * Trained IVF centroids of a committed vector index.
 *
 * <p>Returned as a flat row-major {@code float[]} of size {@code numPartitions * dimension}
 * together with shape metadata. Centroids with a non-{@code float32} element type are converted to
 * {@code float32} before being returned; the original element type is exposed via {@link
 * #getElementType()} for diagnostics.
 *
 * <p>The layout matches {@link VectorTrainer#trainIvfCentroids} and avoids cross-version Arrow-Java
 * {@code LinkageError} that would arise from returning Arrow vectors across the JNI boundary.
 */
public final class IvfCentroids {
  private final float[] flat;
  private final int numPartitions;
  private final int dimension;
  private final String elementType;

  public IvfCentroids(float[] flat, int numPartitions, int dimension, String elementType) {
    Preconditions.checkArgument(flat != null, "flat cannot be null");
    Preconditions.checkArgument(numPartitions > 0, "numPartitions must be positive");
    Preconditions.checkArgument(dimension > 0, "dimension must be positive");
    Preconditions.checkArgument(elementType != null, "elementType cannot be null");
    Preconditions.checkArgument(
        flat.length == (long) numPartitions * dimension,
        "flat length %s does not match numPartitions * dimension = %s",
        flat.length,
        (long) numPartitions * dimension);
    this.flat = flat;
    this.numPartitions = numPartitions;
    this.dimension = dimension;
    this.elementType = elementType;
  }

  /** Row-major centroid values, length {@code numPartitions * dimension}. */
  public float[] getFlat() {
    return flat;
  }

  /** Number of IVF partitions. */
  public int getNumPartitions() {
    return numPartitions;
  }

  /** Indexed vector dimension. */
  public int getDimension() {
    return dimension;
  }

  /**
   * Original element type of the underlying Arrow array (e.g. {@code FLOAT32}, {@code FLOAT16},
   * {@code FLOAT64}, {@code UINT8}).
   */
  public String getElementType() {
    return elementType;
  }
}
