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
 * Trained PQ codebook of an {@code IVF_PQ} / {@code IVF_HNSW_PQ} index.
 *
 * <p>Returned as a flat row-major {@code float[]} of size {@code 2^numBits * dimension} together
 * with shape metadata: {@code numBits}, {@code numSubVectors}, and {@code dimension}. PQ codebooks
 * are always loaded as {@code float32} in Lance core, so no element-type field is needed here.
 *
 * <p>The layout matches what {@link PQBuildParams#getCodebook()} (and {@link
 * VectorTrainer#trainPqCodebook}) consume, so the result can be passed straight back into a
 * downstream distributed build.
 */
public final class PqCodebook {
  private final float[] flat;
  private final int numBits;
  private final int numSubVectors;
  private final int dimension;

  public PqCodebook(float[] flat, int numBits, int numSubVectors, int dimension) {
    Preconditions.checkArgument(flat != null, "flat cannot be null");
    Preconditions.checkArgument(numBits > 0 && numBits <= 8, "numBits must be in (0, 8]");
    Preconditions.checkArgument(numSubVectors > 0, "numSubVectors must be positive");
    Preconditions.checkArgument(dimension > 0, "dimension must be positive");
    Preconditions.checkArgument(
        dimension % numSubVectors == 0,
        "dimension %s must be divisible by numSubVectors %s",
        dimension,
        numSubVectors);
    long expected = (1L << numBits) * dimension;
    Preconditions.checkArgument(
        flat.length == expected,
        "flat length %s does not match 2^numBits * dimension = %s",
        flat.length,
        expected);
    this.flat = flat;
    this.numBits = numBits;
    this.numSubVectors = numSubVectors;
    this.dimension = dimension;
  }

  /** Row-major codebook values, length {@code 2^numBits * dimension}. */
  public float[] getFlat() {
    return flat;
  }

  /** Bits per PQ code (typically 8). */
  public int getNumBits() {
    return numBits;
  }

  /** Number of PQ sub-vectors. */
  public int getNumSubVectors() {
    return numSubVectors;
  }

  /** Indexed vector dimension. */
  public int getDimension() {
    return dimension;
  }
}
