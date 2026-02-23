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

import org.lance.Dataset;
import org.lance.JniLoader;

import org.apache.arrow.util.Preconditions;

/**
 * Training utilities for vector indexes.
 *
 * <p>These helpers expose the underlying Lance training routines so that callers can pre-train
 * models (IVF centroids, PQ codebooks, SQ params) and then pass the resulting artifacts into
 * distributed index build flows.
 */
public final class VectorTrainer {

  static {
    JniLoader.ensureLoaded();
  }

  private VectorTrainer() {}

  /**
   * Train IVF centroids for the given dataset column.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params IVF build parameters (numPartitions, sampleRate, etc.)
   * @return a flattened array of centroids laid out as [numPartitions][dimension]
   */
  public static float[] trainIvfCentroids(Dataset dataset, String column, IvfBuildParams params) {
    Preconditions.checkArgument(dataset != null, "dataset cannot be null");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkArgument(params != null, "params cannot be null");
    return nativeTrainIvfCentroids(dataset, column, params);
  }

  /**
   * Train a PQ codebook for the given dataset column.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params PQ build parameters (numSubVectors, numBits, sampleRate, etc.)
   * @return a flattened array of codebook entries laid out as [num_centroids][dimension]
   */
  public static float[] trainPqCodebook(Dataset dataset, String column, PQBuildParams params) {
    Preconditions.checkArgument(dataset != null, "dataset cannot be null");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkArgument(params != null, "params cannot be null");
    return nativeTrainPqCodebook(dataset, column, params);
  }

  private static native float[] nativeTrainIvfCentroids(
      Dataset dataset, String column, IvfBuildParams params);

  private static native float[] nativeTrainPqCodebook(
      Dataset dataset, String column, PQBuildParams params);
}
