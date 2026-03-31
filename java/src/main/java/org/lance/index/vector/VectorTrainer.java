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

import java.util.List;

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
   * <p>Training samples from the entire dataset.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params IVF build parameters (numPartitions, sampleRate, etc.)
   * @return a flattened array of centroids laid out as [numPartitions][dimension]
   */
  public static float[] trainIvfCentroids(Dataset dataset, String column, IvfBuildParams params) {
    return trainIvfCentroids(dataset, column, params, null);
  }

  /**
   * Train IVF centroids for the given dataset column, optionally restricted to specific fragments.
   *
   * <p>When {@code fragmentIds} is non-null, only the listed fragments are sampled for training.
   * This is useful for per-fragment (non-shared centroid) distributed index builds.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params IVF build parameters (numPartitions, sampleRate, etc.)
   * @param fragmentIds fragment IDs to restrict training to, or {@code null} for the full dataset
   * @return a flattened array of centroids laid out as [numPartitions][dimension]
   */
  public static float[] trainIvfCentroids(
      Dataset dataset, String column, IvfBuildParams params, List<Integer> fragmentIds) {
    Preconditions.checkArgument(dataset != null, "dataset cannot be null");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkArgument(params != null, "params cannot be null");
    return nativeTrainIvfCentroids(dataset, column, params, fragmentIds);
  }

  /**
   * Train a PQ codebook for the given dataset column.
   *
   * <p>Training samples from the entire dataset without IVF residual computation.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params PQ build parameters (numSubVectors, numBits, sampleRate, etc.)
   * @return a flattened array of codebook entries laid out as [num_centroids][dimension]
   */
  public static float[] trainPqCodebook(Dataset dataset, String column, PQBuildParams params) {
    return trainPqCodebook(dataset, column, params, null, null);
  }

  /**
   * Train a PQ codebook for the given dataset column, optionally using pre-trained IVF centroids
   * for residual-based training and restricting to specific fragments.
   *
   * <p>When {@code ivfCentroids} is non-null, PQ training is performed on the residual vectors
   * after IVF assignment (matching the Python {@code train_pq} behavior). When {@code fragmentIds}
   * is non-null, only the listed fragments are sampled.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params PQ build parameters (numSubVectors, numBits, sampleRate, etc.)
   * @param ivfCentroids flattened IVF centroids for residual PQ training, or {@code null} to skip
   *     residual computation
   * @param fragmentIds fragment IDs to restrict training to, or {@code null} for the full dataset
   * @return a flattened array of codebook entries laid out as [num_centroids][dimension]
   */
  public static float[] trainPqCodebook(
      Dataset dataset,
      String column,
      PQBuildParams params,
      float[] ivfCentroids,
      List<Integer> fragmentIds) {
    Preconditions.checkArgument(dataset != null, "dataset cannot be null");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkArgument(params != null, "params cannot be null");
    return nativeTrainPqCodebook(dataset, column, params, ivfCentroids, fragmentIds);
  }

  private static native float[] nativeTrainIvfCentroids(
      Dataset dataset, String column, IvfBuildParams params, List<Integer> fragmentIds);

  private static native float[] nativeTrainPqCodebook(
      Dataset dataset,
      String column,
      PQBuildParams params,
      float[] ivfCentroids,
      List<Integer> fragmentIds);
}
