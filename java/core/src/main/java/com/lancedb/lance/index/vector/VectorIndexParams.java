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

package com.lancedb.lance.index.vector;

import java.util.Optional;

import com.lancedb.lance.index.DistanceType;

public class VectorIndexParams {
  private final DistanceType distanceType;
  private final IvfBuildParams ivfParams;
  private final Optional<PQBuildParams> pqParams;
  private final Optional<HnswBuildParams> hnswParams;
  private final Optional<SQBuildParams> sqParams;

  private VectorIndexParams(Builder builder) {
    this.distanceType = builder.distanceType;
    this.ivfParams = builder.ivfParams;
    this.pqParams = builder.pqParams;
    this.hnswParams = builder.hnswParams;
    this.sqParams = builder.sqParams;
    validate();
  }

  private void validate() {
    if (pqParams.isPresent() && sqParams.isPresent()) {
      throw new IllegalArgumentException("PQ and SQ cannot coexist");
    }
    if (hnswParams.isPresent() && !pqParams.isPresent() && !sqParams.isPresent()) {
      throw new IllegalArgumentException("HNSW must be combined with either PQ or SQ");
    }
    if (sqParams.isPresent() && !hnswParams.isPresent()) {
      throw new IllegalArgumentException("IVF + SQ is not supported");
    }
  }

  public static VectorIndexParams ivfFlat(int numPartitions, DistanceType distanceType) {
    return new Builder(new IvfBuildParams.Builder().setNumPartitions(numPartitions).build())
        .setDistanceType(distanceType)
        .build();
  }

  public static VectorIndexParams ivfPq(int numPartitions, int numBits, int numSubVectors,
      DistanceType distanceType, int maxIterations) {
    IvfBuildParams ivfParams = new IvfBuildParams.Builder().setNumPartitions(numPartitions).build();
    PQBuildParams pqParams = new PQBuildParams.Builder()
        .setNumBits(numBits)
        .setNumSubVectors(numSubVectors)
        .setMaxIters(maxIterations)
        .build();

    return new Builder(ivfParams)
        .setDistanceType(distanceType)
        .setPqParams(pqParams)
        .build();
  }

  public static VectorIndexParams withIvfPqParams(DistanceType distanceType,
      IvfBuildParams ivf,
      PQBuildParams pq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setPqParams(pq)
        .build();
  }

  public static VectorIndexParams withIvfHnswPqParams(DistanceType distanceType,
      IvfBuildParams ivf,
      HnswBuildParams hnsw,
      PQBuildParams pq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setHnswParams(hnsw)
        .setPqParams(pq)
        .build();
  }

  public static VectorIndexParams withIvfHnswSqParams(DistanceType distanceType,
      IvfBuildParams ivf,
      HnswBuildParams hnsw,
      SQBuildParams sq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setHnswParams(hnsw)
        .setSqParams(sq)
        .build();
  }

  public static class Builder {
    private DistanceType distanceType = DistanceType.L2; // Default to L2
    private final IvfBuildParams ivfParams;
    private Optional<PQBuildParams> pqParams = Optional.empty();
    private Optional<HnswBuildParams> hnswParams = Optional.empty();
    private Optional<SQBuildParams> sqParams = Optional.empty();

    public Builder(IvfBuildParams ivfParams) {
      this.ivfParams = ivfParams;
    }

    public Builder setDistanceType(DistanceType distanceType) {
      this.distanceType = distanceType;
      return this;
    }

    public Builder setPqParams(PQBuildParams pqParams) {
      this.pqParams = Optional.of(pqParams);
      return this;
    }

    public Builder setHnswParams(HnswBuildParams hnswParams) {
      this.hnswParams = Optional.of(hnswParams);
      return this;
    }

    public Builder setSqParams(SQBuildParams sqParams) {
      this.sqParams = Optional.of(sqParams);
      return this;
    }

    public VectorIndexParams build() {
      return new VectorIndexParams(this);
    }
  }

  public DistanceType getDistanceType() {
    return distanceType;
  }

  public IvfBuildParams getIvfParams() {
    return ivfParams;
  }

  public Optional<PQBuildParams> getPqParams() {
    return pqParams;
  }

  public Optional<HnswBuildParams> getHnswParams() {
    return hnswParams;
  }

  public Optional<SQBuildParams> getSqParams() {
    return sqParams;
  }
}
