package com.lancedb.lance.index.vector;

import java.util.Optional;

public class HnswBuildParams {
  private final short maxLevel;
  private final int m;
  private final int efConstruction;
  private final Optional<Integer> prefetchDistance;

  private HnswBuildParams(Builder builder) {
    this.maxLevel = builder.maxLevel;
    this.m = builder.m;
    this.efConstruction = builder.efConstruction;
    this.prefetchDistance = builder.prefetchDistance;
  }

  public static class Builder {
    private short maxLevel = 7;
    private int m = 20;
    private int efConstruction = 150;
    private Optional<Integer> prefetchDistance = Optional.of(2);

    public Builder() {
    }

    public Builder maxLevel(short maxLevel) {
      this.maxLevel = maxLevel;
      return this;
    }

    public Builder m(int m) {
      this.m = m;
      return this;
    }

    public Builder setEfConstruction(int efConstruction) {
      this.efConstruction = efConstruction;
      return this;
    }

    public Builder setPrefetchDistance(Integer prefetchDistance) {
      this.prefetchDistance = Optional.ofNullable(prefetchDistance);
      return this;
    }

    public HnswBuildParams build() {
      return new HnswBuildParams(this);
    }
  }

  // Getter methods
  public short getMaxLevel() {
    return maxLevel;
  }

  public int getM() {
    return m;
  }

  public int getEfConstruction() {
    return efConstruction;
  }

  public Optional<Integer> getPrefetchDistance() {
    return prefetchDistance;
  }
}