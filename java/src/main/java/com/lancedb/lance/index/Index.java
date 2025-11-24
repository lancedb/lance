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

import com.google.common.base.MoreObjects;

import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * Metadata for an index in the dataset. This class corresponds to the Rust Index struct in
 * lance/rust/lance-table/src/format/index.rs.
 */
public class Index {
  private final UUID uuid;
  private final List<Integer> fields;
  private final String name;
  private final long datasetVersion;
  private final List<Integer> fragments;
  private final byte[] indexDetails;
  private final int indexVersion;
  private final Instant createdAt;
  private final Integer baseId;

  private Index(
      UUID uuid,
      List<Integer> fields,
      String name,
      long datasetVersion,
      List<Integer> fragments,
      byte[] indexDetails,
      int indexVersion,
      Instant createdAt,
      Integer baseId) {
    this.uuid = uuid;
    this.fields = fields;
    this.name = name;
    this.datasetVersion = datasetVersion;
    this.fragments = fragments;
    this.indexDetails = indexDetails;
    this.indexVersion = indexVersion;
    this.createdAt = createdAt;
    this.baseId = baseId;
  }

  public UUID uuid() {
    return uuid;
  }

  /**
   * Get the field id list included in the index.
   *
   * @return the field IDs
   */
  public List<Integer> fields() {
    return fields;
  }

  /**
   * Human readable index name
   *
   * @return the index name
   */
  public String name() {
    return name;
  }

  /**
   * The latest version of the dataset this index covers
   *
   * @return the dataset version
   */
  public long datasetVersion() {
    return datasetVersion;
  }

  public Optional<List<Integer>> fragments() {
    return Optional.ofNullable(fragments);
  }

  public Optional<byte[]> indexDetails() {
    return Optional.ofNullable(indexDetails);
  }

  public Optional<Integer> baseId() {
    return Optional.ofNullable(baseId);
  }

  /**
   * Get the index version.
   *
   * @return the index version
   */
  public int indexVersion() {
    return indexVersion;
  }

  /**
   * Get the creation time of the index.
   *
   * @return the creation time
   */
  public Optional<Instant> createdAt() {
    return Optional.ofNullable(createdAt);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Index index = (Index) o;
    return datasetVersion == index.datasetVersion
        && indexVersion == index.indexVersion
        && Objects.equals(uuid, index.uuid)
        && Objects.equals(fields, index.fields)
        && Objects.equals(name, index.name)
        && Objects.equals(fragments, index.fragments)
        && Arrays.equals(indexDetails, index.indexDetails)
        && Objects.equals(createdAt, index.createdAt)
        && Objects.equals(baseId, index.baseId);
  }

  @Override
  public int hashCode() {
    int result =
        Objects.hash(
            uuid, fields, name, datasetVersion, indexVersion, createdAt, baseId, fragments);
    result = 31 * result + Arrays.hashCode(indexDetails);
    return result;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("uuid", uuid)
        .add("fields", fields)
        .add("name", name)
        .add("datasetVersion", datasetVersion)
        .add("indexVersion", indexVersion)
        .add("createdAt", createdAt)
        .add("baseId", baseId)
        .toString();
  }

  /**
   * Create a new builder for Index.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {

    private UUID uuid;
    private List<Integer> fields;
    private String name;
    private long datasetVersion;
    private List<Integer> fragments;
    private byte[] indexDetails;
    private int indexVersion;
    private Instant createdAt;
    private Integer baseId;

    private Builder() {}

    public Builder uuid(UUID uuid) {
      this.uuid = uuid;
      return this;
    }

    public Builder fields(List<Integer> fields) {
      this.fields = fields;
      return this;
    }

    public Builder name(String name) {
      this.name = name;
      return this;
    }

    public Builder datasetVersion(long datasetVersion) {
      this.datasetVersion = datasetVersion;
      return this;
    }

    public Builder fragments(List<Integer> fragments) {
      this.fragments = fragments;
      return this;
    }

    public Builder indexDetails(byte[] indexDetails) {
      this.indexDetails = indexDetails;
      return this;
    }

    public Builder indexVersion(int indexVersion) {
      this.indexVersion = indexVersion;
      return this;
    }

    public Builder createdAt(Instant createdAt) {
      this.createdAt = createdAt;
      return this;
    }

    public Builder baseId(Integer baseId) {
      this.baseId = baseId;
      return this;
    }

    public Index build() {
      return new Index(
          uuid,
          fields,
          name,
          datasetVersion,
          fragments,
          indexDetails,
          indexVersion,
          createdAt,
          baseId);
    }
  }
}
