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
package org.lance.index;

import java.util.List;
import java.util.Objects;

/**
 * High-level description of an index, aggregating metadata across all segments.
 *
 * <p>This mirrors the Rust {@code IndexDescription} trait and is returned from {@code
 * Dataset.describeIndices}.
 */
public final class IndexDescription {

  private final String name;
  private final List<Integer> fieldIds;
  private final String typeUrl;
  private final String indexType;
  private final long rowsIndexed;
  private final List<Index> metadata;
  private final String detailsJson;

  public IndexDescription(
      String name,
      List<Integer> fieldIds,
      String typeUrl,
      String indexType,
      long rowsIndexed,
      List<Index> metadata,
      String detailsJson) {
    this.name = Objects.requireNonNull(name, "name must not be null");
    this.fieldIds = Objects.requireNonNull(fieldIds, "fieldIds must not be null");
    this.typeUrl = Objects.requireNonNull(typeUrl, "typeUrl must not be null");
    this.indexType = Objects.requireNonNull(indexType, "indexType must not be null");
    this.rowsIndexed = rowsIndexed;
    this.metadata = Objects.requireNonNull(metadata, "metadata must not be null");
    this.detailsJson = detailsJson;
  }

  /** The logical name of the index. */
  public String getName() {
    return name;
  }

  /** Field ids that this index is built on. */
  public List<Integer> getFieldIds() {
    return fieldIds;
  }

  /** Underlying protobuf type URL for the index details. */
  public String getTypeUrl() {
    return typeUrl;
  }

  /** Human-readable index type identifier (e.g. BTREE, INVERTED, IVF_PQ). */
  public String getIndexType() {
    return indexType;
  }

  /** Approximate number of rows covered by this index. */
  public long getRowsIndexed() {
    return rowsIndexed;
  }

  /**
   * Per-segment metadata objects for this index.
   *
   * <p>Each entry corresponds to a single {@link Index} segment in the manifest.
   */
  public List<Index> getMetadata() {
    return metadata;
  }

  /**
   * JSON representation of index-specific details.
   *
   * <p>The exact structure depends on the index implementation.
   */
  public String getDetailsJson() {
    return detailsJson;
  }
}
