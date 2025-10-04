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
package com.lancedb.lance.blob;

import org.apache.arrow.vector.types.pojo.Field;
import java.util.Map;
import java.util.HashMap;

/**
 * Default implementation of the BlobColumn interface.
 *
 * This class provides a concrete implementation of BlobColumn that can be
 * constructed from Lance schema information and field metadata.
 */
public class BlobColumn {

  private final String name;
  private final Field field;
  private final int fieldId;
  private final boolean externalStorage;
  private final String compressionType;
  private final long maxBlobSize;
  private final String storageClass;
  private final boolean nullable;
  private final Map<String, String> metadata;
  private final BlobColumnStatistics statistics;

  /**
   * Constructor for BlobColumnImpl.
   *
   * @param name the column name
   * @param field the Arrow field definition
   * @param fieldId the field ID in Lance schema
   * @param externalStorage whether external storage is used
   * @param compressionType the compression type
   * @param maxBlobSize the maximum blob size (-1 for no limit)
   * @param storageClass the storage class
   * @param nullable whether the column is nullable
   * @param metadata the column metadata
   * @param statistics the column statistics (can be null)
   */
  public BlobColumn(String name, Field field, int fieldId, boolean externalStorage,
      String compressionType, long maxBlobSize, String storageClass,
      boolean nullable, Map<String, String> metadata,
      BlobColumnStatistics statistics) {
    this.name = name;
    this.field = field;
    this.fieldId = fieldId;
    this.externalStorage = externalStorage;
    this.compressionType = compressionType;
    this.maxBlobSize = maxBlobSize;
    this.storageClass = storageClass;
    this.nullable = nullable;
    this.metadata = metadata != null ? new HashMap<>(metadata) : new HashMap<>();
    this.statistics = statistics;
  }

  /**
   * Simplified constructor with default values.
   *
   * @param name the column name
   * @param field the Arrow field definition
   * @param fieldId the field ID in Lance schema
   */
  public BlobColumn(String name, Field field, int fieldId) {
    this(name, field, fieldId, true, "none", -1, "Blob",
        field.isNullable(), new HashMap<>(), null);
  }

  public String getName() {
    return name;
  }

  public Field getField() {
    return field;
  }

  public int getFieldId() {
    return fieldId;
  }

  public boolean isExternalStorage() {
    return externalStorage;
  }

  public String getCompressionType() {
    return compressionType;
  }

  public long getMaxBlobSize() {
    return maxBlobSize;
  }

  public String getStorageClass() {
    return storageClass;
  }

  public boolean isNullable() {
    return nullable;
  }

  public Map<String, String> getMetadata() {
    return new HashMap<>(metadata);
  }

  public BlobColumnStatistics statistics() {
    return statistics;
  }

  @Override
  public String toString() {
    return String.format(
        "BlobColumnImpl{name='%s', fieldId=%d, storageClass='%s', " +
            "compressionType='%s', maxBlobSize=%d, nullable=%s}",
        name, fieldId, storageClass, compressionType, maxBlobSize, nullable
    );
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null || getClass() != obj.getClass()) return false;

    BlobColumn that = (BlobColumn) obj;
    return fieldId == that.fieldId &&
        name.equals(that.name) &&
        storageClass.equals(that.storageClass);
  }

  @Override
  public int hashCode() {
    int result = name.hashCode();
    result = 31 * result + fieldId;
    result = 31 * result + storageClass.hashCode();
    return result;
  }
}