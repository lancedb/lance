package com.lancedb.lance.blob;

import java.util.HashMap;
import java.util.Map;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

/**
 * Builder class for creating blob columns in Lance datasets.
 *
 * BlobColumnBuilder provides a fluent interface for configuring and creating
 * blob columns with various storage and compression options. It handles the
 * creation of appropriate Arrow field definitions with Lance-specific metadata
 * for blob storage.
 *
 * <p>Example usage:
 * <pre>{@code
 * BlobColumn imageColumn = BlobColumnBuilder.create("image_data")
 *     .compressionType("lz4")
 *     .maxBlobSize(10 * 1024 * 1024) // 10MB limit
 *     .nullable(true)
 *     .metadata("description", "User profile images")
 *     .build();
 *
 * // Add to dataset
 * dataset.addBlobColumn(imageColumn);
 * }</pre>
 */
public class BlobColumnBuilder {

  private final String name;
  private boolean nullable = true;
  private String compressionType = "none";
  private long maxBlobSize = -1; // No limit by default
  private final Map<String, String> metadata = new HashMap<>();

  // Lance-specific metadata keys
  private static final String LANCE_STORAGE_CLASS_KEY = "lance:storage_class";
  private static final String LANCE_COMPRESSION_KEY = "lance:compression";
  private static final String LANCE_MAX_BLOB_SIZE_KEY = "lance:max_blob_size";
  private static final String LANCE_EXTERNAL_STORAGE_KEY = "lance:external_storage";

  /**
   * Private constructor. Use static factory methods to create instances.
   */
  private BlobColumnBuilder(String name) {
    this.name = name;
  }

  /**
   * Set whether the column can contain null values.
   *
   * @param nullable true if the column should be nullable, false otherwise
   * @return this builder instance for method chaining
   */
  public BlobColumnBuilder nullable(boolean nullable) {
    this.nullable = nullable;
    return this;
  }

  /**
   * Set the compression type for blob data.
   *
   * @param compressionType the compression type ("none", "lz4", "zstd", "gzip")
   * @return this builder instance for method chaining
   * @throws IllegalArgumentException if compressionType is null
   */
  public BlobColumnBuilder compressionType(String compressionType) {
    if (compressionType == null) {
      throw new IllegalArgumentException("Compression type cannot be null");
    }
    this.compressionType = compressionType.toLowerCase();
    return this;
  }

  /**
   * Set the maximum size limit for individual blobs in this column.
   *
   * @param maxBlobSize the maximum blob size in bytes, or -1 for no limit
   * @return this builder instance for method chaining
   * @throws IllegalArgumentException if maxBlobSize is 0 or negative (except -1)
   */
  public BlobColumnBuilder maxBlobSize(long maxBlobSize) {
    if (maxBlobSize == 0 || (maxBlobSize < 0 && maxBlobSize != -1)) {
      throw new IllegalArgumentException("Max blob size must be positive or -1 for no limit");
    }
    this.maxBlobSize = maxBlobSize;
    return this;
  }

  /**
   * Set a description for the blob column.
   *
   * @param description the column description
   * @return this builder instance for method chaining
   */
  public BlobColumnBuilder description(String description) {
    if (description != null) {
      this.metadata.put("description", description);
    }
    return this;
  }

  /**
   * Add metadata to the blob column.
   *
   * @param key   the metadata key
   * @param value the metadata value
   * @return this builder instance for method chaining
   * @throws IllegalArgumentException if key or value is null
   */
  public BlobColumnBuilder metadata(String key, String value) {
    if (key == null || value == null) {
      throw new IllegalArgumentException("Metadata key and value cannot be null");
    }
    this.metadata.put(key, value);
    return this;
  }

  /**
   * Add multiple metadata entries to the blob column.
   *
   * @param metadata a map of metadata key-value pairs
   * @return this builder instance for method chaining
   * @throws IllegalArgumentException if metadata is null
   */
  public BlobColumnBuilder metadata(Map<String, String> metadata) {
    if (metadata == null) {
      throw new IllegalArgumentException("Metadata map cannot be null");
    }
    this.metadata.putAll(metadata);
    return this;
  }

  /**
   * Build the BlobColumn with the configured settings.
   *
   * @return a new BlobColumn instance
   */
  public BlobColumn build() {
    // Create Lance-specific metadata
    Map<String, String> fieldMetadata = new HashMap<>(this.metadata);
    fieldMetadata.put(LANCE_STORAGE_CLASS_KEY, "Blob");
    fieldMetadata.put(LANCE_COMPRESSION_KEY, compressionType);
    fieldMetadata.put(LANCE_EXTERNAL_STORAGE_KEY, "true");

    if (maxBlobSize > 0) {
      fieldMetadata.put(LANCE_MAX_BLOB_SIZE_KEY, String.valueOf(maxBlobSize));
    }

    // Create Arrow field with LargeBinary type (appropriate for blobs)
    FieldType fieldType = new FieldType(nullable, ArrowType.LargeBinary.INSTANCE,
        null, fieldMetadata);
    Field field = new Field(name, fieldType, null);

    // For now, we use a placeholder field ID. In a real implementation,
    // this would be assigned by the Lance schema system
    int fieldId = name.hashCode(); // Temporary field ID generation

    return new BlobColumn(name, field, fieldId, true, compressionType,
        maxBlobSize, "Blob", nullable, fieldMetadata, null);
  }

  /**
   * Create a new BlobColumnBuilder with the specified column name.
   *
   * @param name the name of the blob column
   * @return a new BlobColumnBuilder instance
   * @throws IllegalArgumentException if name is null or empty
   */
  public static BlobColumnBuilder create(String name) {
    if (name == null || name.trim().isEmpty()) {
      throw new IllegalArgumentException("Column name cannot be null or empty");
    }
    return new BlobColumnBuilder(name.trim());
  }

  /**
   * Create a simple blob column with default settings.
   *
   * @param name the column name
   * @return a BlobColumn with default settings (nullable, no compression, no size limit)
   */
  public static BlobColumn createSimple(String name) {
    return create(name).build();
  }

  /**
   * Create a blob column optimized for images.
   *
   * @param name the column name
   * @return a BlobColumn configured for image storage
   */
  public static BlobColumn createForImages(String name) {
    return create(name)
        .compressionType("lz4")
        .maxBlobSize(50 * 1024 * 1024) // 50MB limit
        .description("Image data storage")
        .metadata("content_type", "image")
        .build();
  }

  /**
   * Create a blob column optimized for documents.
   *
   * @param name the column name
   * @return a BlobColumn configured for document storage
   */
  public static BlobColumn createForDocuments(String name) {
    return create(name)
        .compressionType("zstd")
        .maxBlobSize(100 * 1024 * 1024) // 100MB limit
        .description("Document data storage")
        .metadata("content_type", "document")
        .build();
  }

  /**
   * Create a blob column optimized for small binary data.
   *
   * @param name the column name
   * @return a BlobColumn configured for small binary data
   */
  public static BlobColumn createForSmallBinary(String name) {
    return create(name)
        .compressionType("none")
        .maxBlobSize(1024 * 1024) // 1MB limit
        .description("Small binary data storage")
        .metadata("content_type", "binary")
        .build();
  }
}
