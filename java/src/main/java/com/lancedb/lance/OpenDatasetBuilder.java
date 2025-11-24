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
package com.lancedb.lance;

import com.lancedb.lance.namespace.LanceNamespace;
import com.lancedb.lance.namespace.LanceNamespaceStorageOptionsProvider;
import com.lancedb.lance.namespace.model.DescribeTableRequest;
import com.lancedb.lance.namespace.model.DescribeTableResponse;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Builder for opening a Dataset.
 *
 * <p>This builder provides a fluent API for opening datasets either directly from a URI or from a
 * LanceNamespace. When using a namespace, the table location and storage options are automatically
 * fetched.
 *
 * <p>Example usage with URI:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.open()
 *     .uri("s3://bucket/table.lance")
 *     .readOptions(options)
 *     .build();
 * }</pre>
 *
 * <p>Example usage with namespace:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.open()
 *     .namespace(myNamespace)
 *     .tableId(Arrays.asList("my_table"))
 *     .build();
 * }</pre>
 */
public class OpenDatasetBuilder {
  private BufferAllocator allocator;
  private boolean selfManagedAllocator = false;
  private String uri;
  private LanceNamespace namespace;
  private List<String> tableId;
  private ReadOptions options = new ReadOptions.Builder().build();
  private boolean ignoreNamespaceTableStorageOptions = false;

  /** Creates a new builder instance. Package-private, use Dataset.open() instead. */
  OpenDatasetBuilder() {}

  /**
   * Sets the buffer allocator.
   *
   * @param allocator Arrow buffer allocator
   * @return this builder instance
   */
  public OpenDatasetBuilder allocator(BufferAllocator allocator) {
    Preconditions.checkNotNull(allocator);
    this.allocator = allocator;
    this.selfManagedAllocator = false;
    return this;
  }

  /**
   * Sets the dataset URI.
   *
   * <p>Either uri() or namespace()+tableId() must be specified, but not both.
   *
   * @param uri The dataset URI (e.g., "s3://bucket/table.lance" or "file:///path/to/table.lance")
   * @return this builder instance
   */
  public OpenDatasetBuilder uri(String uri) {
    this.uri = uri;
    return this;
  }

  /**
   * Sets the namespace.
   *
   * <p>Must be used together with tableId(). Either uri() or namespace()+tableId() must be
   * specified, but not both.
   *
   * @param namespace The namespace implementation to fetch table info from
   * @return this builder instance
   */
  public OpenDatasetBuilder namespace(LanceNamespace namespace) {
    this.namespace = namespace;
    return this;
  }

  /**
   * Sets the table identifier.
   *
   * <p>Must be used together with namespace(). Either uri() or namespace()+tableId() must be
   * specified, but not both.
   *
   * @param tableId The table identifier (e.g., Arrays.asList("my_table"))
   * @return this builder instance
   */
  public OpenDatasetBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Sets the read options.
   *
   * @param options Read options
   * @return this builder instance
   */
  public OpenDatasetBuilder readOptions(ReadOptions options) {
    this.options = options;
    return this;
  }

  /**
   * Sets whether to ignore storage options from the namespace's describe_table().
   *
   * @param ignoreNamespaceTableStorageOptions If true, storage options returned from
   *     describe_table() will be ignored (treated as null)
   * @return this builder instance
   */
  public OpenDatasetBuilder ignoreNamespaceTableStorageOptions(
      boolean ignoreNamespaceTableStorageOptions) {
    this.ignoreNamespaceTableStorageOptions = ignoreNamespaceTableStorageOptions;
    return this;
  }

  /**
   * Opens the dataset with the configured parameters.
   *
   * <p>If a namespace is configured, this automatically fetches the table location and storage
   * options from the namespace via describe_table().
   *
   * @return Dataset
   * @throws IllegalArgumentException if required parameters are missing or invalid
   */
  public Dataset build() {
    // Validate that exactly one of uri or namespace+tableId is provided
    boolean hasUri = uri != null;
    boolean hasNamespace = namespace != null && tableId != null;

    if (hasUri && hasNamespace) {
      throw new IllegalArgumentException(
          "Cannot specify both uri and namespace+tableId. Use one or the other.");
    }
    if (!hasUri && !hasNamespace) {
      if (namespace != null) {
        throw new IllegalArgumentException(
            "namespace is set but tableId is missing. Both namespace and tableId must be"
                + " provided together.");
      } else if (tableId != null) {
        throw new IllegalArgumentException(
            "tableId is set but namespace is missing. Both namespace and tableId must be"
                + " provided together.");
      } else {
        throw new IllegalArgumentException("Either uri or namespace+tableId must be provided.");
      }
    }

    Preconditions.checkNotNull(options, "options must be set");

    // Create allocator if not provided
    if (allocator == null) {
      allocator = new RootAllocator(Long.MAX_VALUE);
      selfManagedAllocator = true;
    }

    // Handle namespace-based opening
    if (hasNamespace) {
      return buildFromNamespace();
    }

    // Handle URI-based opening
    return Dataset.open(allocator, selfManagedAllocator, uri, options);
  }

  private Dataset buildFromNamespace() {
    // Call describe_table to get location and storage options
    DescribeTableRequest request = new DescribeTableRequest();
    request.setId(tableId);
    // Only set version if present
    options.getVersion().ifPresent(v -> request.setVersion(Long.valueOf(v)));

    DescribeTableResponse response = namespace.describeTable(request);

    String location = response.getLocation();
    if (location == null || location.isEmpty()) {
      throw new IllegalArgumentException("Namespace did not return a table location");
    }

    Map<String, String> namespaceStorageOptions =
        ignoreNamespaceTableStorageOptions ? null : response.getStorageOptions();

    ReadOptions.Builder optionsBuilder =
        new ReadOptions.Builder()
            .setIndexCacheSizeBytes(options.getIndexCacheSizeBytes())
            .setMetadataCacheSizeBytes(options.getMetadataCacheSizeBytes());

    if (namespaceStorageOptions != null && !namespaceStorageOptions.isEmpty()) {
      LanceNamespaceStorageOptionsProvider storageOptionsProvider =
          new LanceNamespaceStorageOptionsProvider(namespace, tableId);
      optionsBuilder.setStorageOptionsProvider(storageOptionsProvider);
    }

    options.getVersion().ifPresent(optionsBuilder::setVersion);
    options.getBlockSize().ifPresent(optionsBuilder::setBlockSize);
    options.getSerializedManifest().ifPresent(optionsBuilder::setSerializedManifest);
    options
        .getS3CredentialsRefreshOffsetSeconds()
        .ifPresent(optionsBuilder::setS3CredentialsRefreshOffsetSeconds);

    Map<String, String> storageOptions = new HashMap<>(options.getStorageOptions());
    if (namespaceStorageOptions != null) {
      storageOptions.putAll(namespaceStorageOptions);
    }
    optionsBuilder.setStorageOptions(storageOptions);

    // Open dataset with regular open method
    return Dataset.open(allocator, selfManagedAllocator, location, optionsBuilder.build());
  }
}
