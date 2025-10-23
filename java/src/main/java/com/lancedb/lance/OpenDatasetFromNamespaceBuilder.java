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
 * Builder for opening a Dataset from a LanceNamespace.
 *
 * <p>This builder provides a fluent API for configuring and opening datasets from namespaces.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.openFromNamespace()
 *     .namespace(myNamespace)
 *     .tableId(Arrays.asList("my_table"))
 *     .refreshStorageOptions(true)
 *     .build();
 * }</pre>
 */
public class OpenDatasetFromNamespaceBuilder {
  private BufferAllocator allocator;
  private boolean selfManagedAllocator = false;
  private LanceNamespace namespace;
  private List<String> tableId;
  private ReadOptions options = new ReadOptions.Builder().build();
  private boolean refreshStorageOptions = false;

  /** Creates a new builder instance. Package-private, use Dataset.openFromNamespace() instead. */
  OpenDatasetFromNamespaceBuilder() {}

  /**
   * Sets the buffer allocator.
   *
   * @param allocator Arrow buffer allocator
   * @return this builder instance
   */
  public OpenDatasetFromNamespaceBuilder allocator(BufferAllocator allocator) {
    Preconditions.checkNotNull(allocator);
    this.allocator = allocator;
    this.selfManagedAllocator = false;
    return this;
  }

  /**
   * Sets the namespace.
   *
   * @param namespace The namespace implementation to fetch table info from
   * @return this builder instance
   */
  public OpenDatasetFromNamespaceBuilder namespace(LanceNamespace namespace) {
    this.namespace = namespace;
    return this;
  }

  /**
   * Sets the table identifier.
   *
   * @param tableId The table identifier (e.g., Arrays.asList("my_table"))
   * @return this builder instance
   */
  public OpenDatasetFromNamespaceBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Sets the read options.
   *
   * @param options Read options
   * @return this builder instance
   */
  public OpenDatasetFromNamespaceBuilder options(ReadOptions options) {
    this.options = options;
    return this;
  }

  /**
   * Sets whether storage options should be automatically refreshed before they expire.
   *
   * <p>This is currently only used for refreshing AWS temporary access credentials. When enabled,
   * the namespace will be queried periodically to fetch new temporary credentials before the
   * current ones expire. The new storage options will contain updated AWS access credentials with a
   * new expiration time.
   *
   * @param refreshStorageOptions If true, storage options will be automatically refreshed
   * @return this builder instance
   */
  public OpenDatasetFromNamespaceBuilder refreshStorageOptions(boolean refreshStorageOptions) {
    this.refreshStorageOptions = refreshStorageOptions;
    return this;
  }

  /**
   * Opens the dataset from the configured namespace.
   *
   * <p>This automatically fetches the table location and storage options from the namespace via
   * describe_table().
   *
   * @return Dataset
   * @throws IllegalArgumentException if required parameters are missing or invalid
   */
  public Dataset build() {
    Preconditions.checkNotNull(namespace, "namespace must be set");
    Preconditions.checkNotNull(tableId, "tableId must be set");
    Preconditions.checkNotNull(options, "options must be set");

    // Create allocator if not provided
    if (allocator == null) {
      allocator = new RootAllocator(Long.MAX_VALUE);
      selfManagedAllocator = true;
    }

    // Call describe_table to get location and storage options
    DescribeTableRequest request = new DescribeTableRequest();
    request.setId(tableId);
    // Only set version if present
    options.getVersion().ifPresent(v -> request.setVersion(Long.valueOf(v)));

    DescribeTableResponse response = namespace.describeTable(request);

    // Extract location
    String location = response.getLocation();
    if (location == null || location.isEmpty()) {
      throw new IllegalArgumentException("Namespace did not return a table location");
    }

    // Build new ReadOptions with initial storage options
    ReadOptions.Builder optionsBuilder =
        new ReadOptions.Builder()
            .setIndexCacheSizeBytes(options.getIndexCacheSizeBytes())
            .setMetadataCacheSizeBytes(options.getMetadataCacheSizeBytes());

    // Only set storage options provider if refresh is enabled
    if (refreshStorageOptions) {
      LanceNamespaceStorageOptionsProvider storageOptionsProvider =
          new LanceNamespaceStorageOptionsProvider(namespace, tableId);
      optionsBuilder.setStorageOptionsProvider(storageOptionsProvider);
    }

    // Set optional fields only if present
    options.getVersion().ifPresent(optionsBuilder::setVersion);
    options.getBlockSize().ifPresent(optionsBuilder::setBlockSize);
    options.getSerializedManifest().ifPresent(optionsBuilder::setSerializedManifest);

    // Add initial storage options from describe_table response if present
    Map<String, String> storageOptions = new HashMap<>(options.getStorageOptions());
    if (response.getStorageOptions() != null) {
      storageOptions.putAll(response.getStorageOptions());
    }
    optionsBuilder.setStorageOptions(storageOptions);

    // Open dataset with regular open method
    return Dataset.open(allocator, selfManagedAllocator, location, optionsBuilder.build());
  }
}
