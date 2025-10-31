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
package com.lancedb.lance.namespace;

import com.lancedb.lance.io.StorageOptionsProvider;
import com.lancedb.lance.namespace.model.DescribeTableRequest;
import com.lancedb.lance.namespace.model.DescribeTableResponse;

import java.util.List;
import java.util.Map;

/**
 * Storage options provider that fetches storage options from a LanceNamespace.
 *
 * <p>This provider automatically fetches fresh storage options by calling the namespace's
 * describeTable() method, which returns both the table location and time-limited storage options.
 * This is currently only used for refreshing AWS temporary access credentials.
 *
 * <p>This is the recommended approach for LanceDB Cloud and other namespace-based deployments, as
 * it handles storage options refresh automatically.
 *
 * <h2>Example Usage</h2>
 *
 * <pre>{@code
 * // Connect to a namespace (e.g., LanceDB Cloud)
 * LanceNamespace namespace = LanceNamespaces.connect("rest", Map.of(
 *     "url", "https://api.lancedb.com",
 *     "api_key", "your-api-key"
 * ));
 *
 * // Create storage options provider
 * LanceNamespaceStorageOptionsProvider provider = new LanceNamespaceStorageOptionsProvider(
 *     namespace,
 *     Arrays.asList("workspace", "table_name")
 * );
 *
 * // Use with dataset - storage options auto-refresh!
 * Dataset dataset = Dataset.open(
 *     "s3://bucket/table.lance",
 *     new ReadOptions.Builder()
 *         .setStorageOptionsProvider(provider)
 *         .build()
 * );
 * }</pre>
 */
public class LanceNamespaceStorageOptionsProvider implements StorageOptionsProvider {

  private final com.lancedb.lance.namespace.LanceNamespace namespace;
  private final List<String> tableId;

  /**
   * Create a storage options provider that fetches storage options from a LanceNamespace.
   *
   * @param namespace The namespace instance to fetch storage options from
   * @param tableId The table identifier (e.g., ["workspace", "table_name"])
   */
  public LanceNamespaceStorageOptionsProvider(
      com.lancedb.lance.namespace.LanceNamespace namespace, List<String> tableId) {
    this.namespace = namespace;
    this.tableId = tableId;
  }

  /**
   * Fetch credentials from the namespace.
   *
   * <p>This calls namespace.describeTable() to get the latest credentials and their expiration
   * time.
   *
   * @return Flat map of string key-value pairs containing credentials and expires_at_millis
   * @throws RuntimeException if the namespace doesn't return storage credentials or expiration time
   */
  @Override
  public Map<String, String> fetchStorageOptions() {
    // Create describe table request with table ID
    DescribeTableRequest request = new DescribeTableRequest();
    request.setId(tableId);

    // Call namespace to describe the table and get credentials
    DescribeTableResponse response = namespace.describeTable(request);

    // Extract storage options - should already be a flat Map<String, String>
    Map<String, String> storageOptions = response.getStorageOptions();
    if (storageOptions == null || storageOptions.isEmpty()) {
      throw new RuntimeException(
          "Namespace did not return storage_options. "
              + "Ensure the namespace supports credential vending.");
    }

    // Verify expires_at_millis is present
    if (!storageOptions.containsKey("expires_at_millis")) {
      throw new RuntimeException(
          "Namespace storage_options missing 'expires_at_millis'. "
              + "Credential refresh will not work properly.");
    }

    // Return storage_options directly - it's already a flat Map<String, String>
    return storageOptions;
  }

  /**
   * Return a human-readable unique identifier for this provider instance.
   *
   * <p>This creates a semantic ID based on the namespace's ID and the table ID, enabling proper
   * equality comparison and caching.
   *
   * @return A human-readable unique identifier string combining namespace and table info
   */
  @Override
  public String providerId() {
    // Call namespaceId() on the namespace (requires lance-namespace >= 0.0.20)
    String namespaceId = namespace.namespaceId();
    return String.format(
        "LanceNamespaceStorageOptionsProvider { namespace: %s, table_id: %s }",
        namespaceId, tableId);
  }
}
