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

import com.lancedb.lance.io.CredentialVendor;
import com.lancedb.lance.namespace.model.DescribeTableRequest;
import com.lancedb.lance.namespace.model.DescribeTableResponse;

import java.util.List;
import java.util.Map;

/**
 * Credential vendor that fetches credentials from a LanceNamespace.
 *
 * <p>This vendor automatically fetches fresh credentials by calling the namespace's describeTable()
 * method, which returns both the table location and time-limited storage credentials.
 *
 * <p>This is the recommended approach for LanceDB Cloud and other namespace-based deployments, as
 * it handles credential refresh automatically.
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
 * // Create credential vendor
 * LanceNamespaceCredentialVendor vendor = new LanceNamespaceCredentialVendor(
 *     namespace,
 *     Arrays.asList("workspace", "table_name")
 * );
 *
 * // Use with dataset - credentials auto-refresh!
 * Dataset dataset = Dataset.open(
 *     "s3://bucket/table.lance",
 *     new ReadOptions.Builder()
 *         .setCredentialVendor(vendor)
 *         .build()
 * );
 * }</pre>
 */
public class LanceNamespaceCredentialVendor implements CredentialVendor {

  private final com.lancedb.lance.namespace.LanceNamespace namespace;
  private final List<String> tableId;

  /**
   * Create a credential vendor that fetches credentials from a LanceNamespace.
   *
   * @param namespace The namespace instance to fetch credentials from
   * @param tableId The table identifier (e.g., ["workspace", "table_name"])
   */
  public LanceNamespaceCredentialVendor(
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
  public Map<String, String> getCredentials() {
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
}
