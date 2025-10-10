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

import com.lancedb.lance.namespace.model.DescribeTableRequest;
import com.lancedb.lance.namespace.model.DescribeTableResponse;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class LanceNamespaceCredentialVendorTest {

  @Test
  public void testGetCredentials() {
    // Create a mock namespace
    LanceNamespace mockNamespace =
        new LanceNamespace() {
          @Override
          public void initialize(
              Map<String, String> configProperties,
              org.apache.arrow.memory.BufferAllocator allocator) {}

          @Override
          public DescribeTableResponse describeTable(DescribeTableRequest request) {
            // Return mock credentials
            Map<String, String> storageOptions = new HashMap<>();
            storageOptions.put("aws_access_key_id", "ASIA_TEST");
            storageOptions.put("aws_secret_access_key", "test_secret");
            storageOptions.put("aws_session_token", "test_token");
            storageOptions.put(
                "expires_at_millis", String.valueOf(System.currentTimeMillis() + 3600000));

            DescribeTableResponse response = new DescribeTableResponse();
            response.setStorageOptions(storageOptions);
            response.setLocation("s3://test-bucket/table.lance");
            response.setVersion(1L);

            return response;
          }
        };

    // Create vendor
    List<String> tableId = Arrays.asList("workspace", "table_name");
    LanceNamespaceCredentialVendor vendor =
        new LanceNamespaceCredentialVendor(mockNamespace, tableId);

    // Get credentials
    Map<String, Object> credentials = vendor.getCredentials();

    // Verify structure
    assertNotNull(credentials);
    assertTrue(credentials.containsKey("storage_options"));
    assertTrue(credentials.containsKey("expires_at_millis"));

    @SuppressWarnings("unchecked")
    Map<String, String> storageOptions = (Map<String, String>) credentials.get("storage_options");
    assertEquals("ASIA_TEST", storageOptions.get("aws_access_key_id"));
    assertEquals("test_secret", storageOptions.get("aws_secret_access_key"));
    assertEquals("test_token", storageOptions.get("aws_session_token"));

    Long expiresAtMillis = (Long) credentials.get("expires_at_millis");
    assertTrue(expiresAtMillis > System.currentTimeMillis());
  }

  @Test
  public void testMissingStorageOptions() {
    // Create a mock namespace that doesn't return storage options
    LanceNamespace mockNamespace =
        new LanceNamespace() {
          @Override
          public void initialize(
              Map<String, String> configProperties,
              org.apache.arrow.memory.BufferAllocator allocator) {}

          @Override
          public DescribeTableResponse describeTable(DescribeTableRequest request) {
            DescribeTableResponse response = new DescribeTableResponse();
            response.setLocation("s3://test-bucket/table.lance");
            return response;
          }
        };

    List<String> tableId = Arrays.asList("workspace", "table_name");
    LanceNamespaceCredentialVendor vendor =
        new LanceNamespaceCredentialVendor(mockNamespace, tableId);

    // Should throw exception
    RuntimeException exception =
        assertThrows(RuntimeException.class, () -> vendor.getCredentials());
    assertTrue(exception.getMessage().contains("did not return storage_options"));
  }

  @Test
  public void testMissingExpirationTime() {
    // Create a mock namespace that doesn't return expiration time
    LanceNamespace mockNamespace =
        new LanceNamespace() {
          @Override
          public void initialize(
              Map<String, String> configProperties,
              org.apache.arrow.memory.BufferAllocator allocator) {}

          @Override
          public DescribeTableResponse describeTable(DescribeTableRequest request) {
            Map<String, String> storageOptions = new HashMap<>();
            storageOptions.put("aws_access_key_id", "ASIA_TEST");
            // Missing expires_at_millis

            DescribeTableResponse response = new DescribeTableResponse();
            response.setStorageOptions(storageOptions);
            return response;
          }
        };

    List<String> tableId = Arrays.asList("workspace", "table_name");
    LanceNamespaceCredentialVendor vendor =
        new LanceNamespaceCredentialVendor(mockNamespace, tableId);

    // Should throw exception
    RuntimeException exception =
        assertThrows(RuntimeException.class, () -> vendor.getCredentials());
    assertTrue(exception.getMessage().contains("missing 'expires_at_millis'"));
  }
}
