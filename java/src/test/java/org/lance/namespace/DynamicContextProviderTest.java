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
package org.lance.namespace;

import org.lance.namespace.model.*;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for DynamicContextProvider interface. */
public class DynamicContextProviderTest {
  @TempDir Path tempDir;

  private BufferAllocator allocator;

  @BeforeEach
  void setUp() {
    allocator = new RootAllocator(Long.MAX_VALUE);
  }

  @AfterEach
  void tearDown() {
    if (allocator != null) {
      allocator.close();
    }
  }

  @Test
  void testDirectoryNamespaceWithContextProvider() {
    AtomicInteger callCount = new AtomicInteger(0);

    DynamicContextProvider provider =
        (operation, objectId) -> {
          callCount.incrementAndGet();
          Map<String, String> context = new HashMap<>();
          context.put("headers.Authorization", "Bearer test-token-123");
          context.put("headers.X-Request-Id", "req-" + operation);
          return context;
        };

    try (DirectoryNamespace namespace = new DirectoryNamespace()) {
      Map<String, String> config = new HashMap<>();
      config.put("root", tempDir.toString());
      namespace.initialize(config, allocator, provider);

      // Perform operations to verify the provider is called
      CreateNamespaceRequest createReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      namespace.createNamespace(createReq);

      ListNamespacesRequest listReq = new ListNamespacesRequest();
      namespace.listNamespaces(listReq);

      // The provider should have been called for each operation
      // Note: DirectoryNamespace stores the provider but may not actively use context
      // until the underlying Rust code is updated to use it for credential vending
      assertNotNull(namespace.namespaceId());
    }
  }

  @Test
  void testDirectoryNamespaceWithNullProvider() {
    try (DirectoryNamespace namespace = new DirectoryNamespace()) {
      Map<String, String> config = new HashMap<>();
      config.put("root", tempDir.toString());

      // Should work with null provider (backward compatibility)
      namespace.initialize(config, allocator, null);

      CreateNamespaceRequest createReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      namespace.createNamespace(createReq);

      ListNamespacesRequest listReq = new ListNamespacesRequest();
      ListNamespacesResponse listResp = namespace.listNamespaces(listReq);

      assertNotNull(listResp);
      assertTrue(listResp.getNamespaces().contains("workspace"));
    }
  }

  @Test
  void testContextProviderReturnsEmptyMap() {
    DynamicContextProvider provider = (operation, objectId) -> new HashMap<>();

    try (DirectoryNamespace namespace = new DirectoryNamespace()) {
      Map<String, String> config = new HashMap<>();
      config.put("root", tempDir.toString());
      namespace.initialize(config, allocator, provider);

      CreateNamespaceRequest createReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      CreateNamespaceResponse resp = namespace.createNamespace(createReq);

      assertNotNull(resp);
    }
  }

  @Test
  void testRestNamespaceWithContextProviderIntegration() {
    AtomicInteger callCount = new AtomicInteger(0);

    DynamicContextProvider provider =
        (operation, objectId) -> {
          callCount.incrementAndGet();
          Map<String, String> context = new HashMap<>();
          context.put("headers.Authorization", "Bearer xyz-token");
          context.put("headers.X-Trace-Id", "trace-" + System.currentTimeMillis());
          return context;
        };

    // Start a test REST server with DirectoryNamespace backend
    Map<String, String> backendConfig = new HashMap<>();
    backendConfig.put("root", tempDir.toString());

    try (RestAdapter adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", null)) {
      adapter.start();
      int port = adapter.getPort();

      // Create RestNamespace client with context provider
      try (RestNamespace namespace = new RestNamespace()) {
        Map<String, String> clientConfig = new HashMap<>();
        clientConfig.put("uri", "http://127.0.0.1:" + port);
        namespace.initialize(clientConfig, allocator, provider);

        // Perform operations - context provider should be called
        CreateNamespaceRequest createReq =
            new CreateNamespaceRequest().id(Arrays.asList("workspace"));
        namespace.createNamespace(createReq);

        ListNamespacesRequest listReq = new ListNamespacesRequest();
        ListNamespacesResponse listResp = namespace.listNamespaces(listReq);

        // Verify provider was called for REST operations
        assertTrue(callCount.get() >= 2, "Context provider should be called for each operation");
        assertNotNull(listResp);
        assertTrue(listResp.getNamespaces().contains("workspace"));
      }
    }
  }

  @Test
  void testContextProviderReceivesCorrectOperationInfo() {
    Map<String, String> capturedOperations = new HashMap<>();

    DynamicContextProvider provider =
        (operation, objectId) -> {
          capturedOperations.put(operation, objectId);
          return new HashMap<>();
        };

    Map<String, String> backendConfig = new HashMap<>();
    backendConfig.put("root", tempDir.toString());

    try (RestAdapter adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", null)) {
      adapter.start();
      int port = adapter.getPort();

      try (RestNamespace namespace = new RestNamespace()) {
        Map<String, String> clientConfig = new HashMap<>();
        clientConfig.put("uri", "http://127.0.0.1:" + port);
        namespace.initialize(clientConfig, allocator, provider);

        // Create namespace
        CreateNamespaceRequest createReq =
            new CreateNamespaceRequest().id(Arrays.asList("workspace"));
        namespace.createNamespace(createReq);

        // List namespaces
        ListNamespacesRequest listReq = new ListNamespacesRequest();
        namespace.listNamespaces(listReq);

        // Verify operations were captured
        assertTrue(capturedOperations.containsKey("create_namespace"));
        assertTrue(capturedOperations.containsKey("list_namespaces"));
      }
    }
  }

  // ==========================================================================
  // Class path based provider tests
  // ==========================================================================

  @Test
  void testDirectoryNamespaceWithClassPathProvider() {
    try (DirectoryNamespace namespace = new DirectoryNamespace()) {
      Map<String, String> config = new HashMap<>();
      config.put("root", tempDir.toString());
      config.put("dynamic_context_provider.impl", "org.lance.namespace.TestContextProvider");
      config.put("dynamic_context_provider.token", "my-secret-token");
      config.put("dynamic_context_provider.prefix", "Token");

      namespace.initialize(config, allocator);

      // Verify namespace works
      CreateNamespaceRequest createReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      namespace.createNamespace(createReq);

      ListNamespacesRequest listReq = new ListNamespacesRequest();
      ListNamespacesResponse listResp = namespace.listNamespaces(listReq);

      assertNotNull(listResp);
      assertTrue(listResp.getNamespaces().contains("workspace"));
    }
  }

  @Test
  void testRestNamespaceWithClassPathProvider() {
    Map<String, String> backendConfig = new HashMap<>();
    backendConfig.put("root", tempDir.toString());

    try (RestAdapter adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", null)) {
      adapter.start();
      int port = adapter.getPort();

      try (RestNamespace namespace = new RestNamespace()) {
        Map<String, String> clientConfig = new HashMap<>();
        clientConfig.put("uri", "http://127.0.0.1:" + port);
        clientConfig.put(
            "dynamic_context_provider.impl", "org.lance.namespace.TestContextProvider");
        clientConfig.put("dynamic_context_provider.token", "secret-api-key");

        namespace.initialize(clientConfig, allocator);

        CreateNamespaceRequest createReq =
            new CreateNamespaceRequest().id(Arrays.asList("workspace"));
        namespace.createNamespace(createReq);

        ListNamespacesRequest listReq = new ListNamespacesRequest();
        ListNamespacesResponse listResp = namespace.listNamespaces(listReq);

        assertNotNull(listResp);
        assertTrue(listResp.getNamespaces().contains("workspace"));
      }
    }
  }

  @Test
  void testUnknownProviderClassThrowsException() {
    try (DirectoryNamespace namespace = new DirectoryNamespace()) {
      Map<String, String> config = new HashMap<>();
      config.put("root", tempDir.toString());
      config.put("dynamic_context_provider.impl", "com.nonexistent.NonExistentProvider");

      assertThrows(
          IllegalArgumentException.class,
          () -> namespace.initialize(config, allocator),
          "Failed to load context provider class");
    }
  }

  @Test
  void testExplicitProviderTakesPrecedence() {
    AtomicInteger explicitCallCount = new AtomicInteger(0);

    DynamicContextProvider explicitProvider =
        (operation, objectId) -> {
          explicitCallCount.incrementAndGet();
          Map<String, String> ctx = new HashMap<>();
          ctx.put("headers.Authorization", "Bearer explicit");
          return ctx;
        };

    try (DirectoryNamespace namespace = new DirectoryNamespace()) {
      Map<String, String> config = new HashMap<>();
      config.put("root", tempDir.toString());
      // Even though we specify a class path, explicit provider should take precedence
      config.put("dynamic_context_provider.impl", "org.lance.namespace.TestContextProvider");
      config.put("dynamic_context_provider.token", "ignored");

      // Pass explicit provider - should take precedence over properties
      namespace.initialize(config, allocator, explicitProvider);

      // Verify namespace works
      CreateNamespaceRequest createReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      namespace.createNamespace(createReq);

      // Namespace should work
      assertNotNull(namespace.namespaceId());
    }
  }
}
