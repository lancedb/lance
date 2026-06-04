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

import org.lance.namespace.model.CreateNamespaceRequest;
import org.lance.namespace.model.ListNamespacesRequest;

import com.sun.net.httpserver.HttpServer;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class SigV4AuthTest {
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
  void testSigV4ConnectAndOperate() {
    Map<String, String> backendConfig = new HashMap<>();
    backendConfig.put("root", tempDir.toString());

    RestAdapter adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", 0);
    adapter.start();
    try {
      Map<String, String> clientConfig = new HashMap<>();
      clientConfig.put("uri", "http://127.0.0.1:" + adapter.getPort());
      clientConfig.put("rest.auth.type", "sigv4");
      clientConfig.put("rest.auth.sigv4.region", "us-east-1");
      clientConfig.put("rest.auth.sigv4.service", "execute-api");

      RestNamespace ns = new RestNamespace();
      ns.initialize(clientConfig, allocator);

      ns.createNamespace(new CreateNamespaceRequest().id(Arrays.asList("sigv4test")));
      var resp = ns.listNamespaces(new ListNamespacesRequest());
      assertTrue(resp.getNamespaces().contains("sigv4test"));

      ns.close();
    } finally {
      adapter.close();
    }
  }

  @Test
  void testSigV4MissingRegionFailsAtConnect() {
    Map<String, String> backendConfig = new HashMap<>();
    backendConfig.put("root", tempDir.toString());

    RestAdapter adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", 0);
    adapter.start();
    try {
      Map<String, String> clientConfig = new HashMap<>();
      clientConfig.put("uri", "http://127.0.0.1:" + adapter.getPort());
      clientConfig.put("rest.auth.type", "sigv4");

      RestNamespace ns = new RestNamespace();
      RuntimeException ex =
          assertThrows(RuntimeException.class, () -> ns.initialize(clientConfig, allocator));
      assertTrue(ex.getMessage().contains("rest.auth.sigv4.region"));
    } finally {
      adapter.close();
    }
  }

  // Signature correctness is verified at the Rust layer (AWS test vectors + botocore).
  @Test
  void testSigV4SignatureHeadersPresent() throws IOException {
    List<String> capturedAuth = new ArrayList<>();

    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/",
        exchange -> {
          String auth = exchange.getRequestHeaders().getFirst("Authorization");
          if (auth != null) {
            capturedAuth.add(auth);
          }
          byte[] body = "{\"namespaces\":[]}".getBytes();
          exchange.sendResponseHeaders(200, body.length);
          try (OutputStream os = exchange.getResponseBody()) {
            os.write(body);
          }
        });
    server.start();
    int port = server.getAddress().getPort();

    try {
      Map<String, String> clientConfig = new HashMap<>();
      clientConfig.put("uri", "http://127.0.0.1:" + port);
      clientConfig.put("rest.auth.type", "sigv4");
      clientConfig.put("rest.auth.sigv4.region", "us-east-1");
      clientConfig.put("rest.auth.sigv4.service", "execute-api");

      RestNamespace ns = new RestNamespace();
      ns.initialize(clientConfig, allocator);

      try {
        ns.listNamespaces(new ListNamespacesRequest());
      } catch (Exception ignored) {
      }

      ns.close();

      assertFalse(capturedAuth.isEmpty(), "no Authorization header captured");
      String auth = capturedAuth.get(0);
      assertTrue(auth.startsWith("AWS4-HMAC-SHA256"), "expected SigV4 header, got: " + auth);
      assertTrue(auth.contains("Credential="), "missing Credential in: " + auth);
      assertTrue(auth.contains("SignedHeaders="), "missing SignedHeaders in: " + auth);
      assertTrue(auth.matches(".*Signature=[a-f0-9]{64}.*"), "missing Signature in: " + auth);
    } finally {
      server.stop(0);
    }
  }
}
