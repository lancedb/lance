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

import org.lance.JniLoader;

import java.io.Closeable;
import java.util.Map;

/**
 * REST adapter server for testing namespace implementations.
 *
 * <p>This class wraps a namespace backend (e.g., DirectoryNamespace) and exposes it via a REST API.
 * It's primarily used for testing RestNamespace implementations.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * Map<String, String> backendConfig = new HashMap<>();
 * backendConfig.put("root", "/tmp/test-data");
 *
 * try (RestAdapter adapter = new RestAdapter("dir", backendConfig, "127.0.0.1", 8080)) {
 *     adapter.serve();
 *
 *     // Now you can connect with RestNamespace
 *     Map<String, String> clientConfig = new HashMap<>();
 *     clientConfig.put("uri", "http://127.0.0.1:8080");
 *     RestNamespace client = new RestNamespace();
 *     client.initialize(clientConfig, allocator);
 *
 *     // Use the client...
 * }
 * }</pre>
 */
public class RestAdapter implements Closeable, AutoCloseable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeRestAdapterHandle;
  private boolean serverStarted = false;

  /**
   * Creates a new REST adapter with the given backend namespace.
   *
   * @param namespaceImpl The namespace implementation type (e.g., "dir" for DirectoryNamespace)
   * @param backendConfig Configuration properties for the backend namespace
   * @param host Host to bind the server to
   * @param port Port to bind the server to
   */
  public RestAdapter(
      String namespaceImpl, Map<String, String> backendConfig, String host, int port) {
    if (namespaceImpl == null || namespaceImpl.isEmpty()) {
      throw new IllegalArgumentException("namespace implementation cannot be null or empty");
    }
    if (backendConfig == null) {
      throw new IllegalArgumentException("backend config cannot be null");
    }
    if (host == null || host.isEmpty()) {
      throw new IllegalArgumentException("host cannot be null or empty");
    }
    if (port <= 0 || port > 65535) {
      throw new IllegalArgumentException("port must be between 1 and 65535");
    }

    this.nativeRestAdapterHandle = createNative(namespaceImpl, backendConfig, host, port);
  }

  /**
   * Creates a new REST adapter with default host (127.0.0.1) and port (2333).
   *
   * @param namespaceImpl The namespace implementation type
   * @param backendConfig Configuration properties for the backend namespace
   */
  public RestAdapter(String namespaceImpl, Map<String, String> backendConfig) {
    this(namespaceImpl, backendConfig, "127.0.0.1", 2333);
  }

  /**
   * Start the REST server in the background.
   *
   * <p>This method returns immediately after starting the server. The server runs in a background
   * thread until {@link #stop()} is called or the adapter is closed.
   */
  public void serve() {
    if (nativeRestAdapterHandle == 0) {
      throw new IllegalStateException("RestAdapter not initialized");
    }
    if (serverStarted) {
      throw new IllegalStateException("Server already started");
    }

    serve(nativeRestAdapterHandle);
    serverStarted = true;
  }

  /**
   * Stop the REST server.
   *
   * <p>This method is idempotent - calling it multiple times has no effect.
   */
  public void stop() {
    if (nativeRestAdapterHandle != 0 && serverStarted) {
      stop(nativeRestAdapterHandle);
      serverStarted = false;
    }
  }

  @Override
  public void close() {
    stop();
    if (nativeRestAdapterHandle != 0) {
      releaseNative(nativeRestAdapterHandle);
      nativeRestAdapterHandle = 0;
    }
  }

  // Native methods
  private native long createNative(
      String namespaceImpl, Map<String, String> backendConfig, String host, int port);

  private native void serve(long handle);

  private native void stop(long handle);

  private native void releaseNative(long handle);
}
