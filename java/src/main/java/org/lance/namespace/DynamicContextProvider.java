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

import java.util.Map;

/**
 * Interface for providing dynamic per-request context to namespace operations.
 *
 * <p>Implementations can generate per-request context (e.g., authentication headers) based on the
 * operation being performed. The provider is called synchronously before each namespace operation.
 *
 * <p>For RestNamespace, context keys that start with {@code headers.} are converted to HTTP headers
 * by stripping the prefix. For example, {@code {"headers.Authorization": "Bearer abc123"}} becomes
 * the {@code Authorization: Bearer abc123} header. Keys without the {@code headers.} prefix are
 * ignored for HTTP headers but may be used for other purposes.
 *
 * <p>Example implementation:
 *
 * <pre>
 * public class MyContextProvider implements DynamicContextProvider {
 *   &#64;Override
 *   public Map&lt;String, String&gt; provideContext(String operation, String objectId) {
 *     Map&lt;String, String&gt; context = new HashMap&lt;&gt;();
 *     context.put("headers.Authorization", "Bearer " + getAuthToken());
 *     context.put("headers.X-Request-Id", UUID.randomUUID().toString());
 *     return context;
 *   }
 * }
 * </pre>
 *
 * <p>Usage with DirectoryNamespace:
 *
 * <pre>
 * DynamicContextProvider provider = new MyContextProvider();
 * Map&lt;String, String&gt; properties = Map.of("root", "/path/to/data");
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator, provider);
 * </pre>
 *
 * <p>Usage with RestNamespace:
 *
 * <pre>
 * DynamicContextProvider provider = new MyContextProvider();
 * Map&lt;String, String&gt; properties = Map.of("uri", "https://api.example.com");
 * RestNamespace namespace = new RestNamespace();
 * namespace.initialize(properties, provider);
 * </pre>
 */
public interface DynamicContextProvider {

  /**
   * Provide context for a namespace operation.
   *
   * <p>This method is called synchronously before each namespace operation. Implementations should
   * be thread-safe as multiple operations may be performed concurrently.
   *
   * @param operation The operation name (e.g., "list_tables", "describe_table", "create_namespace")
   * @param objectId The object identifier (namespace or table ID in delimited form, e.g.,
   *     "workspace$table_name")
   * @return Map of context key-value pairs. For HTTP headers, use keys with the "headers." prefix
   *     (e.g., "headers.Authorization"). Return an empty map if no additional context is needed.
   *     Must not return null.
   */
  Map<String, String> provideContext(String operation, String objectId);
}
