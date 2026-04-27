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
package org.lance;

import org.lance.namespace.LanceNamespace;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;

import java.util.List;

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
 * <p>Example usage with namespace client:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.open()
 *     .namespaceClient(myNamespaceClient)
 *     .tableId(Arrays.asList("my_table"))
 *     .build();
 * }</pre>
 */
public class OpenDatasetBuilder {
  private BufferAllocator allocator;
  private boolean selfManagedAllocator = false;
  private String uri;
  private LanceNamespace namespaceClient;
  private List<String> tableId;
  private NamespaceClientTableContext namespaceClientTableContext;
  private ReadOptions options = new ReadOptions.Builder().build();
  private Session session;

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
   * <p>Either uri() or namespaceClient()+tableId() must be specified, but not both.
   *
   * @param uri The dataset URI (e.g., "s3://bucket/table.lance" or "file:///path/to/table.lance")
   * @return this builder instance
   */
  public OpenDatasetBuilder uri(String uri) {
    this.uri = uri;
    return this;
  }

  /**
   * Sets the namespace client.
   *
   * <p>Must be used together with tableId(). Either uri() or namespaceClient()+tableId() must be
   * specified, but not both.
   *
   * @param namespaceClient The namespace implementation to fetch table info from
   * @return this builder instance
   */
  public OpenDatasetBuilder namespaceClient(LanceNamespace namespaceClient) {
    this.namespaceClient = namespaceClient;
    return this;
  }

  /**
   * Sets the table identifier.
   *
   * <p>Must be used together with namespaceClient(). Either uri() or namespaceClient()+tableId()
   * must be specified, but not both.
   *
   * @param tableId The table identifier (e.g., Arrays.asList("my_table"))
   * @return this builder instance
   */
  public OpenDatasetBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Sets a cached namespace context from a prior {@code describeTable} or {@code declareTable}
   * call. When provided, skips the namespace call and uses the cached location, storage options,
   * and managed-versioning flag directly.
   *
   * <p>Cannot be used with {@code uri()}. Must be used together with {@code namespaceClient()} and
   * {@code tableId()}.
   *
   * @param context the cached namespace context
   * @return this builder instance
   */
  public OpenDatasetBuilder namespaceClientTableContext(NamespaceClientTableContext context) {
    this.namespaceClientTableContext = context;
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
   * Sets the session to share caches between multiple datasets.
   *
   * <p>When a session is provided, the index and metadata caches from the session will be used
   * instead of creating new caches. This can improve cache hit rates when opening multiple related
   * datasets.
   *
   * <p>Note: When a session is provided, the indexCacheSizeBytes and metadataCacheSizeBytes
   * settings in ReadOptions are ignored because the session's caches are used instead.
   *
   * @param session The session to use
   * @return this builder instance
   */
  public OpenDatasetBuilder session(Session session) {
    this.session = session;
    return this;
  }

  /**
   * Opens the dataset with the configured parameters.
   *
   * <p>If a namespace client is configured, this automatically fetches the table location and
   * storage options from the namespace client via describeTable().
   *
   * @return Dataset
   * @throws IllegalArgumentException if required parameters are missing or invalid
   */
  public Dataset build() {
    boolean hasUri = uri != null;
    boolean hasNamespaceFields = namespaceClient != null || tableId != null;
    boolean hasCompleteNamespace = namespaceClient != null && tableId != null;
    boolean hasContext = namespaceClientTableContext != null;

    if (hasUri && hasNamespaceFields) {
      throw new IllegalArgumentException("Cannot specify both uri and namespaceClient+tableId.");
    }
    if (hasContext && !hasCompleteNamespace) {
      throw new IllegalArgumentException(
          "namespaceClientTableContext requires namespaceClient and tableId.");
    }
    if (!hasUri && !hasNamespaceFields) {
      throw new IllegalArgumentException("Either uri or namespaceClient+tableId must be provided.");
    }

    if (hasNamespaceFields && !hasCompleteNamespace) {
      if (namespaceClient == null) {
        throw new IllegalArgumentException(
            "tableId is set but namespaceClient is missing. Both must be provided together.");
      } else {
        throw new IllegalArgumentException(
            "namespaceClient is set but tableId is missing. Both must be provided together.");
      }
    }

    Preconditions.checkNotNull(options, "options must be set");

    if (allocator == null) {
      allocator = new RootAllocator(Long.MAX_VALUE);
      selfManagedAllocator = true;
    }

    if (hasCompleteNamespace) {
      return buildFromNamespaceClient();
    }

    return Dataset.open(allocator, selfManagedAllocator, uri, options, session);
  }

  private Dataset buildFromNamespaceClient() {
    return Dataset.open(
        allocator,
        selfManagedAllocator,
        null,
        options,
        session,
        namespaceClient,
        tableId,
        namespaceClientTableContext);
  }
}
