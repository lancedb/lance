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

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Builder for writing datasets.
 *
 * <p>This builder provides a fluent API for creating or writing to datasets either directly to a
 * URI or through a LanceNamespace. When using a namespace, the table location and storage options
 * are automatically managed with credential vending support.
 *
 * <p>Example usage with URI and reader:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.write(allocator)
 *     .reader(myReader)
 *     .uri("s3://bucket/table.lance")
 *     .mode(WriteMode.CREATE)
 *     .execute();
 * }</pre>
 *
 * <p>Example usage with namespace client:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.write(allocator)
 *     .reader(myReader)
 *     .namespaceClient(myNamespaceClient)
 *     .tableId(Arrays.asList("my_table"))
 *     .mode(WriteMode.CREATE)
 *     .execute();
 * }</pre>
 */
public class WriteDatasetBuilder {
  private BufferAllocator allocator;
  private ArrowReader reader;
  private ArrowArrayStream stream;
  private String uri;
  private LanceNamespace namespaceClient;
  private List<String> tableId;
  private NamespaceClientTableContext namespaceClientTableContext;
  private WriteParams.WriteMode mode = WriteParams.WriteMode.CREATE;
  private Schema schema;
  private Map<String, String> storageOptions = new HashMap<>();
  private Optional<Integer> maxRowsPerFile = Optional.empty();
  private Optional<Integer> maxRowsPerGroup = Optional.empty();
  private Optional<Long> maxBytesPerFile = Optional.empty();
  private Optional<Boolean> enableStableRowIds = Optional.empty();
  private Optional<String> dataStorageVersion = Optional.empty();
  private Optional<List<BasePath>> initialBases = Optional.empty();
  private Optional<List<String>> targetBases = Optional.empty();
  private Optional<Boolean> allowExternalBlobOutsideBases = Optional.empty();
  private Optional<Long> blobPackFileSizeThreshold = Optional.empty();
  private Session session;

  /** Creates a new builder instance. Package-private, use Dataset.write() instead. */
  WriteDatasetBuilder() {
    // allocator is optional and can be set via allocator() method
  }

  /**
   * Sets the buffer allocator to use for Arrow operations.
   *
   * <p>If not provided, a default RootAllocator will be created automatically.
   *
   * @param allocator The buffer allocator
   * @return this builder instance
   */
  public WriteDatasetBuilder allocator(BufferAllocator allocator) {
    Preconditions.checkNotNull(allocator, "allocator must not be null");
    this.allocator = allocator;
    return this;
  }

  /**
   * Sets the ArrowReader containing the data to write.
   *
   * <p>Either reader() or stream() or schema() (for empty tables) must be provided.
   *
   * @param reader ArrowReader containing the data
   * @return this builder instance
   */
  public WriteDatasetBuilder reader(ArrowReader reader) {
    Preconditions.checkNotNull(reader);
    this.reader = reader;
    return this;
  }

  /**
   * Sets the ArrowArrayStream containing the data to write.
   *
   * <p>Either reader() or stream() or schema() (for empty tables) must be provided.
   *
   * @param stream ArrowArrayStream containing the data
   * @return this builder instance
   */
  public WriteDatasetBuilder stream(ArrowArrayStream stream) {
    Preconditions.checkNotNull(stream);
    this.stream = stream;
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
  public WriteDatasetBuilder uri(String uri) {
    this.uri = uri;
    return this;
  }

  /**
   * Sets the namespace client.
   *
   * <p>Must be used together with tableId(). Either uri() or namespaceClient()+tableId() must be
   * specified, but not both.
   *
   * @param namespaceClient The namespace implementation to use for table operations
   * @return this builder instance
   */
  public WriteDatasetBuilder namespaceClient(LanceNamespace namespaceClient) {
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
  public WriteDatasetBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Sets a cached namespace context from a prior {@code describeTable} or {@code declareTable}
   * call. When provided, skips the namespace call and uses the cached location, storage options,
   * and managed-versioning flag directly.
   *
   * <p>Cannot be used with {@code uri()} or {@code namespaceClient()}/{@code tableId()}.
   *
   * @param context the cached namespace context
   * @return this builder instance
   */
  public WriteDatasetBuilder namespaceClientTableContext(NamespaceClientTableContext context) {
    this.namespaceClientTableContext = context;
    return this;
  }

  /**
   * Sets the write mode.
   *
   * @param mode The write mode (CREATE, APPEND, or OVERWRITE)
   * @return this builder instance
   */
  public WriteDatasetBuilder mode(WriteParams.WriteMode mode) {
    Preconditions.checkNotNull(mode);
    this.mode = mode;
    return this;
  }

  /**
   * Sets the schema for the dataset.
   *
   * <p>If the reader and stream not provided, this is used to create an empty dataset
   *
   * @param schema The dataset schema
   * @return this builder instance
   */
  public WriteDatasetBuilder schema(Schema schema) {
    this.schema = schema;
    return this;
  }

  /**
   * Sets storage options for the dataset.
   *
   * @param storageOptions Storage configuration options
   * @return this builder instance
   */
  public WriteDatasetBuilder storageOptions(Map<String, String> storageOptions) {
    this.storageOptions = new HashMap<>(storageOptions);
    return this;
  }

  /**
   * Sets the maximum number of rows per file.
   *
   * @param maxRowsPerFile Maximum rows per file
   * @return this builder instance
   */
  public WriteDatasetBuilder maxRowsPerFile(int maxRowsPerFile) {
    this.maxRowsPerFile = Optional.of(maxRowsPerFile);
    return this;
  }

  /**
   * Sets the maximum number of rows per group.
   *
   * @param maxRowsPerGroup Maximum rows per group
   * @return this builder instance
   */
  public WriteDatasetBuilder maxRowsPerGroup(int maxRowsPerGroup) {
    this.maxRowsPerGroup = Optional.of(maxRowsPerGroup);
    return this;
  }

  /**
   * Sets the maximum number of bytes per file.
   *
   * @param maxBytesPerFile Maximum bytes per file
   * @return this builder instance
   */
  public WriteDatasetBuilder maxBytesPerFile(long maxBytesPerFile) {
    this.maxBytesPerFile = Optional.of(maxBytesPerFile);
    return this;
  }

  /**
   * Sets whether to enable stable row IDs.
   *
   * @param enableStableRowIds Whether to enable stable row IDs
   * @return this builder instance
   */
  public WriteDatasetBuilder enableStableRowIds(boolean enableStableRowIds) {
    this.enableStableRowIds = Optional.of(enableStableRowIds);
    return this;
  }

  /**
   * Sets the data storage version.
   *
   * @param dataStorageVersion The Lance file version to use (e.g., "legacy", "stable", "2.0")
   * @return this builder instance
   */
  public WriteDatasetBuilder dataStorageVersion(String dataStorageVersion) {
    this.dataStorageVersion = Optional.of(dataStorageVersion);
    return this;
  }

  public WriteDatasetBuilder initialBases(List<BasePath> bases) {
    this.initialBases = Optional.of(bases);
    return this;
  }

  public WriteDatasetBuilder targetBases(List<String> targetBases) {
    this.targetBases = Optional.of(targetBases);
    return this;
  }

  /**
   * Sets whether to allow external blob URIs outside registered base paths.
   *
   * @param allowExternalBlobOutsideBases Whether to allow external blob URIs outside bases
   * @return this builder instance
   */
  public WriteDatasetBuilder allowExternalBlobOutsideBases(boolean allowExternalBlobOutsideBases) {
    this.allowExternalBlobOutsideBases = Optional.of(allowExternalBlobOutsideBases);
    return this;
  }

  /**
   * Sets the maximum size in bytes for blob v2 pack (.blob) sidecar files.
   *
   * <p>When a pack file reaches this size, a new one is started. If not set, defaults to 1 GiB.
   *
   * @param blobPackFileSizeThreshold maximum pack file size in bytes
   * @return this builder instance
   */
  public WriteDatasetBuilder blobPackFileSizeThreshold(long blobPackFileSizeThreshold) {
    this.blobPackFileSizeThreshold = Optional.of(blobPackFileSizeThreshold);
    return this;
  }

  /**
   * Sets the session to share caches with other datasets.
   *
   * <p>Note: For write operations, the session is currently not used during the write itself, but
   * is stored for future use when the resulting dataset needs to be reopened with the same session.
   * This is a placeholder for future session support in write operations.
   *
   * @param session The session to use
   * @return this builder instance
   */
  public WriteDatasetBuilder session(Session session) {
    this.session = session;
    return this;
  }

  /**
   * Executes the write operation and returns the created dataset.
   *
   * <p>If a namespace client is configured via namespaceClient()+tableId(), this automatically
   * handles table creation or retrieval through the namespace client API with credential vending
   * support.
   *
   * @return Dataset
   * @throws IllegalArgumentException if required parameters are missing or invalid
   */
  public Dataset execute() {
    if (allocator == null) {
      allocator = new RootAllocator(Long.MAX_VALUE);
    }

    boolean hasUri = uri != null;
    boolean hasNamespaceClient = namespaceClient != null || tableId != null;
    boolean hasContext = namespaceClientTableContext != null;

    if (hasUri && hasNamespaceClient) {
      throw new IllegalArgumentException(
          "Cannot specify both uri() and namespaceClient()+tableId().");
    }
    if (hasContext && !hasNamespaceClient) {
      throw new IllegalArgumentException(
          "namespaceClientTableContext() requires namespaceClient() and tableId().");
    }
    if (!hasUri && !hasNamespaceClient) {
      throw new IllegalArgumentException(
          "Either uri() or namespaceClient()+tableId() must be called.");
    }

    if (hasNamespaceClient) {
      if (namespaceClient == null) {
        throw new IllegalArgumentException(
            "tableId() is set but namespaceClient() is missing. Both must be provided together.");
      }
      if (tableId == null) {
        throw new IllegalArgumentException(
            "namespaceClient() is set but tableId() is missing. Both must be provided together.");
      }
    }

    int dataSourceCount = 0;
    if (reader != null) dataSourceCount++;
    if (stream != null) dataSourceCount++;
    if (schema != null && reader == null && stream == null) dataSourceCount++;

    if (dataSourceCount == 0) {
      throw new IllegalArgumentException(
          "Must provide data via reader(), stream(), or schema() (for empty tables).");
    }
    if (dataSourceCount > 1) {
      throw new IllegalArgumentException(
          "Cannot specify multiple data sources. "
              + "Use only one of: reader(), stream(), or schema().");
    }

    if (hasNamespaceClient) {
      return executeWithNamespaceClient();
    }

    return executeWithUri();
  }

  private Dataset executeWithNamespaceClient() {
    WriteParams.Builder paramsBuilder =
        new WriteParams.Builder().withMode(mode).withStorageOptions(storageOptions);

    maxRowsPerFile.ifPresent(paramsBuilder::withMaxRowsPerFile);
    maxRowsPerGroup.ifPresent(paramsBuilder::withMaxRowsPerGroup);
    maxBytesPerFile.ifPresent(paramsBuilder::withMaxBytesPerFile);
    enableStableRowIds.ifPresent(paramsBuilder::withEnableStableRowIds);
    dataStorageVersion.ifPresent(paramsBuilder::withDataStorageVersion);
    initialBases.ifPresent(paramsBuilder::withInitialBases);
    targetBases.ifPresent(paramsBuilder::withTargetBases);
    allowExternalBlobOutsideBases.ifPresent(paramsBuilder::withAllowExternalBlobOutsideBases);
    blobPackFileSizeThreshold.ifPresent(paramsBuilder::withBlobPackFileSizeThreshold);

    WriteParams params = paramsBuilder.build();

    return createDatasetWithStreamAndNamespaceClient(
        null, params, namespaceClient, tableId, namespaceClientTableContext);
  }

  private Dataset executeWithUri() {
    WriteParams.Builder paramsBuilder =
        new WriteParams.Builder().withMode(mode).withStorageOptions(storageOptions);

    maxRowsPerFile.ifPresent(paramsBuilder::withMaxRowsPerFile);
    maxRowsPerGroup.ifPresent(paramsBuilder::withMaxRowsPerGroup);
    maxBytesPerFile.ifPresent(paramsBuilder::withMaxBytesPerFile);
    enableStableRowIds.ifPresent(paramsBuilder::withEnableStableRowIds);
    dataStorageVersion.ifPresent(paramsBuilder::withDataStorageVersion);
    initialBases.ifPresent(paramsBuilder::withInitialBases);
    targetBases.ifPresent(paramsBuilder::withTargetBases);
    allowExternalBlobOutsideBases.ifPresent(paramsBuilder::withAllowExternalBlobOutsideBases);
    blobPackFileSizeThreshold.ifPresent(paramsBuilder::withBlobPackFileSizeThreshold);

    WriteParams params = paramsBuilder.build();

    return createDatasetWithStreamAndNamespaceClient(uri, params, null, null, null);
  }

  private Dataset createDatasetWithStreamAndNamespaceClient(
      String path,
      WriteParams params,
      LanceNamespace namespaceClient,
      List<String> tableId,
      NamespaceClientTableContext namespaceClientTableContext) {
    if (stream != null) {
      return Dataset.create(
          allocator, stream, path, params, namespaceClient, tableId, namespaceClientTableContext);
    }

    if (reader != null) {
      try (ArrowArrayStream tempStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, tempStream);
        return Dataset.create(
            allocator,
            tempStream,
            path,
            params,
            namespaceClient,
            tableId,
            namespaceClientTableContext);
      }
    }

    if (schema != null) {
      return Dataset.create(allocator, path, schema, params);
    }

    throw new IllegalStateException("No data source provided");
  }
}
