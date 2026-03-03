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

import org.lance.io.StorageOptionsProvider;
import org.lance.namespace.LanceNamespace;
import org.lance.namespace.LanceNamespaceStorageOptionsProvider;
import org.lance.namespace.model.CreateEmptyTableRequest;
import org.lance.namespace.model.CreateEmptyTableResponse;
import org.lance.namespace.model.DeclareTableRequest;
import org.lance.namespace.model.DeclareTableResponse;
import org.lance.namespace.model.DescribeTableRequest;
import org.lance.namespace.model.DescribeTableResponse;

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
 * <p>Example usage with namespace:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.write(allocator)
 *     .reader(myReader)
 *     .namespace(myNamespace)
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
  private LanceNamespace namespace;
  private List<String> tableId;
  private WriteParams.WriteMode mode = WriteParams.WriteMode.CREATE;
  private Schema schema;
  private Map<String, String> storageOptions = new HashMap<>();
  private boolean ignoreNamespaceStorageOptions = false;
  private Optional<Integer> maxRowsPerFile = Optional.empty();
  private Optional<Integer> maxRowsPerGroup = Optional.empty();
  private Optional<Long> maxBytesPerFile = Optional.empty();
  private Optional<Boolean> enableStableRowIds = Optional.empty();
  private Optional<String> dataStorageVersion = Optional.empty();
  private Optional<List<BasePath>> initialBases = Optional.empty();
  private Optional<List<String>> targetBases = Optional.empty();
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
   * <p>Either uri() or namespace()+tableId() must be specified, but not both.
   *
   * @param uri The dataset URI (e.g., "s3://bucket/table.lance" or "file:///path/to/table.lance")
   * @return this builder instance
   */
  public WriteDatasetBuilder uri(String uri) {
    this.uri = uri;
    return this;
  }

  /**
   * Sets the namespace.
   *
   * <p>Must be used together with tableId(). Either uri() or namespace()+tableId() must be
   * specified, but not both.
   *
   * @param namespace The namespace implementation to use for table operations
   * @return this builder instance
   */
  public WriteDatasetBuilder namespace(LanceNamespace namespace) {
    this.namespace = namespace;
    return this;
  }

  /**
   * Sets the table identifier.
   *
   * <p>Must be used together with namespace(). Either uri() or namespace()+tableId() must be
   * specified, but not both.
   *
   * @param tableId The table identifier (e.g., Arrays.asList("my_table"))
   * @return this builder instance
   */
  public WriteDatasetBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
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
   * Sets whether to ignore storage options from the namespace's describeTable() or
   * createEmptyTable().
   *
   * @param ignoreNamespaceStorageOptions If true, storage options returned from namespace will be
   *     ignored
   * @return this builder instance
   */
  public WriteDatasetBuilder ignoreNamespaceStorageOptions(boolean ignoreNamespaceStorageOptions) {
    this.ignoreNamespaceStorageOptions = ignoreNamespaceStorageOptions;
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
   * <p>If a namespace is configured via namespace()+tableId(), this automatically handles table
   * creation or retrieval through the namespace API with credential vending support.
   *
   * @return Dataset
   * @throws IllegalArgumentException if required parameters are missing or invalid
   */
  public Dataset execute() {
    // Auto-create allocator if not provided
    if (allocator == null) {
      allocator = new RootAllocator(Long.MAX_VALUE);
    }

    // Validate that exactly one of uri or namespace is provided
    boolean hasUri = uri != null;
    boolean hasNamespace = namespace != null && tableId != null;

    if (hasUri && hasNamespace) {
      throw new IllegalArgumentException(
          "Cannot specify both uri() and namespace()+tableId(). Use one or the other.");
    }
    if (!hasUri && !hasNamespace) {
      if (namespace != null) {
        throw new IllegalArgumentException(
            "namespace() is set but tableId() is missing. Both must be provided together.");
      } else if (tableId != null) {
        throw new IllegalArgumentException(
            "tableId() is set but namespace() is missing. Both must be provided together.");
      } else {
        throw new IllegalArgumentException("Either uri() or namespace()+tableId() must be called.");
      }
    }

    // Validate data source - exactly one of reader, stream, or schema must be provided
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

    // Handle namespace-based writing
    if (hasNamespace) {
      return executeWithNamespace();
    }

    // Handle URI-based writing
    return executeWithUri();
  }

  private Dataset executeWithNamespace() {
    String tableUri;
    Map<String, String> namespaceStorageOptions = null;
    boolean managedVersioning = false;

    // Mode-specific namespace operations
    if (mode == WriteParams.WriteMode.CREATE) {
      // Try declareTable first, fall back to deprecated createEmptyTable
      // for backward compatibility with older namespace implementations.
      // createEmptyTable support will be removed in 3.0.0.
      String location;
      Map<String, String> responseStorageOptions;

      try {
        DeclareTableRequest declareRequest = new DeclareTableRequest();
        declareRequest.setId(tableId);
        DeclareTableResponse declareResponse = namespace.declareTable(declareRequest);
        location = declareResponse.getLocation();
        responseStorageOptions = declareResponse.getStorageOptions();
        managedVersioning = Boolean.TRUE.equals(declareResponse.getManagedVersioning());
      } catch (UnsupportedOperationException e) {
        // Fall back to deprecated createEmptyTable
        // Note: createEmptyTable doesn't support managedVersioning
        CreateEmptyTableRequest fallbackRequest = new CreateEmptyTableRequest();
        fallbackRequest.setId(tableId);
        CreateEmptyTableResponse fallbackResponse = namespace.createEmptyTable(fallbackRequest);
        location = fallbackResponse.getLocation();
        responseStorageOptions = fallbackResponse.getStorageOptions();
        managedVersioning = false;
      }

      tableUri = location;
      if (tableUri == null || tableUri.isEmpty()) {
        throw new IllegalArgumentException("Namespace did not return a table location");
      }

      namespaceStorageOptions = ignoreNamespaceStorageOptions ? null : responseStorageOptions;
    } else {
      // For APPEND/OVERWRITE modes, call namespace.describeTable()
      DescribeTableRequest request = new DescribeTableRequest();
      request.setId(tableId);

      DescribeTableResponse response = namespace.describeTable(request);

      tableUri = response.getLocation();
      if (tableUri == null || tableUri.isEmpty()) {
        throw new IllegalArgumentException("Namespace did not return a table location");
      }

      namespaceStorageOptions = ignoreNamespaceStorageOptions ? null : response.getStorageOptions();
      managedVersioning = Boolean.TRUE.equals(response.getManagedVersioning());
    }

    // Merge storage options (namespace options + user options, with namespace taking precedence)
    Map<String, String> mergedStorageOptions = new HashMap<>(storageOptions);
    if (namespaceStorageOptions != null && !namespaceStorageOptions.isEmpty()) {
      mergedStorageOptions.putAll(namespaceStorageOptions);
    }

    // Build WriteParams with merged storage options
    WriteParams.Builder paramsBuilder =
        new WriteParams.Builder().withMode(mode).withStorageOptions(mergedStorageOptions);

    maxRowsPerFile.ifPresent(paramsBuilder::withMaxRowsPerFile);
    maxRowsPerGroup.ifPresent(paramsBuilder::withMaxRowsPerGroup);
    maxBytesPerFile.ifPresent(paramsBuilder::withMaxBytesPerFile);
    enableStableRowIds.ifPresent(paramsBuilder::withEnableStableRowIds);
    dataStorageVersion.ifPresent(paramsBuilder::withDataStorageVersion);

    initialBases.ifPresent(paramsBuilder::withInitialBases);
    targetBases.ifPresent(paramsBuilder::withTargetBases);

    WriteParams params = paramsBuilder.build();

    // Create storage options provider for credential refresh during long-running writes
    StorageOptionsProvider storageOptionsProvider =
        ignoreNamespaceStorageOptions
            ? null
            : new LanceNamespaceStorageOptionsProvider(namespace, tableId);

    // Only use namespace for commit handling if managedVersioning is enabled
    if (managedVersioning) {
      return createDatasetWithStreamAndNamespace(
          tableUri, params, storageOptionsProvider, namespace, tableId);
    } else {
      return createDatasetWithStream(tableUri, params, storageOptionsProvider);
    }
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

    WriteParams params = paramsBuilder.build();

    return createDatasetWithStream(uri, params, null);
  }

  private Dataset createDatasetWithStream(
      String path, WriteParams params, StorageOptionsProvider storageOptionsProvider) {
    // If stream is directly provided, use it
    if (stream != null) {
      return Dataset.create(allocator, stream, path, params, storageOptionsProvider);
    }

    // If reader is provided, convert to stream
    if (reader != null) {
      try (ArrowArrayStream tempStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, tempStream);
        return Dataset.create(allocator, tempStream, path, params, storageOptionsProvider);
      }
    }

    // If only schema is provided (empty table), use Dataset.create with schema
    if (schema != null) {
      return Dataset.create(allocator, path, schema, params);
    }

    throw new IllegalStateException("No data source provided");
  }

  private Dataset createDatasetWithStreamAndNamespace(
      String path,
      WriteParams params,
      StorageOptionsProvider storageOptionsProvider,
      LanceNamespace namespace,
      List<String> tableId) {
    // If stream is directly provided, use it
    if (stream != null) {
      return Dataset.create(
          allocator, stream, path, params, storageOptionsProvider, namespace, tableId);
    }

    // If reader is provided, convert to stream
    if (reader != null) {
      try (ArrowArrayStream tempStream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, tempStream);
        return Dataset.create(
            allocator, tempStream, path, params, storageOptionsProvider, namespace, tableId);
      }
    }

    // If only schema is provided (empty table), use Dataset.create with schema
    // Note: Schema-only creation doesn't support namespace-based commit handling
    if (schema != null) {
      return Dataset.create(allocator, path, schema, params);
    }

    throw new IllegalStateException("No data source provided");
  }
}
