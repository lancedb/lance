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
import org.lance.namespace.model.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.arrow.memory.BufferAllocator;

import java.io.Closeable;
import java.util.Map;

/**
 * RestNamespace implementation that provides Lance namespace functionality via REST API endpoints.
 *
 * <p>This class wraps the native Rust implementation and provides a Java interface that implements
 * the LanceNamespace interface from lance-namespace-core.
 *
 * <p>Configuration properties:
 *
 * <ul>
 *   <li>uri (required): REST API endpoint URL
 *   <li>delimiter (optional): Namespace delimiter (default: "$")
 *   <li>header.* (optional): HTTP headers (e.g., header.Authorization=Bearer token)
 *   <li>tls.cert_file (optional): Path to client certificate file
 *   <li>tls.key_file (optional): Path to client key file
 *   <li>tls.ssl_ca_cert (optional): Path to CA certificate file
 *   <li>tls.assert_hostname (optional): "true" or "false" (default: true)
 * </ul>
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("uri", "https://api.example.com");
 * properties.put("delimiter", ".");
 * properties.put("header.Authorization", "Bearer my-token");
 *
 * RestNamespace namespace = new RestNamespace();
 * namespace.initialize(properties, allocator);
 *
 * // Use namespace...
 * ListTablesResponse tables = namespace.listTables(request);
 *
 * // Clean up
 * namespace.close();
 * }</pre>
 */
public class RestNamespace implements LanceNamespace, Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  private long nativeRestNamespaceHandle;
  private BufferAllocator allocator;

  /** Creates a new RestNamespace. Must call initialize() before use. */
  public RestNamespace() {}

  @Override
  public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
    if (this.nativeRestNamespaceHandle != 0) {
      throw new IllegalStateException("RestNamespace already initialized");
    }
    this.allocator = allocator;
    this.nativeRestNamespaceHandle = createNative(configProperties);
  }

  @Override
  public String namespaceId() {
    ensureInitialized();
    return namespaceIdNative(nativeRestNamespaceHandle);
  }

  @Override
  public ListNamespacesResponse listNamespaces(ListNamespacesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listNamespacesNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, ListNamespacesResponse.class);
  }

  @Override
  public DescribeNamespaceResponse describeNamespace(DescribeNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeNamespaceNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeNamespaceResponse.class);
  }

  @Override
  public CreateNamespaceResponse createNamespace(CreateNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createNamespaceNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateNamespaceResponse.class);
  }

  @Override
  public DropNamespaceResponse dropNamespace(DropNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropNamespaceNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DropNamespaceResponse.class);
  }

  @Override
  public void namespaceExists(NamespaceExistsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    namespaceExistsNative(nativeRestNamespaceHandle, requestJson);
  }

  @Override
  public ListTablesResponse listTables(ListTablesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTablesNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTablesResponse.class);
  }

  @Override
  public DescribeTableResponse describeTable(DescribeTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableResponse.class);
  }

  @Override
  public RegisterTableResponse registerTable(RegisterTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = registerTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, RegisterTableResponse.class);
  }

  @Override
  public void tableExists(TableExistsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    tableExistsNative(nativeRestNamespaceHandle, requestJson);
  }

  @Override
  public DropTableResponse dropTable(DropTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DropTableResponse.class);
  }

  @Override
  public DeregisterTableResponse deregisterTable(DeregisterTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deregisterTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DeregisterTableResponse.class);
  }

  @Override
  public Long countTableRows(CountTableRowsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    return countTableRowsNative(nativeRestNamespaceHandle, requestJson);
  }

  @Override
  public CreateTableResponse createTable(CreateTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableNative(nativeRestNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, CreateTableResponse.class);
  }

  @Override
  public CreateEmptyTableResponse createEmptyTable(CreateEmptyTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createEmptyTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateEmptyTableResponse.class);
  }

  @Override
  public DeclareTableResponse declareTable(DeclareTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = declareTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DeclareTableResponse.class);
  }

  @Override
  public InsertIntoTableResponse insertIntoTable(
      InsertIntoTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        insertIntoTableNative(nativeRestNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, InsertIntoTableResponse.class);
  }

  @Override
  public MergeInsertIntoTableResponse mergeInsertIntoTable(
      MergeInsertIntoTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        mergeInsertIntoTableNative(nativeRestNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, MergeInsertIntoTableResponse.class);
  }

  @Override
  public UpdateTableResponse updateTable(UpdateTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = updateTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, UpdateTableResponse.class);
  }

  @Override
  public DeleteFromTableResponse deleteFromTable(DeleteFromTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deleteFromTableNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DeleteFromTableResponse.class);
  }

  @Override
  public byte[] queryTable(QueryTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    return queryTableNative(nativeRestNamespaceHandle, requestJson);
  }

  @Override
  public CreateTableIndexResponse createTableIndex(CreateTableIndexRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableIndexNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateTableIndexResponse.class);
  }

  @Override
  public ListTableIndicesResponse listTableIndices(ListTableIndicesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTableIndicesNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTableIndicesResponse.class);
  }

  @Override
  public DescribeTableIndexStatsResponse describeTableIndexStats(
      DescribeTableIndexStatsRequest request, String indexName) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTableIndexStatsNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableIndexStatsResponse.class);
  }

  @Override
  public DescribeTransactionResponse describeTransaction(DescribeTransactionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTransactionNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTransactionResponse.class);
  }

  @Override
  public AlterTransactionResponse alterTransaction(AlterTransactionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = alterTransactionNative(nativeRestNamespaceHandle, requestJson);
    return fromJson(responseJson, AlterTransactionResponse.class);
  }

  @Override
  public void close() {
    if (nativeRestNamespaceHandle != 0) {
      releaseNative(nativeRestNamespaceHandle);
      nativeRestNamespaceHandle = 0;
    }
  }

  private void ensureInitialized() {
    if (nativeRestNamespaceHandle == 0) {
      throw new IllegalStateException("RestNamespace not initialized. Call initialize() first.");
    }
  }

  private static String toJson(Object obj) {
    try {
      return OBJECT_MAPPER.writeValueAsString(obj);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize request to JSON", e);
    }
  }

  private static <T> T fromJson(String json, Class<T> clazz) {
    try {
      return OBJECT_MAPPER.readValue(json, clazz);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to deserialize response from JSON", e);
    }
  }

  // Native methods
  private native long createNative(Map<String, String> properties);

  private native void releaseNative(long handle);

  private native String namespaceIdNative(long handle);

  private native String listNamespacesNative(long handle, String requestJson);

  private native String describeNamespaceNative(long handle, String requestJson);

  private native String createNamespaceNative(long handle, String requestJson);

  private native String dropNamespaceNative(long handle, String requestJson);

  private native void namespaceExistsNative(long handle, String requestJson);

  private native String listTablesNative(long handle, String requestJson);

  private native String describeTableNative(long handle, String requestJson);

  private native String registerTableNative(long handle, String requestJson);

  private native void tableExistsNative(long handle, String requestJson);

  private native String dropTableNative(long handle, String requestJson);

  private native String deregisterTableNative(long handle, String requestJson);

  private native long countTableRowsNative(long handle, String requestJson);

  private native String createTableNative(long handle, String requestJson, byte[] requestData);

  private native String createEmptyTableNative(long handle, String requestJson);

  private native String declareTableNative(long handle, String requestJson);

  private native String insertIntoTableNative(long handle, String requestJson, byte[] requestData);

  private native String mergeInsertIntoTableNative(
      long handle, String requestJson, byte[] requestData);

  private native String updateTableNative(long handle, String requestJson);

  private native String deleteFromTableNative(long handle, String requestJson);

  private native byte[] queryTableNative(long handle, String requestJson);

  private native String createTableIndexNative(long handle, String requestJson);

  private native String listTableIndicesNative(long handle, String requestJson);

  private native String describeTableIndexStatsNative(long handle, String requestJson);

  private native String describeTransactionNative(long handle, String requestJson);

  private native String alterTransactionNative(long handle, String requestJson);
}
